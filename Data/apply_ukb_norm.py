"""Apply the published UKB-derived normalization factors to a user matrix.

Z-scores a private cohort's feature matrix so it can be fed to the released
MetAgeFormer models (input layer `'Z-score normalized'`, see
`Data/ukb_norm_factors/readme.md` and `Docs/SKILL_CUSTOM_COHORT.md`).

Two modes select which factor set is applied (both shipped in this repo):

    nmr107   the 107-feature UKB NMR panel (order = Model_Weights/MetAgeFormer/vocab.txt)
    blood14  the 14-feature blood panel used by the Lightweight model

CSV mode (numpy/pandas only):

    python Data/apply_ukb_norm.py --input my_cohort.csv --mode nmr107 --out my_cohort_z.csv

h5ad mode (needs anndata; rebuilds the file with the full factor panel and the
`'Z-score normalized'` layer):

    python Data/apply_ukb_norm.py --input my_cohort.csv --mode blood14 \
        --h5ad my_cohort.h5ad --h5ad-out my_cohort_z.h5ad

Rules:
- Feature names are matched exact -> case-insensitive -> punctuation-insensitive
  (same convention as the ADNI preprocessing). Unmatched input columns and
  missing factor features are reported as warnings.
- Output columns follow the factor order. Missing features become NaN columns
  (NaN = missing is the model convention; values are NEVER zero-filled).
- This script does NOT convert units. It prints the expected unit per matched
  feature so you can check your input; unit-conversion recipes live in
  `Docs/SKILL_CUSTOM_COHORT.md` and `Docs/SKILL_DATA_PREPROCESSING.md`.
- h5ad mode writes a NEW file with the full panel; the input file is never
  modified.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

FACTOR_DIR = Path(__file__).resolve().parent / "ukb_norm_factors"
MODE_FILES = {"nmr107": "nmr107_factors.csv", "blood14": "blood14_factors.csv"}
LAYER = "Z-score normalized"


def _norm_key(name: str) -> str:
    """Normalize a feature name for matching: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_factors(mode: str) -> pd.DataFrame:
    """Load the factor table for `mode` (error with a pointer if missing)."""
    if mode not in MODE_FILES:
        raise SystemExit(
            f"error: unknown --mode '{mode}' (expected nmr107 or blood14)"
        )
    path = FACTOR_DIR / MODE_FILES[mode]
    if not path.exists():
        raise SystemExit(
            f"error: factor file not found: {path}\n"
            f"See Data/ukb_norm_factors/readme.md — this file is part of the repository."
        )
    factors = pd.read_csv(path)
    required = {"feature", "z_mean", "z_std"}
    if not required <= set(factors.columns):
        raise SystemExit(
            f"error: {path} is missing columns {sorted(required - set(factors.columns))}"
        )
    return factors


def match_columns(input_names: list, factors: pd.DataFrame):
    """Match input column names to factor features (exact -> normalized key).

    Returns (order, matched_pairs, missing_factors): `order` is the factor
    rows (in factor order) that have a match; `matched_pairs` maps factor
    feature -> input column name; `missing_factors` lists factor features
    with no match.
    """
    norm_map = {_norm_key(f): f for f in factors["feature"]}
    if len(norm_map) != len(factors):
        dup = factors["feature"][factors["feature"].map(_norm_key).duplicated(keep=False)]
        raise SystemExit(f"error: factor file has ambiguous feature names: {list(dup)}")

    factor_names = set(factors["feature"])
    matched_pairs = {}
    unmatched_input = []
    for col in input_names:
        if col in factor_names:  # exact
            matched_pairs[col] = col
            continue
        key = _norm_key(col)
        if key in norm_map and norm_map[key] not in matched_pairs:
            matched_pairs[norm_map[key]] = col
        else:
            unmatched_input.append(col)

    missing_factors = [f for f in factors["feature"] if f not in matched_pairs]
    if not matched_pairs:
        raise SystemExit(
            "error: no input columns matched any factor feature.\n"
            "Check that your feature names follow Model_Weights/MetAgeFormer/vocab.txt\n"
            "(nmr107) or the 14 blood-panel names in "
            "Data/ukb_norm_factors/blood14_factors.csv."
        )
    if unmatched_input:
        print(
            f"warning: {len(unmatched_input)} input columns do not match any factor "
            f"feature and are dropped: "
            f"{unmatched_input[:10]}{' ...' if len(unmatched_input) > 10 else ''}"
        )
    if missing_factors:
        print(
            f"warning: {len(missing_factors)} factor features are absent from the input; "
            f"they become NaN columns (masked at inference): "
            f"{missing_factors[:10]}{' ...' if len(missing_factors) > 10 else ''}"
        )
    order = factors[factors["feature"].isin(matched_pairs)].copy()
    return order, matched_pairs, missing_factors


def zscore_matrix(df: pd.DataFrame, matched_pairs: dict, order: pd.DataFrame) -> pd.DataFrame:
    """Return z-scored columns in factor order, NaN where input was NaN."""
    means = order.set_index("feature")["z_mean"]
    stds = order.set_index("feature")["z_std"]
    out = pd.DataFrame(index=df.index)
    for feat in order["feature"]:
        col = matched_pairs[feat]
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        out[feat] = (x - means[feat]) / stds[feat]
    return out


def print_expected_units(factors: pd.DataFrame, features: list) -> None:
    """Print the expected unit per feature (no conversion is performed)."""
    if "unit" not in factors.columns:
        return
    units = factors.set_index("feature")["unit"]
    print("expected input units (no conversion performed — check yours match):")
    for feat in features:
        print(f"  {feat}: {units.get(feat, '?')}")


def run_csv(args, factors: pd.DataFrame) -> None:
    """CSV/TSV mode: z-score a matrix file, write a new CSV.

    Input convention: features as ROWS (one feature-name column named
    `feature`, `--features-column`, or the index), samples as columns.
    Output: samples as rows, features (factor order) as columns.
    """
    sep = args.sep if args.sep is not None else (
        "\t" if args.input.endswith((".tsv", ".txt")) else ","
    )
    try:
        df = pd.read_csv(args.input, sep=sep)
    except FileNotFoundError:
        raise SystemExit(f"error: input file not found: {args.input}")

    feat_col = args.features_column
    if feat_col is None and "feature" in df.columns:
        feat_col = "feature"
    if feat_col is not None:
        if feat_col not in df.columns:
            raise SystemExit(
                f"error: --features-column '{feat_col}' not found in {args.input}"
            )
        names = df[feat_col].astype(str).tolist()
        if len(set(names)) != len(names):
            raise SystemExit(
                f"error: duplicate feature names in column '{feat_col}'"
            )
        data = df.drop(columns=[feat_col])
        data.index = names
    else:
        is_default_index = (
            df.index.name is None and list(df.index) == list(range(len(df)))
        )
        if is_default_index:
            raise SystemExit(
                "error: no feature-name column found. Pass --features-column <col>, "
                "use a column named 'feature', or read with index_col=0."
            )
        data = df
        names = [str(i) for i in df.index]

    # wide orientation: samples as rows, feature names as columns
    wide = data.T
    wide.columns = [str(c) for c in wide.columns]

    order, pairs, missing = match_columns(list(wide.columns), factors)
    df = wide
    z = zscore_matrix(df, pairs, order)
    if args.fill_missing and missing:
        for feat in missing:
            z[feat] = np.nan
        z = z[factors["feature"]]  # full panel, factor order

    out_path = args.out or str(
        Path(args.input).with_name(Path(args.input).stem + "_zscored.csv")
    )
    z.to_csv(out_path, index=False, float_format="%.8g")
    print(f"wrote {out_path}  (shape {z.shape}; columns follow factor order)")
    print_expected_units(factors, list(z.columns))


def run_h5ad(args, factors: pd.DataFrame) -> None:
    """h5ad mode: rebuild the file with the full factor panel + z-scored layer."""
    try:
        import anndata as ad
    except ImportError:
        raise SystemExit("error: --h5ad mode requires anndata (pip install anndata)")
    if not args.h5ad_out:
        raise SystemExit(
            "error: --h5ad mode requires --h5ad-out <new.h5ad> (input is never overwritten)"
        )
    if Path(args.h5ad).resolve() == Path(args.h5ad_out).resolve():
        raise SystemExit(
            "error: --h5ad-out must differ from --h5ad (input is never overwritten)"
        )

    adata = ad.read_h5ad(args.h5ad)
    input_names = [str(n) for n in adata.var_names]
    order, pairs, missing = match_columns(input_names, factors)
    var_idx = {name: i for i, name in enumerate(input_names)}

    # raw values in factor order; missing features -> all-NaN columns
    n = adata.n_obs
    full = np.full((n, len(factors)), np.nan, dtype=np.float64)
    for i, feat in enumerate(factors["feature"]):
        if feat in pairs:
            col = pairs[feat]
            col_x = adata.X[:, var_idx[col]]
            if hasattr(col_x, "toarray"):  # sparse X
                col_x = col_x.toarray()
            full[:, i] = np.asarray(col_x, dtype=np.float64).reshape(-1)

    means = factors["z_mean"].to_numpy()
    stds = factors["z_std"].to_numpy()
    z = (full - means) / stds

    # rebuild var with factor metadata
    new_var = factors.set_index("feature").rename(
        columns={"z_mean": "Z-score mean", "z_std": "Z-score std", "unit": "Unit"}
    )
    keep_cols = [c for c in new_var.columns if c in ("display_name", "Unit")]
    var = new_var[["Z-score mean", "Z-score std"] + keep_cols].copy()

    out = ad.AnnData(X=full, var=var)
    out.var_names = factors["feature"].tolist()
    out.layers[LAYER] = z.astype(np.float32)
    out.obs = adata.obs.copy()
    out.obsm = dict(adata.obsm)
    out.uns = dict(adata.uns)
    out.write_h5ad(args.h5ad_out)
    print(
        f"wrote {args.h5ad_out}  (layer '{LAYER}', float32; "
        f"{len(missing)} missing feature(s) left as NaN)"
    )
    if missing:
        print(f"missing features (NaN tokens, masked at inference): {missing}")
    print_expected_units(factors, factors["feature"].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None,
                        help="CSV mode: input matrix file (.csv/.tsv). Not used with --h5ad.")
    parser.add_argument("--mode", required=True, choices=sorted(MODE_FILES),
                        help="factor set to apply")
    parser.add_argument("--features-column", default=None,
                        help="CSV mode: column holding feature names "
                             "(default: a column named 'feature', else the index)")
    parser.add_argument("--sep", default=None,
                        help="CSV delimiter (default: auto by extension, ',' otherwise)")
    parser.add_argument("--out", default=None,
                        help="CSV output path (default: <input>_zscored.csv)")
    parser.add_argument("--fill-missing", action="store_true",
                        help="CSV mode: emit ALL factor features; absent ones become NaN columns")
    parser.add_argument("--h5ad", default=None,
                        help="apply to this h5ad's var_names and rebuild with the full panel "
                             "(requires --h5ad-out)")
    parser.add_argument("--h5ad-out", default=None,
                        help="output h5ad path (input is never overwritten)")
    args = parser.parse_args()

    factors = load_factors(args.mode)
    if args.h5ad:
        run_h5ad(args, factors)
    elif args.input:
        run_csv(args, factors)
    else:
        parser.error("either --input <file> (CSV mode) or --h5ad <file> is required")

    print(
        "Reminder: z-scores are only valid if input values are in the expected units "
        "(see the unit table above). Never zero-fill missing values — NaN means missing."
    )


if __name__ == "__main__":
    main()
