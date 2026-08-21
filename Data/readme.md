# Data directory

Real cohort data (UK Biobank, CHARLS, ADNI, …) is **not** included in this repository
due to access restrictions. The demo notebooks read AnnData (`.h5ad`) files from the
following layout:

```
Data/
├── NMR_dataset_fullcohort_107nonderived/          # UKB NMR panel (notebooks 1 & 2)
│   └── val.h5ad                                   #   layer "Z-score normalized", obs age column
├── Blood_dataset_fullcohort_107nonderived_mlm/    # UKB blood biochemistry panel (notebook 4)
│   └── val.h5ad
├── ukb_norm_factors/                              # published UKB-derived normalization factors
│   ├── nmr107_factors.csv                         #   107 NMR features (vocab order): mean/std/unit
│   ├── blood14_factors.csv                        #   14 blood features (Lightweight panel)
│   └── readme.md                                  #   provenance & conventions (aggregate stats only)
├── apply_ukb_norm.py                              # z-score a private cohort with the factors
└── fake/                                          # synthetic demo data (git-ignored)
```

## Demo data without restricted access

Use `generate_fake_data.py` to build synthetic datasets that keep the real schema
but contain only randomized values — no real measurements are ever copied:

```bash
# Mode A: mirror the schema of an existing .h5ad (randomized values, fake_ prefix)
python Data/generate_fake_data.py --from <real.h5ad> --n_samples 1000

# Mode B: synthetic from scratch — creates
#   Data/fake/NMR_dataset_fake/{train,val,test}.h5ad      (pretrain + notebooks 1/2)
#   Data/fake/deep_gompertz_fake/{train,val,test}.h5ad    (DeepGompertz finetune demo)
#   Data/fake/Blood_dataset_fake/{train,val,test}.h5ad    (distillation demo)
python Data/generate_fake_data.py --synthetic --outdir Data/fake
```

Mode B names the 107 NMR features from the released tokenizer vocabulary
(`Model_Weights/MetAgeFormer/vocab.txt`), so the fake data is directly
tokenizable by the released model (falls back to positional `NMR_%03d` names
if `vocab.txt` is missing).

The fake datasets are also what the training demos in
[`Src/docs/run_examples.md`](../Src/docs/run_examples.md) run on — every
training stage has a demo command that finishes in ~1 minute on CPU.

Then point the notebook `DATA_PATH` at `Data/fake/NMR_dataset_fake/val.h5ad`
(a commented line is provided in each notebook). Dependencies: `numpy`, `pandas`,
`anndata` only.

## Custom cohort → model input

For your own cohort (NMR metabolomics or blood-test panel), apply the published
UKB-derived normalization factors with `apply_ukb_norm.py` (see
`Docs/SKILL_CUSTOM_COHORT.md` for units, matching rules, and verification):

```bash
# NMR 107 panel → z-scored matrix (features as rows in the input CSV)
python Data/apply_ukb_norm.py --input /your/cohort.csv --mode nmr107 --out my_cohort_z.csv

# or rebuild an .h5ad into canonical model-input format (never modifies the input)
python Data/apply_ukb_norm.py --mode blood14 \
    --h5ad /your/cohort.h5ad --h5ad-out my_cohort_z.h5ad
```

Raw-data pipelines for the restricted cohorts (UKB / CHARLS / ADNI) are documented in
`Docs/SKILL_DATA_PREPROCESSING.md` (requires your own data access).
