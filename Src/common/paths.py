"""Central default artifact paths for MetAgeFormer."""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "artifacts")
CHECKPOINTS_ROOT = os.path.join(ARTIFACTS_ROOT, "checkpoints")
EVAL_ROOT = os.path.join(ARTIFACTS_ROOT, "eval")
DATA_ROOT = os.path.join(REPO_ROOT, "Data")

PRETRAINED_NMR_ROOT = os.path.join(CHECKPOINTS_ROOT, "pretrained", "nmr")
FINETUNED_DEEP_GOMPERTZ_ROOT = os.path.join(CHECKPOINTS_ROOT, "finetuned", "deep_gompertz")
DISTILLED_LIGHTWEIGHT_ROOT = os.path.join(CHECKPOINTS_ROOT, "distilled", "lightweight")

EVAL_PRETRAINED_NMR_ROOT = os.path.join(EVAL_ROOT, "pretrained", "nmr")
EVAL_DEEP_GOMPERTZ_ROOT = os.path.join(EVAL_ROOT, "deep_gompertz")
EVAL_ADNI_ROOT = os.path.join(EVAL_ROOT, "adni")
EVAL_DISTILLED_LIGHTWEIGHT_ROOT = os.path.join(EVAL_ROOT, "distilled", "lightweight")

DEFAULT_COHORT = "fullcohort_107nonderived_mlm"
DEFAULT_PRETRAINED_DIR = os.path.join(PRETRAINED_NMR_ROOT, DEFAULT_COHORT)
DEFAULT_DEEP_GOMPERTZ_DIR = os.path.join(FINETUNED_DEEP_GOMPERTZ_ROOT, DEFAULT_COHORT)
DEFAULT_DISTILLED_DIR = os.path.join(
    DISTILLED_LIGHTWEIGHT_ROOT, DEFAULT_COHORT
)
DEFAULT_DISTILLED_ABLATION_DIR = os.path.join(
    DISTILLED_LIGHTWEIGHT_ROOT, "ablation_107nonderived_mlm"
)
DEFAULT_DISTILLED_ABLATION_LEGACY_DIR = os.path.join(
    DISTILLED_LIGHTWEIGHT_ROOT, "old_ablation_107nonderived_mlm"
)

DEFAULT_NMR_DATA = os.path.join(DATA_ROOT, "NMR_dataset_fullcohort_107nonderived")
DEFAULT_EMBEDDING_DATA = os.path.join(DATA_ROOT, "Blood_dataset_fullcohort_107nonderived_mlm")
DEFAULT_ADNI_DATA = os.path.join(DATA_ROOT, "ADNI_datasets", "adni_nmr_processed.h5ad")
DEFAULT_ADNI_Q300_DATA = os.path.join(DATA_ROOT, "ADNI_datasets", "adni_q300_processed.h5ad")
EVAL_ADNI_Q300_OVERLAP_ROOT = os.path.join(EVAL_ADNI_ROOT, "q300_overlap")

REPRO_BASELINE_PATH = os.path.join(ARTIFACTS_ROOT, "repro_baseline.json")
