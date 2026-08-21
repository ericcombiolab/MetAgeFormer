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
