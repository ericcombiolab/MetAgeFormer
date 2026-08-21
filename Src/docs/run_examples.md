# Run examples

Run from `Src/` with `PYTHONPATH=.` (or `cd Src && export PYTHONPATH=.`).

All commands below are plain local `python` invocations. They work on synthetic
demo data (no restricted cohort data required) out of the box; the real-data
variants only need the corresponding `.h5ad` files under `Data/` (see
`Data/readme.md`).

## 0. Generate demo data

```bash
# From the repository root:
python Data/generate_fake_data.py --synthetic --outdir Data/fake
```

Creates:

- `Data/fake/NMR_dataset_fake/` — 107-measure NMR panels (pretrain + notebook 1/2)
- `Data/fake/deep_gompertz_fake/` — 512-dim embeddings + survival obs (DeepGompertz finetune)
- `Data/fake/Blood_dataset_fake/` — 14-feature blood panels (distillation)

All values are random; only the schema is real. Never use for actual analysis.

## 1. Pretrain NMR (masked concentration imputation)

Demo (small model, 2 epochs, ~1 min on CPU):

```bash
python pretrain/train.py --train_config ./pretrain/config/mlmtask_demo.json
```

Imputation-quality eval on the demo data:

```bash
python pretrain/eval.py \
  --model_dir ../artifacts/demo/checkpoints/pretrained/nmr/demo \
  --data_path ../Data/fake/NMR_dataset_fake \
  --save_dir ../artifacts/demo/eval/pretrained/nmr/demo
```

Full-scale run (real data required):

```bash
python pretrain/train.py --train_config ./pretrain/config/mlmtask.json          # 107-measure
python pretrain/train.py --train_config ./pretrain/config/mlmtask_allmeasures.json  # all-measure
```

## 2. DeepGompertz finetune (metabolomic aging clock)

Demo (small head, few epochs, ~1 min on CPU):

```bash
python finetune/deep_gompertz/train.py \
  --data_path ../Data/fake/deep_gompertz_fake \
  --save_dir ../artifacts/demo/checkpoints/finetuned/deep_gompertz/demo \
  --batch_size 32 --n_epoch 3 --baseline_epoch 5 --baseline_n_toler 2 --n_toler 2 \
  --use_wandb false
```

Full-scale run (real data required):

```bash
python finetune/deep_gompertz/train.py \
  --data_path ../Data/Blood_dataset_fullcohort_107nonderived_mlm \
  --save_dir ../artifacts/checkpoints/finetuned/deep_gompertz/fullcohort_107nonderived_mlm \
  --batch_size 512 --baseline_epoch 1000 --n_epoch 1000 --n_toler 5 \
  --use_wandb false
```

Evaluation (real UKB NMR / ADNI data required):

```bash
python finetune/deep_gompertz/eval_ukb.py \
  --pretrained_dir ../artifacts/checkpoints/pretrained/nmr/fullcohort_107nonderived_mlm \
  --model_dir ../artifacts/checkpoints/finetuned/deep_gompertz/fullcohort_107nonderived_mlm \
  --data_path ../Data/NMR_dataset_fullcohort_107nonderived \
  --save_dir ../artifacts/eval/deep_gompertz/fullcohort_107nonderived_mlm \
  --batch_size 512

python finetune/deep_gompertz/eval_adni.py \
  --pretrained_dir ../artifacts/checkpoints/pretrained/nmr/fullcohort_107nonderived_mlm \
  --model_dir ../artifacts/checkpoints/finetuned/deep_gompertz/fullcohort_107nonderived_mlm \
  --data_path ../Data/ADNI_datasets/adni_nmr_processed.h5ad \
  --save_dir ../artifacts/eval/adni/deep_gompertz

# Missing-value robustness simulations
python finetune/deep_gompertz/eval_ukb_missing_simulation.py \
  --pretrained_dir ../artifacts/checkpoints/pretrained/nmr/fullcohort_107nonderived_mlm \
  --model_dir ../artifacts/checkpoints/finetuned/deep_gompertz/fullcohort_107nonderived_mlm \
  --data_path ../Data/NMR_dataset_fullcohort_107nonderived \
  --save_dir ../artifacts/eval/deep_gompertz/missing_simulation
```

## 3. Distillation (lightweight blood-token Transformer + DeepGompertz)

Demo (tiny student, 3 epochs, ~1 min on CPU; uses the released DeepGompertz
checkpoint as the teacher):

```bash
python distillation/train.py --train_config ./distillation/config/blood_Distill_DeepGompertz_demo.json
```

Full-scale run (real blood data required):

```bash
python distillation/train.py --train_config ./distillation/config/blood_Distill_DeepGompertz.json
python distillation/train_ablation.py --train_config ./distillation/config/blood_Ablation_DeepGompertz.json
```

Evaluation (real UKB blood / CHARLS data required):

```bash
python distillation/eval_ukb.py \
  --model_dir ../artifacts/checkpoints/distilled/lightweight/fullcohort_107nonderived_mlm \
  --gompertz_head_path ../artifacts/checkpoints/finetuned/deep_gompertz/fullcohort_107nonderived_mlm \
  --data_path ../Data/Blood_dataset_fullcohort_107nonderived_mlm \
  --save_dir ../artifacts/eval/distilled/lightweight/ukb

python distillation/eval_charls.py \
  --model_dir ../artifacts/checkpoints/distilled/lightweight/fullcohort_107nonderived_mlm \
  --gompertz_head_path ../artifacts/checkpoints/finetuned/deep_gompertz/fullcohort_107nonderived_mlm \
  --charls_dir ../Data/CHARLS_datasets
```

## 4. Ablations (from scratch / fully finetuned DeepGompertz)

Real data required:

```bash
python ablation/train_from_scratch.py \
  --data_path ../Data/NMR_dataset_fullcohort_107nonderived \
  --save_dir ../artifacts/checkpoints/ablation/deep_gompertz_from_scratch \
  --use_wandb false

python ablation/train_fully_finetune.py \
  --pretrained_dir ../artifacts/checkpoints/pretrained/nmr/fullcohort_107nonderived_mlm \
  --data_path ../Data/NMR_dataset_fullcohort_107nonderived \
  --save_dir ../artifacts/checkpoints/ablation/deep_gompertz_fully_finetuned \
  --use_wandb false

python ablation/eval.py \
  --model_dir ../artifacts/checkpoints/ablation/deep_gompertz_from_scratch \
  --data_path ../Data/NMR_dataset_fullcohort_107nonderived \
  --save_dir ../artifacts/eval/ablation
```

## Notes

- All training outputs go to `artifacts/` (git-ignored); demo runs use
  `artifacts/demo/...`.
- `wandb` logging is off by default (`--use_wandb false` / `"wandb_monitor": false`).
  Install `wandb` and set `--use_wandb true` (or `wandb_monitor: true`) to enable it.
- On clusters, submit the same commands via your own scheduler (e.g., `sbatch`),
  which is environment-specific and therefore not included in this repository.
