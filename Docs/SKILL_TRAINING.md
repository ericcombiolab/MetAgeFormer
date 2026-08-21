# Skill: Training Pipelines (for AI Agents) / 训练管线（供 AI Agents 使用）

> English: Use this skill when the user wants to run the training pipelines (pretrain /
> DeepGompertz finetune / distillation / ablations). All commands are **plain local
> python** (no cluster tooling), run from `Src/`. Every stage ships a **demo command**
> (fake data, ~1 minute on CPU); real-data commands are listed alongside.
>
> 中文：用户想跑 MetAgeFormer 的训练管线（预训练 / DeepGompertz 微调 / 蒸馏 / 消融）
> 时按本 Skill 操作。所有命令均为**本地 python 命令**（无集群工具），在 `Src/` 下运行。
> 每个阶段都有 **demo 命令**（假数据，CPU ~1 分钟级）；真实数据命令同文档列出。

## 0. Preparation / 准备

```bash
cd Src && export PYTHONPATH=.
# From the repo root, generate fake data (if not done yet):
python Data/generate_fake_data.py --synthetic --outdir Data/fake
```

Rules: all outputs go under `artifacts/` (git-ignored); demo runs use `artifacts/demo/...`;
wandb is off by default. Real-data (UKB/CHARLS/ADNI) layout: `Data/readme.md`.

规则：所有输出写入 `artifacts/`（git-ignored），demo 用 `artifacts/demo/...`；
wandb 默认关闭。真实数据（UKB/CHARLS/ADNI）布局见 `Data/readme.md`。

## 1. Pretrain NMR (masked concentration imputation) / 预训练 NMR

```bash
# demo (2 epochs, small model) / demo（2 epoch，小模型）
python pretrain/train.py --train_config ./pretrain/config/mlmtask_demo.json
# real data (107 measures / all measures) / 真实数据（107 指标 / 全部指标）
python pretrain/train.py --train_config ./pretrain/config/mlmtask.json
python pretrain/train.py --train_config ./pretrain/config/mlmtask_allmeasures.json
# imputation-quality eval / 插补质量评估
python pretrain/eval.py --model_dir <save_dir> --data_path ../Data/NMR_dataset_fullcohort_107nonderived --save_dir ../artifacts/eval/pretrained/nmr/fullcohort_107nonderived_mlm
```

## 2. DeepGompertz Finetune (metabolomic aging clock) / DeepGompertz 微调

```bash
# demo (fake data, 128 training samples) / demo（假数据，128 训练样本）
python finetune/deep_gompertz/train.py \
  --data_path ../Data/fake/deep_gompertz_fake \
  --save_dir ../artifacts/demo/checkpoints/finetuned/deep_gompertz/demo \
  --batch_size 32 --n_epoch 3 --baseline_epoch 5 --baseline_n_toler 2 --n_toler 2 \
  --use_wandb false
# real data: point --data_path at the real embedding dataset; eval in run_examples.md
# 真实数据：--data_path 换成真实 embedding 数据集；评估见 run_examples.md
```

## 3. Distillation (lightweight blood-panel model) / 蒸馏（轻量血液面板模型）

```bash
# demo (teacher = released DeepGompertz weights) / demo（teacher = 发布的 DeepGompertz 权重）
python distillation/train.py --train_config ./distillation/config/blood_Distill_DeepGompertz_demo.json
# real data / 真实数据
python distillation/train.py --train_config ./distillation/config/blood_Distill_DeepGompertz.json
python distillation/train_ablation.py --train_config ./distillation/config/blood_Ablation_DeepGompertz.json
```

Note: the student `d_model` must match the teacher head input dim (512 for the released
model). / 注意：蒸馏的学生 `d_model` 必须与 teacher 头输入维度一致（发布模型为 512）。

## 4. Ablations (real data) / 消融（真实数据）

```bash
python ablation/train_from_scratch.py \
  --data_path ../Data/NMR_dataset_fullcohort_107nonderived \
  --save_dir ../artifacts/checkpoints/ablation/deep_gompertz_from_scratch --use_wandb false
python ablation/train_fully_finetune.py \
  --pretrained_dir ../artifacts/checkpoints/pretrained/nmr/fullcohort_107nonderived_mlm \
  --data_path ../Data/NMR_dataset_fullcohort_107nonderived \
  --save_dir ../artifacts/checkpoints/ablation/deep_gompertz_fully_finetuned --use_wandb false
```

## 5. Evaluation (real data) / 评估（真实数据）

- UKB NMR: `finetune/deep_gompertz/eval_ukb.py`, `eval_ukb_missing_simulation.py`
- ADNI: `finetune/deep_gompertz/eval_adni.py`, `eval_adni_q300_overlap.py`, `eval_adni_missing_simulation.py`
- Blood/CHARLS: `distillation/eval_ukb.py`, `distillation/eval_charls.py`
- Ablations: `ablation/eval.py`

Full command templates: `Src/docs/run_examples.md`. / 完整命令模板：`Src/docs/run_examples.md`。

## 6. Notes / 注意事项

- On clusters, submit the same commands via the user's own scheduler (no cluster tooling
  in this repo) / 集群作业请用用户自己的调度器提交同一命令（仓库不含集群工具）
- wandb optional: `pip install wandb`, then `--use_wandb true` / `"wandb_monitor": true`
  / wandb 可选：`pip install wandb` 后 `--use_wandb true` / `"wandb_monitor": true`
- Data schema before training: pretrain needs `layers['Z-score normalized']`;
  finetune needs `obsm['metabolomic embedding']` + obs age/event/time; distillation
  needs layer + embedding + survival obs (see the fake-data implementation in
  `Data/generate_fake_data.py`) / 训练前检查数据 schema：pretrain 需
  `layers['Z-score normalized']`；finetune 需 `obsm['metabolomic embedding']` + obs 的
  age/event/time；distillation 需 layer + embedding + 生存 obs（见
  `Data/generate_fake_data.py` 的假数据实现）
