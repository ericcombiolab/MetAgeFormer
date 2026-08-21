# AGENTS.md — Entry Point for AI Agents / AI Agent 入口指引

**English**: Operating guides for AI agents (Claude Code, Codex, etc.) in this repository:

- [`Docs/AGENT_SETUP_EN.md`](Docs/AGENT_SETUP_EN.md) — environment auto-setup & verification (English): prerequisites → git-lfs clone → conda env → fake-data smoke test → repository conventions → troubleshooting
- [`Docs/AGENT_SETUP_CN.md`](Docs/AGENT_SETUP_CN.md) — Chinese version

Per-capability skills (bilingual, English-first; one per model capability):

| Skill | Covers |
|---|---|
| [`Docs/SKILL_FAKE_DATA.md`](Docs/SKILL_FAKE_DATA.md) | Fake data generation (demo / smoke tests) |
| [`Docs/SKILL_EMBEDDINGS.md`](Docs/SKILL_EMBEDDINGS.md) | Extract 512-dim metabolomic embeddings (Notebook 1) |
| [`Docs/SKILL_AGING_CLOCK.md`](Docs/SKILL_AGING_CLOCK.md) | Metabolomic age + age gap, DeepGompertz (Notebook 2) |
| [`Docs/SKILL_SUBTYPES.md`](Docs/SKILL_SUBTYPES.md) | Metabolic subtype / meta-subtype assignment (Notebook 3) |
| [`Docs/SKILL_LIGHTWEIGHT.md`](Docs/SKILL_LIGHTWEIGHT.md) | Lightweight blood-panel model (Notebook 4) |
| [`Docs/SKILL_TRAINING.md`](Docs/SKILL_TRAINING.md) | Training pipelines (pretrain / finetune / distillation / ablation, incl. demo commands) |

Quick rules:

| Rule | Detail |
|---|---|
| Environment | `conda env create -f environment_cpu.yml` (or `environment_gpu.yml`), env name `metageformer`; install only the minimal dependency set |
| Running | Scripts run from `Src/`: `cd Src && export PYTHONPATH=.` |
| Data | Real cohort data is restricted, not distributed (see `Data/readme.md`) |
| Fake data | Generate with `Data/generate_fake_data.py` (rules in `Docs/SKILL_FAKE_DATA.md`); outputs have a `fake_` prefix and are never committed |
| Outputs | Always under `artifacts/` (git-ignored) |
| Weights | `Model_Weights/*.pth` are Git LFS-tracked; do not plain-commit modified .pth files |
| wandb | Off by default; enable only when the user explicitly asks |
| Terminology | plasma NMR (not blood NMR); pretraining task is "masked concentration imputation"; metabolomic age is the primary output, mortality risk is auxiliary |
| Scope | Only touch files in this repository |

**中文**：本仓库面向 AI 代理（Claude Code、Codex 等）的操作指南：

- [`Docs/AGENT_SETUP_EN.md`](Docs/AGENT_SETUP_EN.md) — 环境自动安装与自检（英文）
- [`Docs/AGENT_SETUP_CN.md`](Docs/AGENT_SETUP_CN.md) — 中文版
- 各功能 Skill 见上表（中英双语、英文为主，每个模型能力一份）

核心规则见上表（环境/运行/数据/假数据/输出/权重/wandb/术语/修改范围）。

Please read the guide for your language first, then follow it to install and
verify the environment autonomously.
