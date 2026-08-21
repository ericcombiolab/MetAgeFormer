# AGENTS.md — AI Agent 入口指引 / Entry Point for AI Agents

**中文**：本仓库面向 AI 代理（Claude Code、Codex 等）的操作指南：

- [`Docs/AGENT_SETUP_CN.md`](Docs/AGENT_SETUP_CN.md) — 环境自动安装与自检（中文）：前置检查 → git-lfs 克隆 → conda 建环境 → 假数据冒烟测试 → 仓库约定 → 故障排查
- [`Docs/AGENT_SETUP_EN.md`](Docs/AGENT_SETUP_EN.md) — 同内容英文版

各功能 Skill（中英双语，每个模型能力一份）：

| Skill | 内容 |
|---|---|
| [`Docs/SKILL_FAKE_DATA.md`](Docs/SKILL_FAKE_DATA.md) | 假数据生成（demo / 冒烟测试） |
| [`Docs/SKILL_EMBEDDINGS.md`](Docs/SKILL_EMBEDDINGS.md) | 提取 512 维 metabolomic embeddings（Notebook 1） |
| [`Docs/SKILL_AGING_CLOCK.md`](Docs/SKILL_AGING_CLOCK.md) | 代谢年龄 + age gap，DeepGompertz（Notebook 2） |
| [`Docs/SKILL_SUBTYPES.md`](Docs/SKILL_SUBTYPES.md) | 代谢亚型/元亚型分型（Notebook 3） |
| [`Docs/SKILL_LIGHTWEIGHT.md`](Docs/SKILL_LIGHTWEIGHT.md) | 轻量血液面板模型（Notebook 4） |
| [`Docs/SKILL_TRAINING.md`](Docs/SKILL_TRAINING.md) | 训练管线（pretrain / finetune / distillation / ablation，含 demo 命令） |

核心规则速览 / Quick rules：

| 规则 / Rule | 说明 / Detail |
|---|---|
| 环境 / Environment | `conda env create -f environment_cpu.yml`（或 `environment_gpu.yml`），环境名 `metageformer`；只装最小依赖，勿加冗余包 |
| 运行 / Running | 脚本在 `Src/` 下运行：`cd Src && export PYTHONPATH=.` |
| 数据 / Data | 真实数据受限，不随仓库分发（见 `Data/readme.md`） |
| 假数据 / Fake data | 用 `Data/generate_fake_data.py` 生成（规则见 `Docs/SKILL_FAKE_DATA.md`），输出带 `fake_` 前缀且不入 git |
| 训练输出 / Outputs | 一律写入 `artifacts/`（git-ignored） |
| 权重 / Weights | `Model_Weights/*.pth` 走 Git LFS，勿直接提交改动后的 pth |
| wandb | 默认关闭，用户明确要求才开 |
| 术语 / Terminology | 血浆 NMR（plasma NMR）；预训练任务叫 masked concentration imputation；代谢年龄（metabolomic age）是主输出，mortality risk 只是辅助 |
| 修改范围 / Scope | 只改本仓库内文件 |

**English**: Operating guides for AI agents in this repository:

- [`Docs/AGENT_SETUP_EN.md`](Docs/AGENT_SETUP_EN.md) — environment auto-setup & verification (English)
- [`Docs/AGENT_SETUP_CN.md`](Docs/AGENT_SETUP_CN.md) — Chinese version
- Per-capability skills in `Docs/SKILL_*.md` (bilingual), one per model capability
  (fake data, embeddings, aging clock, subtypes, lightweight, training).

Please read the guide for your language first, then follow it to install and
verify the environment autonomously.
