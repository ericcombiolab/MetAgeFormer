# Skill: 假数据生成（供 AI Agents 使用）/ Skill: Fake Data Generation (for AI Agents)

> 中文：当用户需要演示数据、冒烟测试数据、或想在不使用受限队列数据的情况下跑通管线时，
> 按本 Skill 生成**假数据**。假数据只保留真实数据的**结构**（schema），数值全部随机，
> 绝不复制任何真实测量值。
>
> English: Use this skill whenever the user needs demo data, smoke-test data, or wants to
> run the pipelines without restricted cohort data. Fake data keeps only the **schema** of
> real data; all values are random and no real measurements are ever copied.

---

## 1. 何时使用 / When to Use

- 用户要求生成演示/示例数据（demo/example/fake/synthetic data）
- 需要验证 pipeline / notebook 在无数据权限的情况下可运行
- 为新用户准备 Notebook 或训练 demo 所需的数据

## 2. 生成方式 / How to Generate

工具脚本：`Data/generate_fake_data.py`（依赖仅 numpy/pandas/anndata，不要引入
scanpy/sklearn）。两种模式：

### 模式 A / Mode A：镜像真实 h5ad 的结构（数值随机）

```bash
python Data/generate_fake_data.py --from <真实数据.h5ad> --n_samples 1000
# 输出: <同目录>/fake_<原名>.h5ad（--out_dir 可改输出目录）
```

规则：
- 每个特征单独随机：按原数据该特征 mean±std 采样，clip 到原 min/max
- `var`/`obs` 的 schema 保留；obs 数值列按原分布随机，类别列从原取值中抽样
- `layers['Z-score normalized']`：若 var 含 `Z-score mean`/`Z-score std` 列，用它们重新
  计算 z-score；否则按原 layer 逐特征随机
- `obsm` 键保留（如 512 维 `metabolomic embedding`），生成均值 0 方差 1 的随机 embedding
- 文件名加 `fake_` 前缀

### 模式 B / Mode B：全合成（无需任何数据访问）

```bash
python Data/generate_fake_data.py --synthetic --outdir Data/fake
```

生成三套数据（train/val/test 划分）：

| 目录 | 内容 | 服务于 |
|---|---|---|
| `Data/fake/NMR_dataset_fake/` | 107 指标 + Z-score layer + obs age | pretrain demo、Notebook 1/2 |
| `Data/fake/deep_gompertz_fake/` | 512 维 embedding + Gompertz 生存 obs | DeepGompertz finetune demo |
| `Data/fake/Blood_dataset_fake/` | 14 血液指标 + layer + embedding + 生存 obs | distillation demo |

107 个 NMR 指标名取自 `Model_Weights/MetAgeFormer/vocab.txt`（保证可被 tokenizer
直接分词；文件缺失时回退为位置编号 `NMR_%03d`）。

## 3. 注意事项 / Notes

- `Data/fake/` 已被 `.gitignore` 忽略，不会入库
- 假数据**严禁**用于任何真实分析/论文结论，脚本末尾也会打印此警告
- 真实数据布局见 `Data/readme.md`
