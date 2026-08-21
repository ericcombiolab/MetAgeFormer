# Model Weights

Released weights are committed to the repository via **Git LFS** (`*.pth` is
LFS-tracked, see `.gitattributes`) and downloaded automatically on
`git clone` / `git lfs pull`. The demo notebooks in `Notebooks/` expect exactly
the paths below.

```
Model_Weights/
├── MetAgeFormer/          # pretrained NMR backbone
│   ├── config.json        # backbone architecture (n_heads, n_blocks, d_ff, d_model, ...)
│   ├── tokenizer.pkl      # metabolite vocabulary (pickled MetAgeFormer_Tokenizer)
│   ├── vocab.txt          # the 107 NMR measure names (plain-text mirror of tokenizer.pkl)
│   └── model_weights.pth  # keys: METAGEFORMER / CONCENTRATION_PREDICTOR / MULTITASK_HEADS
├── DeepGompertz/          # finetuned DeepGompertz survival head
│   └── model_weights.pth  # keys: state_dict / config / baseline_params
├── Lightweight/           # distilled blood-token Transformer + DeepGompertz
│   ├── model_weights.pth  # key: METAGEFORMER_DISTILLED (full combined state_dict)
│   └── model_conf.json    # student architecture {n_features, d_model, n_heads, n_blocks, d_ff, dropout}
└── SubtypeClassifier/
    └── subtype_mlp_classifier_focal.joblib  # metabolic subtype classifier (13 clusters)
```

Notes:

- `tokenizer.pkl` is a pickled `metageformer_torch.tokenizer.MetAgeFormer_Tokenizer`
  instance; it must be loaded through `Src/utils.py::load_tokenizer` with `Src/` on
  the Python path (the notebooks set this up automatically).
- `DeepGompertz/model_weights.pth` is also read by notebook 4 as the **teacher**
  checkpoint to recover the Gompertz head config and baseline parameters
  (`Src/metageformer_torch/checkpoint.py::load_teacher_gompertz_config`).
- The `Lightweight` model input is a z-scored blood panel whose feature order
  matches `model_conf.json["n_features"]`.
- `SubtypeClassifier/subtype_mlp_classifier_focal.joblib` is a **PyTorch**
  `torch.save` archive despite the `.joblib` extension — load it with
  `Src/metageformer_torch/subtype_mlp.py::FocalMLPClassifier.load()` (never
  `joblib.load()`). Input: raw 512-dim embeddings from the released backbone.
