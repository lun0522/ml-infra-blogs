# Post Classifier Standard & Workflow

Newly discovered posts are tagged at update time by a quantized multi-label ONNX classifier (`BAAI/bge-small-en-v1.5` fine-tuned on [`backend/data/labels.jsonl`](../backend/data/labels.jsonl)) stored in [`backend/models/post_classifier/`](../backend/models/post_classifier/).

In GitHub Actions and `python -m app.scripts.update`, inference uses only `onnxruntime`, `tokenizers`, and `numpy` (no PyTorch required at runtime) and scores **newly added URLs only** using `(title, excerpt or summary)`. Each post receives a `topics` list (`0–3` of the 5 pillars below):
- **Kept (`>= 1` pillar):** Displayed in the main feed and selectable by topic pillar in the frontend UI (`label: "infra"`).
- **Filtered (`0` pillars / `topics: []`):** Soft-filtered (`label: "drop"`) and hidden by default behind the UI `Filtered (N)` toggle.

## Keep vs. Drop Standard & The 5 Pillars

Every example in [`backend/data/labels.jsonl`](../backend/data/labels.jsonl) carries `topics: [...]` from the five pillars below (`topics: []` for filtered posts).

### Kept Pillars (`>= 1` Topic)

1. **`inference` (`Inference & Serving`)**
   - Serving engines (`vLLM`, `SGLang`, `TensorRT-LLM`), PagedAttention, RadixCache, disaggregated prefill/decode (PD), KV-cache compression/offloading/routing, speculative decoding (`EAGLE`, `DSpark`, `DFlash`), continuous batching, autoscaling internals, quantization (`FP8`, `FP4`, `INT4`), serving kernels (`FlashInfer`, `FlashAttention`, `MegaMoE`), request isolation/postmortems.
2. **`training` (`Training & RL`)**
   - Distributed RL frameworks (`prime-rl`, `Miles`, `veRL`, `SkyRL`), post-training (`GRPO`, `PPO`, `IPO`, `RLVR`), delta weight sync, pretraining/FSDP/Megatron/TorchTitan, compiler optimizations (`torch.compile`, `Inductor`), scaling laws, synthetic RL data/recipes, bitwise train–rollout consistency.
3. **`hardware` (`Hardware & Silicon`)**
   - GPU/TPU/Trainium/wafer-scale chips (`Blackwell`, `Rubin`, `Ironwood`, `Cerebras WSE`, `SRAM`), low-level GPU languages & compilers (`CUDA`, `Triton`, `CUTLASS`, `Mojo`, `MLIR`, `LLVM`, `Metal`), NVLink/RDMA/InfiniBand/CPO photonics interconnects, HBM/DRAM/SSD memory hierarchies, wafer yield, datacenter buildout/power engineering.
4. **`systems` (`Clusters, Sandboxes & Storage`)**
   - Kubernetes (`KubeRay`), Slurm, Ray, SkyPilot orchestration, sandboxes & microVMs (`Firecracker`, `gVisor`, confidential VMs), cold-start/image optimizations, hardware health telemetry & auto-repair, collectives (`NCCL`, `PCCL`, `NIXL`), distributed filesystems, object-storage queues, and vector/text search engines (`turbopuffer`, `Habitat`).
5. **`agents` (`Agents, Harnesses & Evals`)**
   - Agent runtimes & coding harnesses (`Codex`, `Claude Code`, `MCP`, `Symphony`), agent memory/context engineering, benchmarks (`MLPerf`, `FrontierSWE`, `KernelBench`), safety/eval infrastructure, reward hacking & eval awareness studies.

### Filtered (`topics: []`, `0` Pillars)

- **Non-AI / Out-of-Domain Engineering:** Pure web/CDN/DNS/WAF engineering, generic cybersecurity CVEs (`ruby2shell`, `phpAfterShell`, `nginx2root`, `Java4Shell`), DDoS/election traffic reports, non-AI distributed consensus/caching, legal/compliance certifications, quantum computing (`QPUs`), and domain-only biology/protein pipelines without general AI infra focus.
- **ML-101 Tutorials & Career/Opinion Essays:** Introductory ML/RL/CV/XGBoost tutorials, personal career advice, and high-level opinion essays, market taxonomies, or AI policy commentary lacking technical systems depth.
- **PR, Corporate News & Events:** Funding rounds, M&A, office openings, awards, partnership announcements, newsletter roundups (`Modverse`), conference recaps without technical talk substance, and community commissions.
- **Hosted Cloud Features & Model/Pricing Promos:** Managed SaaS feature releases, pricing/SLA tiers, generic customer spotlights, model-vs-model leaderboard comparisons, and routine **"Model X is now available / Day-0 support"** announcements unless accompanied by a genuinely novel systems deep dive.

### Tiebreakers & Asymmetric Cost Policy

1. **Bias Toward Keep on Genuine Ambiguity:** Missing real infra content is worse than letting an occasional borderline post through. During threshold calibration, `keep_threshold` is tuned on out-of-fold probabilities so `keep_false_drop_rate <= 0.05` (a post is assigned `0` pillars only when `max_p < keep_threshold`; otherwise it receives every pillar above its calibrated threshold, or `[argmax(p_j / threshold_j)]` if none clear individually).
2. **Promo Wrapper vs. Engineering Substance:** If a vendor post mentions a hosted platform but leads with concrete cluster architecture, kernel optimizations, ablations, or eval methodology (verified via `excerpt`), tag its matching pillar(s); otherwise leave `topics: []`.
3. **Use Article Text, Never Source Name:** Model inputs use only `(title, excerpt or summary)` (`source_name` is excluded).

## Model Architecture

- **Encoder & Head:** `ClsMeanPillarClassifier` atop `BAAI/bge-small-en-v1.5`, concatenating final-layer `[CLS]` and attention-masked mean-pooled embeddings (`768`-dim) with 5-sample dropout during training and a linear 5-logit head (`inference`, `training`, `hardware`, `systems`, `agents`), up to `max_length = 512`.
- **Asymmetric Multi-Label Loss:** Mildly pos-weighted `BCEWithLogitsLoss` across the 5 pillars plus an auxiliary Keep-vs-Drop loss (`aux_binary_weight=0.5`, `aux_keep_weight=2.0`) that boosts the strongest *true active* pillar on Keep rows without inflating inactive pillars, and suppresses `max(logits)` across all 5 pillars on Filtered (`0`-pillar) rows.
- **Quantized Export:** Exported via `torch.onnx.export(..., dynamo=True)` with dynamic `(batch_size, seq_len)`, quantized to `QInt8` (`per_channel=True`) with `FP16` embedding compression (`~44 MB`) in [`backend/models/post_classifier/model.onnx`](../backend/models/post_classifier/model.onnx).

## Retraining & Contributing Labels

### 1. Install training dependencies locally

Runtime dependencies (`onnxruntime`, `tokenizers`, `numpy`) live in `backend/requirements.txt`. Heavy training packages (`torch`, `transformers`, `scikit-learn`, `onnx`, `onnxscript`) are isolated in `backend/requirements-train.txt` (or `pip install -e ".[train]"`):

```bash
cd backend
python3.14 -m venv .venv  # Python 3.11–3.14 supported by onnxruntime/torch wheels
.venv/bin/pip install -r requirements.txt -r requirements-train.txt
```

### 2. Export newly archived posts for labeling

```bash
cd backend
.venv/bin/python -m app.scripts.export_unlabeled
```

This writes any posts from `data/posts.json` not yet present in `backend/data/labels.jsonl` (including `title`, `summary`, and `excerpt`) to `backend/data/unlabeled.jsonl`. Assign `"topics"` (`1–3` of `["inference", "training", "hardware", "systems", "agents"]`, or `[]` for filtered posts) and append reviewed lines to `backend/data/labels.jsonl`.

### 3. Train, evaluate (5-fold CV), and export quantized ONNX

```bash
cd backend
.venv/bin/python -m app.scripts.train_classifier
```

This script:
1. Runs 5-fold stratified cross-validation on `backend/data/labels.jsonl` using `(title, excerpt or summary)` (`max_length = 512`).
2. Sweeps `keep_threshold` on out-of-fold `max_p` to hold `keep_false_drop_rate <= 0.05` (with `--max-keep-false-drop`), and tunes per-pillar thresholds for best per-pillar F1.
3. Trains the final model on all labeled rows, exports quantized `backend/models/post_classifier/model.onnx`, checks PyTorch-vs-ORT parity, and writes `config.json` and `metrics.json`.

### 4. Re-tag the existing archive & run tests

```bash
cd backend
.venv/bin/python -m app.scripts.update --from-archive --reclassify
.venv/bin/pytest tests -v
```
