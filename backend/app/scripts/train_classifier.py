"""Train, cross-validate, and export a quantized multi-label 5-pillar ONNX classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from ..classifier import (
    DEFAULT_MODEL_DIR,
    PILLARS,
    OnnxPostClassifier,
    decode_pillar_probabilities,
    format_post_text,
)
from ..extractors import canonical_key

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS_PATH = REPO_ROOT / "backend" / "data" / "labels.jsonl"
DEFAULT_BASE_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class LabeledExample:
    url: str
    source_id: str
    title: str
    summary: str
    topics: list[str]
    excerpt: str = ""

    @property
    def text(self) -> str:
        return format_post_text(self.title, self.excerpt or self.summary)


class ClsMeanPillarClassifier(torch.nn.Module):
    """Multi-label 5-pillar classifier combining [CLS] + mean pooling and multi-sample dropout."""

    def __init__(self, base_model: str, num_pillars: int = len(PILLARS)) -> None:
        super().__init__()
        # Use eager attention so ONNX tracing works cleanly across dynamic sequence lengths.
        self.encoder = AutoModel.from_pretrained(base_model, attn_implementation="eager")
        hidden_size = self.encoder.config.hidden_size
        self.dropouts = torch.nn.ModuleList(
            [torch.nn.Dropout(p) for p in (0.1, 0.15, 0.2, 0.25, 0.3)]
        )
        self.head = torch.nn.Linear(hidden_size * 2, num_pillars)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
        mean_emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = torch.cat([out.last_hidden_state[:, 0], mean_emb], dim=-1)
        if self.training:
            return sum(self.head(drop(emb)) for drop in self.dropouts) / len(self.dropouts)
        return self.head(emb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 5-fold CV multi-label 5-pillar classifier and export quantized ONNX."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--aux-binary-weight",
        type=float,
        default=0.5,
        help="Weight on auxiliary Keep-vs-Drop max-logit BCE loss.",
    )
    parser.add_argument(
        "--aux-keep-weight",
        type=float,
        default=2.0,
        help="Positive class weight inside the auxiliary Keep-vs-Drop max-logit loss.",
    )
    parser.add_argument(
        "--max-keep-false-drop",
        type=float,
        default=0.05,
        help="Target upper bound on OOF Keep posts assigned 0 pillars (false drop).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataset_rows(path: Path) -> list[LabeledExample]:
    if not path.exists():
        raise FileNotFoundError(f"Labels dataset not found: {path}")

    valid_pillars = set(PILLARS)
    rows: list[LabeledExample] = []
    seen_urls: set[str] = set()
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        topics = [str(t).strip() for t in (item.get("topics") or []) if str(t).strip()]
        invalid = [t for t in topics if t not in valid_pillars]
        if invalid:
            raise ValueError(f"{path}:{line_no} has unknown topics {invalid} (expected {PILLARS})")

        url = str(item.get("url", "")).strip()
        key = canonical_key(url) if url else f"row-{line_no}"
        if key in seen_urls:
            continue
        seen_urls.add(key)
        rows.append(
            LabeledExample(
                url=url,
                source_id=str(item.get("source_id", "")),
                title=str(item.get("title", "")),
                summary=str(item.get("summary", "")),
                topics=topics,
                excerpt=str(item.get("excerpt") or ""),
            )
        )
    if not rows:
        raise ValueError(f"No labeled examples found in {path}")
    return rows


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_pillar_classifier(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    y_topics: np.ndarray,
    base_model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    aux_binary_weight: float,
    aux_keep_weight: float,
    seed: int,
    device: torch.device,
) -> ClsMeanPillarClassifier:
    torch.manual_seed(seed)
    model = ClsMeanPillarClassifier(base_model, num_pillars=len(PILLARS)).to(device)

    y_tensor = torch.tensor(y_topics, dtype=torch.float32)
    dataset = TensorDataset(input_ids, attention_mask, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    pos_counts = np.maximum(y_topics.sum(axis=0), 1.0)
    neg_counts = np.maximum(len(y_topics) - pos_counts, 1.0)
    # Mild sqrt damping keeps positive recall healthy without inflating drop probabilities.
    damped_weights = np.clip(np.sqrt(neg_counts / pos_counts), 1.0, 2.0)
    pos_weight_t = torch.tensor(damped_weights, dtype=torch.float32, device=device)

    topic_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    keep_loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(aux_keep_weight, dtype=torch.float32, device=device)
    )

    model.train()
    for _ in range(epochs):
        for b_ids, b_mask, b_y in loader:
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_y = b_y.to(device)
            b_keep = (b_y.sum(dim=1) > 0).float()

            optimizer.zero_grad()
            logits = model(input_ids=b_ids, attention_mask=b_mask)
            loss = topic_loss_fn(logits, b_y)
            if aux_binary_weight > 0.0:
                # On Keep rows, push up the strongest true active pillar (never inflate false pillars);
                # on Drop rows, suppress the highest logit across all 5 pillars.
                active_max = logits.masked_fill(b_y < 0.5, -1e4).max(dim=-1).values
                all_max = logits.max(dim=-1).values
                keep_target_logit = torch.where(b_keep > 0.5, active_max, all_max)
                loss = loss + aux_binary_weight * keep_loss_fn(keep_target_logit, b_keep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    return model.eval()


@torch.no_grad()
def _predict_probs(
    model: ClsMeanPillarClassifier,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(input_ids, attention_mask), batch_size=batch_size)
    chunks: list[np.ndarray] = []
    for b_ids, b_mask in loader:
        logits = model(input_ids=b_ids.to(device), attention_mask=b_mask.to(device))
        chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(chunks)


def _stratify_key(y_topics: np.ndarray) -> np.ndarray:
    """Assign each row to its rarest active pillar (or 0 for Filtered/drop) for balanced CV."""
    freq = y_topics.sum(axis=0)
    strata = np.zeros(len(y_topics), dtype=np.int64)
    for i, row in enumerate(y_topics):
        active = np.where(row > 0.5)[0]
        if len(active) == 0:
            strata[i] = 0
        else:
            rarest = active[int(np.argmin(freq[active]))]
            strata[i] = rarest + 1
    return strata


def evaluate_cv(
    rows: list[LabeledExample],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    y_topics: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], float, list[float]]:
    oof_probs = np.zeros((len(rows), len(PILLARS)), dtype=np.float64)
    strata = _stratify_key(y_topics)

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(rows)), strata), start=1):
        print(
            f"[CV {fold_idx}/{args.cv_folds}] training on {len(tr_idx)} rows, validating on {len(va_idx)}...",
            flush=True,
        )
        fold_model = fit_pillar_classifier(
            input_ids=input_ids[tr_idx],
            attention_mask=attention_mask[tr_idx],
            y_topics=y_topics[tr_idx],
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            aux_binary_weight=args.aux_binary_weight,
            aux_keep_weight=args.aux_keep_weight,
            seed=args.seed + fold_idx - 1,
            device=device,
        )
        oof_probs[va_idx] = _predict_probs(
            fold_model, input_ids[va_idx], attention_mask[va_idx], device=device
        )

    y_keep = (y_topics.sum(axis=1) > 0).astype(bool)
    p_keep_oof = np.max(oof_probs, axis=1)
    bin_roc_auc = float(roc_auc_score(y_keep.astype(int), p_keep_oof))

    # Balanced 0.50 operating point for comparison
    pred_keep_50 = p_keep_oof >= 0.50
    uncal_bin_acc = float(np.mean(pred_keep_50 == y_keep))

    keep_th, pillar_ths = _select_thresholds(
        y_topics=y_topics,
        oof_probs=oof_probs,
        max_keep_false_drop=args.max_keep_false_drop,
    )

    decoded = decode_pillar_probabilities(
        oof_probs,
        pillars=PILLARS,
        keep_threshold=keep_th,
        pillar_thresholds=pillar_ths,
    )
    y_pred = np.zeros_like(y_topics, dtype=np.int64)
    for i, topics in enumerate(decoded):
        for t in topics:
            y_pred[i, PILLARS.index(t)] = 1

    pred_keep = y_pred.sum(axis=1) > 0
    bin_acc = float(np.mean(pred_keep == y_keep))
    keep_false_drop = float(np.mean((~pred_keep)[y_keep]))
    drop_recall = float(np.mean((~pred_keep)[~y_keep]))
    drop_denom = int(np.sum(~pred_keep))
    drop_precision = float(np.sum((~pred_keep) & (~y_keep)) / drop_denom) if drop_denom else 1.0

    y_true_int = y_topics.astype(np.int64)
    micro_f1 = float(f1_score(y_true_int, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true_int, y_pred, average="macro", zero_division=0))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true_int, y_pred, average=None, zero_division=0
    )

    per_pillar: dict[str, dict[str, float | int]] = {}
    print("\n=== Out-of-Fold Multi-Label 5-Pillar Report ===")
    for idx, pillar in enumerate(PILLARS):
        auc = float(roc_auc_score(y_true_int[:, idx], oof_probs[:, idx]))
        per_pillar[pillar] = {
            "threshold": round(float(pillar_ths[idx]), 2),
            "roc_auc": round(auc, 4),
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(sup[idx]),
        }
        print(
            f"  {pillar:10s} (th={pillar_ths[idx]:.2f}, n={int(sup[idx]):3d}): "
            f"P={prec[idx]:.3f} R={rec[idx]:.3f} F1={f1[idx]:.3f} AUC={auc:.4f}"
        )

    print(
        f"\nKeep/Drop ROC-AUC: {bin_roc_auc:.4f} | At th=0.50: bin_acc={uncal_bin_acc:.3f}\n"
        f"Deployed Safe Mode (keep_th={keep_th:.2f}): bin_acc={bin_acc:.3f}, "
        f"keep_false_drop={keep_false_drop:.3f} ({int(np.sum((~pred_keep) & y_keep))}/{int(np.sum(y_keep))}), "
        f"drop_recall={drop_recall:.3f}, drop_precision={drop_precision:.3f}\n"
        f"Multi-label 5-pillar Micro-F1={micro_f1:.3f}, Macro-F1={macro_f1:.3f}"
    )

    false_drops = [
        (float(p_keep_oof[i]), rows[i])
        for i in range(len(rows))
        if y_keep[i] and not pred_keep[i]
    ]
    false_drops.sort(key=lambda item: item[0])
    print(f"\nKeep posts falsely assigned 0 pillars ({len(false_drops)}):")
    for pk, row in false_drops[:15]:
        print(f"  [{row.source_id:14s}] max_p={pk:.2f} true={row.topics} | {row.title}")

    return (
        {
            "cv_folds": args.cv_folds,
            "keep_drop_roc_auc": round(bin_roc_auc, 4),
            "at_0_50_operating_point": {
                "binary_keep_drop_accuracy": round(uncal_bin_acc, 4),
            },
            "binary_keep_drop_accuracy": round(bin_acc, 4),
            "keep_false_drop_rate": round(keep_false_drop, 4),
            "drop_recall": round(drop_recall, 4),
            "drop_precision": round(drop_precision, 4),
            "pillar_micro_f1": round(micro_f1, 4),
            "pillar_macro_f1": round(macro_f1, 4),
            "per_pillar": per_pillar,
        },
        keep_th,
        pillar_ths,
    )


def _select_thresholds(
    y_topics: np.ndarray,
    oof_probs: np.ndarray,
    max_keep_false_drop: float,
) -> tuple[float, list[float]]:
    y_keep = y_topics.sum(axis=1) > 0
    p_keep = np.max(oof_probs, axis=1)

    print("\n=== Keep-Threshold Sweep (OOF: 0 pillars iff max_p < keep_th) ===")
    best_valid: tuple[float, float] | None = None
    fallback_best: tuple[tuple[float, float], float] | None = None
    for thresh in (0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.16, 0.14, 0.12, 0.10, 0.08):
        pred_keep = p_keep >= thresh
        keep_false_drop = float(np.mean((~pred_keep)[y_keep]))
        drop_recall = float(np.mean((~pred_keep)[~y_keep]))
        denom = int(np.sum(~pred_keep))
        drop_precision = float(np.sum((~pred_keep) & (~y_keep)) / denom) if denom else 1.0
        bin_acc = float(np.mean(pred_keep == y_keep))
        print(
            f"  keep_th={thresh:.2f} -> bin_acc={bin_acc:.3f}, keep_false_drop={keep_false_drop:.3f}, "
            f"drop_recall={drop_recall:.3f}, drop_precision={drop_precision:.3f}"
        )
        if keep_false_drop <= max_keep_false_drop:
            score = bin_acc + drop_recall
            if best_valid is None or score > best_valid[0]:
                best_valid = (score, thresh)
        if drop_recall >= 0.45:
            fb_key = (-keep_false_drop, drop_recall)
            if fallback_best is None or fb_key > fallback_best[0]:
                fallback_best = (fb_key, thresh)

    keep_th = best_valid[1] if best_valid is not None else (fallback_best[1] if fallback_best else 0.14)

    # Tune per-pillar threshold on OOF samples clearing keep_th to maximize each pillar's F1.
    pillar_ths: list[float] = []
    kept_mask = p_keep >= keep_th
    for j in range(len(PILLARS)):
        best_f1 = -1.0
        best_t = 0.55
        y_col = y_topics[:, j] > 0.5
        for cand in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            pred_col = kept_mask & (oof_probs[:, j] >= cand)
            score = float(f1_score(y_col, pred_col, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_t = cand
        pillar_ths.append(best_t)

    print(
        f"Selected keep_threshold={keep_th:.2f} (target keep_false_drop <= {max_keep_false_drop:.2f}), "
        f"pillar_thresholds={dict(zip(PILLARS, pillar_ths))}"
    )
    return keep_th, pillar_ths


def export_quantized_onnx(
    model: ClsMeanPillarClassifier,
    tokenizer: Any,
    rows: list[LabeledExample],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    out_dir: Path,
    max_length: int,
    keep_threshold: float,
    pillar_thresholds: list[float],
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_model = model.cpu().eval()

    tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "pillars": list(PILLARS),
                "max_length": max_length,
                "keep_threshold": keep_threshold,
                "pillar_thresholds": {p: round(float(t), 2) for p, t in zip(PILLARS, pillar_thresholds)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fp32_path = out_dir / "model.fp32.onnx"
    int8_path = out_dir / "model.onnx"
    batch_dim = torch.export.Dim("batch_size", min=1, max=256)
    seq_dim = torch.export.Dim("seq_len", min=1, max=max_length)
    sample_seq = min(64, input_ids.shape[1])
    with torch.no_grad():
        torch.onnx.export(
            cpu_model,
            (input_ids[:2, :sample_seq].cpu(), attention_mask[:2, :sample_seq].cpu()),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_shapes={
                "input_ids": {0: batch_dim, 1: seq_dim},
                "attention_mask": {0: batch_dim, 1: seq_dim},
            },
            dynamo=True,
        )

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    _compress_large_embeddings_fp16(int8_path)
    fp32_path.unlink(missing_ok=True)
    fp32_data = fp32_path.with_suffix(".onnx.data")
    fp32_data.unlink(missing_ok=True)

    sample_count = min(32, len(rows))
    ref_probs = _predict_probs(
        cpu_model,
        input_ids[:sample_count],
        attention_mask[:sample_count],
        device=torch.device("cpu"),
    )
    ref_topics = decode_pillar_probabilities(
        ref_probs,
        pillars=PILLARS,
        keep_threshold=keep_threshold,
        pillar_thresholds=pillar_thresholds,
    )

    ort_classifier = OnnxPostClassifier(out_dir)
    ort_preds = ort_classifier.predict_batch(
        [(r.title, r.excerpt or r.summary) for r in rows[:sample_count]]
    )
    ort_probs = np.array([[p.probabilities[lbl] for lbl in PILLARS] for p in ort_preds])

    max_prob_diff = float(np.max(np.abs(ref_probs - ort_probs)))
    exact_topic_agreement = float(np.mean([r == p.topics for r, p in zip(ref_topics, ort_preds)]))
    keep_agreement = float(np.mean([bool(r) == bool(p.topics) for r, p in zip(ref_topics, ort_preds)]))
    model_size_mb = int8_path.stat().st_size / (1024 * 1024)
    print(
        f"\nExported quantized ONNX model to {int8_path} ({model_size_mb:.2f} MB) | "
        f"keep parity={keep_agreement:.3f}, topic parity={exact_topic_agreement:.3f}, max_prob_diff={max_prob_diff:.4f}"
    )
    return {
        "model_size_mb": round(model_size_mb, 2),
        "quantization_keep_parity": keep_agreement,
        "quantization_topic_parity": exact_topic_agreement,
        "quantization_max_prob_diff": round(max_prob_diff, 5),
    }


def _compress_large_embeddings_fp16(model_path: Path) -> None:
    """Store large Gather embedding tables in float16 and cast back to float32 after lookup."""
    proto = onnx.load(str(model_path))
    init_by_name = {init.name: init for init in proto.graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in proto.graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)

    converted = 0
    for init in list(proto.graph.initializer):
        if init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        if arr.size < 100_000:
            continue
        nodes = consumers.get(init.name, [])
        if not nodes or any(n.op_type != "Gather" for n in nodes):
            continue

        fp16_init = numpy_helper.from_array(arr.astype(np.float16), name=init.name)
        init.CopyFrom(fp16_init)

        for idx, gather_node in enumerate(nodes):
            orig_out = gather_node.output[0]
            fp16_out = f"{orig_out}__fp16_{idx}"
            gather_node.output[0] = fp16_out
            cast_node = helper.make_node(
                "Cast",
                inputs=[fp16_out],
                outputs=[orig_out],
                to=TensorProto.FLOAT,
                name=f"{gather_node.name or 'gather'}_to_fp32_{idx}",
            )
            node_pos = list(proto.graph.node).index(gather_node)
            proto.graph.node.insert(node_pos + 1, cast_node)
        converted += 1

    if converted:
        onnx.save(proto, str(model_path))


def main() -> int:
    args = parse_args()
    rows = load_dataset_rows(args.labels)
    y_topics = np.zeros((len(rows), len(PILLARS)), dtype=np.float32)
    for idx, row in enumerate(rows):
        for topic in row.topics:
            y_topics[idx, PILLARS.index(topic)] = 1.0

    counts = {p: int(y_topics[:, idx].sum()) for idx, p in enumerate(PILLARS)}
    filtered_count = int((y_topics.sum(axis=1) == 0).sum())
    print(f"Loaded {len(rows)} labeled examples from {args.labels}: pillars={counts}, filtered={filtered_count}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    encoded = tokenizer(
        [r.text for r in rows],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    device = _select_device()
    cv_metrics, keep_th, pillar_ths = evaluate_cv(
        rows=rows,
        input_ids=input_ids,
        attention_mask=attention_mask,
        y_topics=y_topics,
        args=args,
        device=device,
    )

    print(f"\nTraining final model on all {len(rows)} rows (device={device})...", flush=True)
    final_model = fit_pillar_classifier(
        input_ids=input_ids,
        attention_mask=attention_mask,
        y_topics=y_topics,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        aux_binary_weight=args.aux_binary_weight,
        aux_keep_weight=args.aux_keep_weight,
        seed=args.seed,
        device=device,
    )
    export_metrics = export_quantized_onnx(
        model=final_model,
        tokenizer=tokenizer,
        rows=rows,
        input_ids=input_ids,
        attention_mask=attention_mask,
        out_dir=args.out_dir,
        max_length=args.max_length,
        keep_threshold=keep_th,
        pillar_thresholds=pillar_ths,
    )

    metrics_payload = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model": args.base_model,
        "architecture": "cls_mean_5pillar_multilabel",
        "aux_binary_weight": args.aux_binary_weight,
        "aux_keep_weight": args.aux_keep_weight,
        "dataset_size": len(rows),
        "pillar_counts": counts,
        "filtered_count": filtered_count,
        "keep_threshold": keep_th,
        "pillar_thresholds": {p: round(float(t), 2) for p, t in zip(PILLARS, pillar_ths)},
        "cv": cv_metrics,
        "export": export_metrics,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved metrics to {args.out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
