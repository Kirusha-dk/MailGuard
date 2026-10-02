#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score

import benchmark_v18_overnight as v18
import benchmark_v25_soft_veto as v25
import benchmark_v27_multidomain as md
import benchmark_v28_crosssource as v28
import benchmark_v30_balanced_lowfp as v30
import benchmark_v31_sanitized_crosssource as v31

base = v25.base
v3 = base.v3

SEED = 20261013
OUT_MODELS = Path("models/v33")
REPORT_JSON = Path("reports/v33-overnight-neural-50k.json")
REPORT_MD = Path("reports/v33-overnight-neural-50k.md")
SOURCE_BY_PATH: dict[str, str] = {}


class OvernightMailNet(nn.Module):
    """
    Bigger two-tower network than the old v18 model.

    Content and Rspamd/numeric evidence are encoded separately, then the
    numeric tower softly gates the content tower before the final classifier.
    """

    def __init__(self):
        super().__init__()
        self.word = nn.EmbeddingBag(
            v18.WORD_VOCAB, 192, mode="mean", include_last_offset=False
        )
        self.char = nn.EmbeddingBag(
            v18.CHAR_VOCAB, 128, mode="mean", include_last_offset=False
        )

        self.content = nn.Sequential(
            nn.Linear(192 + 128, 320),
            nn.GELU(),
            nn.LayerNorm(320),
            nn.Dropout(0.16),
            nn.Linear(320, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )
        self.numeric = nn.Sequential(
            nn.Linear(v18.NUMERIC_DIM, 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Linear(96, 96),
            nn.GELU(),
            nn.LayerNorm(96),
        )
        self.meta_to_content = nn.Linear(96, 256)
        self.gate = nn.Sequential(nn.Linear(96, 256), nn.Sigmoid())

        self.head = nn.Sequential(
            nn.Linear(256 + 96, 320),
            nn.GELU(),
            nn.LayerNorm(320),
            nn.Dropout(0.22),
            nn.Linear(320, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Dropout(0.15),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 1),
        )

    def forward(self, words, word_offsets, chars, char_offsets, numeric):
        w = self.word(words, word_offsets)
        c = self.char(chars, char_offsets)
        content = self.content(torch.cat([w, c], dim=1))
        meta = self.numeric(numeric)
        gate = self.gate(meta)
        fused = content * (0.65 + 0.70 * gate) + self.meta_to_content(meta)
        return self.head(torch.cat([fused, meta], dim=1)).squeeze(1)


def mild_source_weights(rows, deep=False):
    """
    Start from the proven v18 weights and only mildly correct source imbalance.

    V32 showed that full source equalisation was too aggressive. Here the
    correction is square-rooted and clipped, so useful large-corpus signal is
    retained while small corpora cannot be ignored completely.
    """
    w = v18._v33_original_base_weights(rows, deep=deep).astype(np.float32)
    groups = Counter()
    for row in rows:
        source = SOURCE_BY_PATH.get(str(row["path"]), "unknown")
        groups[(source, int(bool(row["y"])))] += 1

    if not groups:
        return w

    target = float(np.median(list(groups.values())))
    for i, row in enumerate(rows):
        source = SOURCE_BY_PATH.get(str(row["path"]), "unknown")
        count = max(1, groups[(source, int(bool(row["y"])))])
        correction = math.sqrt(target / count)
        correction = min(1.55, max(0.78, correction))
        w[i] *= correction
    return w


def _internal_split(dataset, seed, frac=0.10):
    labels = np.asarray([int(dataset.items[i].label > 0.5) for i in range(len(dataset))])
    rng = np.random.default_rng(seed)
    fit_idx = []
    hold_idx = []

    for label in (0, 1):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_hold = max(32, int(round(len(idx) * frac)))
        n_hold = min(max(1, len(idx) - 1), n_hold)
        hold_idx.extend(idx[:n_hold].tolist())
        fit_idx.extend(idx[n_hold:].tolist())

    rng.shuffle(fit_idx)
    rng.shuffle(hold_idx)
    return fit_idx, hold_idx


def _low_fp_checkpoint_metric(labels, probs):
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, kind="mergesort")
    yy = labels[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    spam = max(1, int((labels == 1).sum()))
    ham = max(1, int((labels == 0).sum()))

    # Internal checkpoint budget: roughly 0.1% ham, at least one FP.
    budget = max(1, int(math.floor(ham * 0.001)))
    valid = np.where(fp <= budget)[0]
    low_fp_recall = float(tp[valid[-1]] / spam) if len(valid) else 0.0
    ap = float(average_precision_score(labels, probs))
    return low_fp_recall, ap, budget


def overnight_train_model(
    name,
    train_ds,
    val_ds,
    val_rows,
    seed,
    max_epochs=32,
    patience=7,
    min_epochs=10,
    max_added_fp=0,
):
    """
    Long training with an internal checkpoint split.

    Important: the external 6.4k calibration set passed as val_ds/val_rows is
    intentionally NOT used for neural checkpoint selection. It remains for the
    later stack/gate calibration only.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    is_deep = "deep" in name.lower()
    forced_epochs = 82 if is_deep else 74
    forced_min_epochs = 54 if is_deep else 48
    forced_patience = 18 if is_deep else 16

    fit_idx, hold_idx = _internal_split(train_ds, seed, frac=0.10)
    fit_ds = Subset(train_ds, fit_idx)
    hold_ds = Subset(train_ds, hold_idx)

    model = OvernightMailNet()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.15e-3, weight_decay=8e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=forced_epochs, eta_min=7e-5
    )

    train_loader = DataLoader(
        fit_ds,
        batch_size=160,
        shuffle=True,
        collate_fn=v18.collate,
        num_workers=0,
    )

    hold_labels = np.asarray(
        [int(train_ds.items[i].label > 0.5) for i in hold_idx], dtype=np.int32
    )

    best_state = None
    best_rank = None
    best_meta = None
    stale = 0

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT_MODELS / f"{name}.pt"

    print(
        f"{name} overnight fit={len(fit_idx)} internal_holdout={len(hold_idx)} "
        f"max_epochs={forced_epochs}",
        flush=True,
    )

    for epoch in range(1, forced_epochs + 1):
        started = time.time()
        model.train()
        losses = []

        for batch in train_loader:
            words, wo, chars, co, numeric, labels, weights = batch
            optimizer.zero_grad(set_to_none=True)
            logits = model(words, wo, chars, co, numeric)
            loss = v18.weighted_focal_loss(
                logits, labels, weights, gamma=1.15
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.5)
            optimizer.step()
            losses.append(float(loss.detach()))

        scheduler.step()

        hold_probs = v18.predict_one(model, hold_ds, batch_size=320)
        low_fp_recall, ap, fp_budget = _low_fp_checkpoint_metric(
            hold_labels, hold_probs
        )
        mean_loss = float(np.mean(losses))
        rank = (low_fp_recall, ap, -mean_loss)

        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_meta = {
                "epoch": epoch,
                "internalLowFpRecall": low_fp_recall,
                "internalAveragePrecision": ap,
                "internalFpBudget": fp_budget,
                "trainLoss": mean_loss,
                "fitCount": len(fit_idx),
                "internalHoldoutCount": len(hold_idx),
            }
            torch.save(
                {
                    "state_dict": best_state,
                    "meta": best_meta,
                    "name": name,
                    "architecture": "OvernightMailNet-v33",
                },
                checkpoint,
            )
            stale = 0
        else:
            stale += 1

        print(
            f"{name} epoch={epoch:02d}/{forced_epochs} loss={mean_loss:.5f} "
            f"internal_lowfp_recall={low_fp_recall:.4%} ap={ap:.5f} "
            f"best_epoch={best_meta['epoch']} stale={stale} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )

        if epoch >= forced_min_epochs and stale >= forced_patience:
            print(f"{name} overnight early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"{name}: no neural checkpoint produced")

    model.load_state_dict(best_state)
    model.eval()
    return model, best_meta


def write_report():
    src_json = Path("reports/v24-crossfit.json")
    src_md = Path("reports/v24-crossfit.md")

    if src_json.exists():
        obj = json.loads(src_json.read_text())
        obj["version"] = "v33-overnight-neural-lowfp"
        obj["dataset"]["training"] = v28.EXPECTED_TRAIN
        obj["dataset"]["validation"] = v28.EXPECTED_VAL
        obj["dataset"]["lockedTest"] = v28.EXPECTED_TEST
        obj["dataset"]["trainingSources"] = [
            "TREC-07", "Enron", "TREC-06", "Assassin", "Ling"
        ]
        obj["dataset"]["testSources"] = ["TREC-05", "CEAS-08"]
        obj["method"]["sourceArtifactSanitization"] = True
        obj["method"]["neural"] = (
            "larger two-tower PyTorch ensemble; 3 general + 3 deep; "
            "74/82 epoch caps; internal train-only checkpoint holdouts"
        )
        obj["method"]["neuralCheckpointUsesExternalCalibration"] = False
        obj["method"]["mildSourceWeighting"] = (
            "sqrt source+label correction clipped to 0.78..1.55"
        )
        obj["method"]["gate"] = "v30 hard low-FP two-tier gate"
        obj["method"]["testLabelsUsedForTrainingOrThresholds"] = False
        obj["method"]["individualTestErrorsInspected"] = False
        obj["warning"] = (
            "TREC-05 + CEAS-08 has already been reused for engineering comparison. "
            "It is not a pristine final lockbox; validate the frozen design on a "
            "new untouched corpus before final generalization claims."
        )
        REPORT_JSON.write_text(json.dumps(obj, indent=2))

    if src_md.exists():
        txt = src_md.read_text()
        txt = txt.replace(
            "# MailGuard v24 hard-ham-veto benchmark",
            "# MailGuard v33 overnight neural low-FP 50k benchmark",
        )
        txt = txt.replace("v24 ", "v33 ")
        txt += (
            "\nV33 is the overnight neural run: a larger two-tower PyTorch network "
            "replaces the small v18 net. Three general and three deep specialists train "
            "for long schedules (74/82 epoch caps). Neural checkpoint selection uses "
            "only an internal 10% holdout from the training set, not the external "
            "6.4k calibration set. Source balancing is deliberately mild because v32 "
            "showed that full equalisation hurt recall. V31 sanitization and the v30 "
            "low-FP gate are retained.\n"
        )
        REPORT_MD.write_text(txt)


def main():
    torch.set_num_threads(min(6, torch.get_num_threads()))

    md.render = v31.sanitized_render
    md.CACHE = Path(".cache/v31-sanitized-crosssource")
    md.EML_DIR = md.CACHE / "eml"
    md.MANIFEST = md.CACHE / "manifest.json"
    md.TRAIN_PLAN = v28.TRAIN_PLAN
    md.VAL_PLAN = v28.VAL_PLAN
    md.TEST_PLAN = v28.TEST_PLAN
    md.EXPECTED_TRAIN = v28.EXPECTED_TRAIN
    md.EXPECTED_VAL = v28.EXPECTED_VAL
    md.EXPECTED_TEST = v28.EXPECTED_TEST

    data = md.prepare()

    global SOURCE_BY_PATH
    SOURCE_BY_PATH = {
        str(row["path"]): row.get("source", "unknown")
        for split in ("train", "val", "test")
        for row in data[split]
    }

    v18.enron.prepare_enron = lambda: data["train"] + data["val"] + data["test"]
    v18.split_enron = lambda _rows: (data["train"], data["val"], data["test"])
    v3.build_splits = lambda _groups: ([], [], [])

    # Save original function so mild_source_weights can call it.
    if not hasattr(v18, "_v33_original_base_weights"):
        v18._v33_original_base_weights = v18.base_weights

    v18.base_weights = mild_source_weights
    v18.DeepMailNet = OvernightMailNet
    v18.train_model = overnight_train_model
    v18.MODELS = OUT_MODELS

    # Go back to the v31 forensic branch (v32's full source balancing reduced
    # recall too much), but retain v30's low-FP gate.
    base.select_dual_guard = v30.select_balanced_guard
    base.apply_dual_guard = v30.apply_balanced_guard
    base.chronological_folds = v28.source_heldout_folds
    base.MODELS = OUT_MODELS
    base.SEED = SEED

    base.main()
    write_report()


if __name__ == "__main__":
    main()
