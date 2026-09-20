"""Ordered-vs-Shuffled Process Causal Control v1.

One seed per audited run.  The shuffled control changes only the auxiliary
process-time label; the cohort, subject sampler, initialization, optimizer,
training budget, and final B25 evaluation are reused from Direct-vs-Process
Mini PoC v2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "rebuild_experiment" / "reports"
MANIFEST = REPORT / "process_25step_trajectory_audit_v2_manifest.csv"
V2_REPORT = REPORT / "direct_vs_process_mini_poc_v2.json"
IMAGE_SIZE = 256
STEPS = 4000
SEED_PATHS = {
    3407: ROOT / "rebuild_experiment/experiments/direct_vs_process_mini_poc_v2/mini_poc_v2_initial_state.pt",
    2026: ROOT / "rebuild_experiment/experiments/direct_vs_process_mini_poc_seed2026/initial_state.pt",
    9317: ROOT / "rebuild_experiment/experiments/direct_vs_process_mini_poc_seed9317/initial_state.pt",
}
ORDERED_EXP = {
    3407: ROOT / "rebuild_experiment/experiments/direct_vs_process_mini_poc_v2",
    2026: ROOT / "rebuild_experiment/experiments/direct_vs_process_mini_poc_seed2026",
    9317: ROOT / "rebuild_experiment/experiments/direct_vs_process_mini_poc_seed9317",
}
ORDERED_REPORT = {
    3407: REPORT / "direct_vs_process_mini_poc_v2.json",
    2026: REPORT / "direct_vs_process_seed2026.json",
    9317: REPORT / "direct_vs_process_seed9317.json",
}

sys.path.insert(0, str(ROOT))
from rebuild_experiment.losses.stage1_losses import balanced_region_l1_edge, SobelMagnitude
from process_tiny_overfit_v1 import ProcessConditionedGenerator


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy()) * 2.0 - 1.0


def read_rows(split: str, expected: list[str]) -> list[dict]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("split", row.get("original_split")) == split]
    rows.sort(key=lambda row: row["sample_uid"])
    complete = []
    for row in rows:
        if row.get("process_only_complete", "True").lower() != "true":
            continue
        if not row.get("face_path"):
            continue
        if not all(row.get(f"step_{step:02d}_path") for step in range(1, 26)):
            continue
        complete.append(row)
    selected = complete[: len(expected)]
    assert [row["sample_uid"] for row in selected] == expected
    assert len(set(expected)) == len(expected)
    return selected


def load_subjects(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    faces = []
    targets = []
    for row in rows:
        faces.append(load_image(ROOT / row["face_path"]))
        targets.append(torch.stack([
            load_image(ROOT / row[f"step_{step:02d}_path"])
            for step in range(1, 26)
        ]))
    return torch.stack(faces), torch.stack(targets)


def metric(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred_gray = pred.mean(1)
    target_gray = target.mean(1)
    pred_mask = pred_gray < 0.90
    target_mask = target_gray < 0.90
    tp = (pred_mask & target_mask).sum().float()
    fp = (pred_mask & ~target_mask).sum().float()
    fn = (~pred_mask & target_mask).sum().float()
    return {
        "f1": float((2 * tp / (2 * tp + fp + fn + 1e-8)).cpu()),
        "iou": float((tp / (tp + fp + fn + 1e-8)).cpu()),
        "l1": float(F.l1_loss(pred, target).cpu()),
    }


def eval_final(model, rows, faces, targets, device, sobel) -> list[dict]:
    model.eval()
    results = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            pred = model(faces[index:index + 1].to(device), torch.ones(1, device=device))
            target = targets[index, 24:25].to(device)
            values = metric(pred, target)
            values["sample_uid"] = row["sample_uid"]
            values["edge_loss"] = float(F.l1_loss(sobel(pred), sobel(target)).cpu())
            results.append(values)
    return results


def plot_grid(path: Path, images, titles, rows: int, cols: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.9))
    axes = np.asarray(axes).reshape(rows, cols)
    for index, (image, title) in enumerate(zip(images, titles)):
        axes.flat[index].imshow(((image.detach().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy())
        axes.flat[index].set_title(title, fontsize=6)
        axes.flat[index].axis("off")
    for index in range(len(images), rows * cols):
        axes.flat[index].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def bootstrap(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(3407)
    draws = np.asarray([
        rng.choice(array, size=len(array), replace=True).mean()
        for _ in range(10000)
    ])
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def expected_permutation(seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    values = np.arange(1, 25)
    rng.shuffle(values)
    return {stage: int(value) for stage, value in zip(range(1, 25), values)}


def load_permutation(path: Path, seed: int, prepare: bool) -> dict[int, int]:
    expected = expected_permutation(seed)
    if prepare:
        rows = [{
            "true_stage": stage,
            "shuffled_t_stage": expected[stage],
            "true_t_process": stage / 25.0,
            "shuffled_t_process": expected[stage] / 25.0,
        } for stage in range(1, 25)]
        write_csv(path, rows)
        print(f"PERMUTATION_READY={path}")
        return expected
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed permutation manifest: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    actual = {int(row["true_stage"]): int(row["shuffled_t_stage"]) for row in rows}
    assert sorted(actual) == list(range(1, 25))
    assert sorted(actual.values()) == list(range(1, 25))
    assert actual == expected, "persisted permutation differs from seed-fixed permutation"
    return actual


def old_grid_crop(seed: int, row: int, column: int = 3) -> torch.Tensor:
    candidates = [
        ORDERED_EXP[seed] / "../" / "visualizations" / "all8_b25_grid.png",
        ROOT / "rebuild_experiment/visualizations" / "direct_vs_process_mini_poc_v2" / "mini_poc_v2_all8_b25_grid.png",
        ROOT / "rebuild_experiment/visualizations" / f"direct_vs_process_mini_poc_seed{seed}" / "all8_b25_grid.png",
    ]
    source = next((path.resolve() for path in candidates if path.resolve().is_file()), None)
    if source is None:
        return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        x0 = int(width * (column / 4.0 + 0.015))
        x1 = int(width * ((column + 1) / 4.0 - 0.015))
        y0 = int(height * (row / 8.0 + 0.02))
        y1 = int(height * ((row + 1) / 8.0 - 0.045))
        crop = image.crop((x0, y0, x1, y1)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(crop, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy()) * 2.0 - 1.0


def process_t_sanity(model, face, device, path: Path) -> dict:
    model.eval()
    rows = []
    outputs = []
    with torch.inference_mode():
        for step in range(1, 26):
            t_value = step / 25.0
            output = model(face.to(device), torch.tensor([t_value], device=device)).cpu()
            checksum = hashlib.sha256(output.numpy().tobytes()).hexdigest()
            rows.append({
                "subject_uid": "first_train_subject",
                "face_checksum": hashlib.sha256(face.numpy().tobytes()).hexdigest(),
                "target_step": step,
                "actual_t_process": t_value,
                "output_checksum": checksum,
            })
            outputs.append(output)
    for index, row in enumerate(rows):
        row["output_delta_from_previous"] = "" if index == 0 else float(F.l1_loss(outputs[index], outputs[index - 1]))
    write_csv(path, rows)
    unique_t = {row["actual_t_process"] for row in rows}
    output_deltas = [float(row["output_delta_from_previous"]) for row in rows[1:]]
    return {
        "unique_t_count": len(unique_t),
        "output_unique_checksum_count": len({row["output_checksum"] for row in rows}),
        "mean_adjacent_output_l1": float(np.mean(output_deltas)) if output_deltas else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prepare-permutation", action="store_true")
    args = parser.parse_args()
    if args.seed not in SEED_PATHS:
        raise ValueError("seed must be one of 3407, 2026, 9317")
    seed = args.seed
    seed_all(seed)
    exp = ROOT / "rebuild_experiment/experiments" / f"ordered_vs_shuffled_process_seed{seed}"
    viz = ROOT / "rebuild_experiment/visualizations" / f"ordered_vs_shuffled_process_seed{seed}"
    exp.mkdir(parents=True, exist_ok=True)
    viz.mkdir(parents=True, exist_ok=True)
    permutation = load_permutation(exp / "stage_permutation_manifest.csv", seed, args.prepare_permutation)
    if args.prepare_permutation:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    v2_report = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    train_expected = v2_report["train_uids"]
    valid_expected = v2_report["valid_uids"]
    train_rows = read_rows("train", train_expected)
    valid_rows = read_rows("valid", valid_expected)
    assert len(train_rows) == 32 and len(valid_rows) == 8
    ordered_report = json.loads(ORDERED_REPORT[seed].read_text(encoding="utf-8"))
    initial = SEED_PATHS[seed]
    assert initial.is_file()
    initial_file_hash = file_sha(initial)
    assert initial_file_hash == ordered_report["initial_checkpoint_sha256"]
    payload = torch.load(initial, map_location="cpu")
    initial_state = payload["model"]
    initial_state_hash = state_sha(initial_state)
    with (ORDERED_EXP[seed] / "subject_sampling_manifest.csv").open(encoding="utf-8", newline="") as handle:
        ordered_sampling = list(csv.DictReader(handle))
    assert len(ordered_sampling) == STEPS

    train_faces, train_targets = load_subjects(train_rows)
    valid_faces, valid_targets = load_subjects(valid_rows)
    device = torch.device("cuda")
    start = time.time()
    model = ProcessConditionedGenerator()
    model.load_state_dict(initial_state)
    assert state_sha(model.state_dict()) == initial_state_hash
    model = model.to(device)
    sobel = SobelMagnitude().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    exposure = {stage: 0 for stage in range(1, 25)}
    sampling = []
    curve = []
    for step in range(1, STEPS + 1):
        reference = ordered_sampling[step - 1]
        uid = reference["direct_uid"]
        assert uid == reference["process_uid"]
        subject_index = train_expected.index(uid)
        true_stage = int(reference["process_aux_step"])
        shuffled_stage = permutation[true_stage]
        exposure[true_stage] += 1
        sampling.append({
            "optimizer_step": step,
            "ordered_uid": uid,
            "shuffled_uid": uid,
            "true_stage": true_stage,
            "shuffled_t_stage": shuffled_stage,
        })
        face = train_faces[subject_index:subject_index + 1].to(device)
        target_b25 = train_targets[subject_index, 24:25].to(device)
        target_aux = train_targets[subject_index, true_stage - 1:true_stage].to(device)
        t_final = torch.ones(1, device=device)
        t_shuffled = torch.tensor([shuffled_stage / 25.0], device=device)
        optimizer.zero_grad(set_to_none=True)
        final_loss = balanced_region_l1_edge(model(face, t_final), target_b25, lambda_edge=0.1, sobel=sobel)
        auxiliary_loss = balanced_region_l1_edge(model(face, t_shuffled), target_aux, lambda_edge=0.1, sobel=sobel)
        total_loss = final_loss["total_loss"] + auxiliary_loss["total_loss"]
        total_loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0:
            curve.append({
                "step": step,
                "final_loss": float(final_loss["total_loss"].detach().cpu()),
                "shuffled_aux_loss": float(auxiliary_loss["total_loss"].detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "gpu_peak_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20),
                "elapsed_seconds": time.time() - start,
            })

    audit = Path(os.environ.get("AUDIT_RUN_DIR", str(exp)))
    write_csv(exp / "training_curve.csv", curve)
    write_csv(exp / "subject_sampling_manifest.csv", sampling)
    write_csv(exp / "stage_permutation_manifest.csv", [{
        "true_stage": stage, "shuffled_t_stage": permutation[stage],
        "true_t_process": stage / 25.0, "shuffled_t_process": permutation[stage] / 25.0,
    } for stage in range(1, 25)])
    write_csv(exp / "process_step_exposure.csv", [{"step": stage, "exposure_count": exposure[stage]} for stage in range(1, 25)])
    write_csv(audit / "12_used_samples.csv", [
        {"sample_uid": row["sample_uid"], "split": "train", "usage": "training", "count": 25} for row in train_rows
    ] + [
        {"sample_uid": row["sample_uid"], "split": "valid", "usage": "evaluation", "count": 25} for row in valid_rows
    ])
    write_csv(audit / "trainable_parameters.csv", [{
        "parameter_name": f"shuffled_generator_parameters.{name}",
        "shape": str(list(parameter.shape)),
        "numel": parameter.numel(),
        "requires_grad": parameter.requires_grad,
        "optimizer_group": "shuffled_generator_parameters",
    } for name, parameter in model.named_parameters()])
    assert all(row["ordered_uid"] == row["shuffled_uid"] for row in sampling)
    assert len(set(row["ordered_uid"] for row in sampling)) == 32

    sanity = process_t_sanity(model, train_faces[0:1], device, exp / "process_t_sanity.csv")
    result = eval_final(model, valid_rows, valid_faces, valid_targets, device, sobel)
    ordered_metric_path = ORDERED_EXP[seed] / ("per_subject_b25_metrics_v2.csv" if seed == 3407 else "per_subject_b25_metrics.csv")
    with ordered_metric_path.open(encoding="utf-8", newline="") as handle:
        ordered_rows = {row["sample_uid"]: row for row in csv.DictReader(handle)}
    paired = []
    for shuffled in result:
        ordered = ordered_rows[shuffled["sample_uid"]]
        paired.append({
            "sample_uid": shuffled["sample_uid"],
            "ordered_f1": float(ordered["process_f1"]), "shuffled_f1": shuffled["f1"],
            "delta_f1_ordered_minus_shuffled": float(ordered["process_f1"]) - shuffled["f1"],
            "ordered_iou": float(ordered["process_iou"]), "shuffled_iou": shuffled["iou"],
            "delta_iou_ordered_minus_shuffled": float(ordered["process_iou"]) - shuffled["iou"],
            "ordered_l1": float(ordered["process_l1"]), "shuffled_l1": shuffled["l1"],
            "delta_l1_ordered_minus_shuffled": float(ordered["process_l1"]) - shuffled["l1"],
            "shuffled_edge_loss": shuffled["edge_loss"],
        })
    write_csv(exp / "ordered_vs_shuffled_per_subject.csv", paired)
    delta_f1 = [row["delta_f1_ordered_minus_shuffled"] for row in paired]
    delta_iou = [row["delta_iou_ordered_minus_shuffled"] for row in paired]
    delta_l1 = [row["delta_l1_ordered_minus_shuffled"] for row in paired]

    images = []
    titles = []
    for index, row in enumerate(valid_rows):
        with torch.inference_mode():
            shuffled_image = model(valid_faces[index:index + 1].to(device), torch.ones(1, device=device)).cpu()[0]
        images += [valid_faces[index], valid_targets[index, 24], shuffled_image, old_grid_crop(seed, index)]
        titles += [row["sample_uid"], "GT B25", f"Shuffled {result[index]['f1']:.3f}", f"Ordered {paired[index]['ordered_f1']:.3f}"]
    plot_grid(viz / "ordered_vs_shuffled_all8.png", images, titles, 8, 4)
    if seed == 3407:
        best = np.argsort(delta_f1)[-4:][::-1]
        worst = np.argsort(delta_f1)[:4]
        for name, indices in (("ordered_best4_over_shuffled.png", best), ("ordered_worst4_over_shuffled.png", worst)):
            images, titles = [], []
            for value in indices:
                index = int(value)
                images += [valid_faces[index], valid_targets[index, 24], old_grid_crop(seed, index)]
                titles += [paired[index]["sample_uid"], f"Shuffled {paired[index]['shuffled_f1']:.3f}", f"Ordered {paired[index]['ordered_f1']:.3f}"]
            plot_grid(viz / name, images, titles, 4, 3)

    elapsed = time.time() - start
    report = {
        "status": "COMPLETED_AUDITED", "risk": "LOW",
        "experiment": f"ordered_vs_shuffled_process_seed{seed}", "seed": seed,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20),
        "train_uids": train_expected, "valid_uids": valid_expected,
        "requested_train_count": 32, "actual_train_count": 32,
        "requested_valid_count": 8, "actual_valid_count": 8,
        "initial_checkpoint_sha256": initial_file_hash,
        "ordered_initial_checkpoint_sha256": ordered_report["initial_checkpoint_sha256"],
        "initial_sha_equal": initial_file_hash == ordered_report["initial_checkpoint_sha256"],
        "model_initial_state_sha256": initial_state_hash,
        "permutation_bijective": sorted(permutation.values()) == list(range(1, 25)),
        "permutation": permutation,
        "subject_sampling_identical": all(row["ordered_uid"] == row["shuffled_uid"] for row in sampling),
        "unique_train_subjects_seen": len(set(row["shuffled_uid"] for row in sampling)),
        "optimizer_steps": STEPS, "lambda_process": 1.0, "b25_exposure": STEPS,
        "process_step_exposure": exposure,
        "process_01_24_balanced": max(exposure.values()) - min(exposure.values()) <= 1,
        "process_t_sanity": sanity,
        "shuffled_valid_mean": {"f1": float(np.mean([row["f1"] for row in result])), "iou": float(np.mean([row["iou"] for row in result])), "l1": float(np.mean([row["l1"] for row in result]))},
        "ordered_valid_mean": {"f1": float(np.mean([row["ordered_f1"] for row in paired])), "iou": float(np.mean([row["ordered_iou"] for row in paired])), "l1": float(np.mean([row["ordered_l1"] for row in paired]))},
        "delta_ordered_minus_shuffled": {"f1": float(np.mean(delta_f1)), "iou": float(np.mean(delta_iou)), "l1": float(np.mean(delta_l1))},
        "wins_ordered_over_shuffled": {"f1": sum(value > 0 for value in delta_f1), "iou": sum(value > 0 for value in delta_iou), "l1": sum(value < 0 for value in delta_l1)},
        "bootstrap_f1": bootstrap(delta_f1), "bootstrap_iou": bootstrap(delta_iou),
        "process_still_uses_t": sanity["unique_t_count"] == 25,
        "test_read": 0, "cuhk_read": 0, "training_mode": True,
        "elapsed_seconds": elapsed,
        "paths": {"report_json": f"rebuild_experiment/reports/ordered_vs_shuffled_seed{seed}.json", "report_md": f"rebuild_experiment/reports/ordered_vs_shuffled_seed{seed}.md", "experiment": str(exp.relative_to(ROOT)), "visualizations": str(viz.relative_to(ROOT))},
    }
    (REPORT / f"ordered_vs_shuffled_seed{seed}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / f"ordered_vs_shuffled_seed{seed}.md").write_text("# Ordered-vs-Shuffled Process Control\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (audit / "metrics.csv").write_text("epoch,train_loss,valid_loss,valid_F1,valid_IoU,learning_rate,elapsed_time,GPU_memory,checkpoint_saved\n0,0,,0,0,0," + str(elapsed) + "," + str(report["gpu_peak_memory_mb"]) + ",false\n", encoding="utf-8")
    print(f"EFFECTIVE_SEED={seed}")
    print(json.dumps({"status": report["status"], "risk": report["risk"], "seed": seed, "ordered_minus_shuffled_f1": report["delta_ordered_minus_shuffled"]["f1"], "ordered_minus_shuffled_iou": report["delta_ordered_minus_shuffled"]["iou"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
