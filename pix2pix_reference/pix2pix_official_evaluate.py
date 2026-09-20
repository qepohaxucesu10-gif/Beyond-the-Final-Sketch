"""Frozen-checkpoint evaluator for the official pix2pix FSS reference run.

This performs inference only.  It consumes Face--B25 pairs from the frozen
canonical split transfer, writes no training state, and computes the same
thresholded ink F1/IoU, RGB L1, and Sobel edge-L1 metrics used by the FSS
reports.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pair_split(root: Path, split: str) -> tuple[Path, dict[str, Path]]:
    source = root / "pix2pix_canonical_transfer" / split
    base = root / "paired_data" / split
    output = base / "test"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    targets: dict[str, Path] = {}
    for subject_dir in sorted(source.iterdir()):
        if not subject_dir.is_dir():
            continue
        # One historical CUFS face uses the filename without the ``-00``
        # suffix.  The frozen manifest identifies the subject; B25 remains
        # unambiguous, so select the sole non-B25 JPEG as its face.
        b25 = next(subject_dir.glob("*-25.jpg"))
        faces = [p for p in subject_dir.glob("*.jpg") if p != b25]
        if len(faces) != 1:
            raise RuntimeError(f"expected one face and B25 in {subject_dir}; got {faces}")
        face = faces[0]
        uid = subject_dir.name
        with Image.open(face) as a, Image.open(b25) as b:
            a = a.convert("RGB").resize((256, 256), Image.BICUBIC)
            b = b.convert("RGB").resize((256, 256), Image.BICUBIC)
            ab = Image.new("RGB", (512, 256))
            ab.paste(a, (0, 0)); ab.paste(b, (256, 0))
            ab.save(output / f"{uid}.png")
        targets[uid] = b25
    return base, targets


def run_inference(root: Path, seed: int, split: str, pair_dir: Path, python_exe: str) -> Path:
    run = root / "runs" / f"seed{seed}"
    out = run / "predictions" / split
    if out.exists():
        shutil.rmtree(out)
    cmd = [
        python_exe, str(root / "source" / "test.py"),
        "--dataroot", str(pair_dir), "--name", f"seed{seed}",
        "--checkpoints_dir", str(run / "checkpoints"), "--results_dir", str(out),
        "--model", "pix2pix", "--dataset_mode", "aligned", "--direction", "AtoB",
        "--netG", "unet_256", "--norm", "batch", "--preprocess", "resize",
        "--load_size", "256", "--crop_size", "256", "--no_flip", "--num_threads", "0",
        "--batch_size", "1", "--num_test", "999", "--epoch", "125", "--eval",
    ]
    env = {"PYTHONPATH": str(root / "source"), "WANDB_MODE": "disabled"}
    subprocess.run(cmd, check=True, cwd=root / "source", env={**__import__("os").environ, **env})
    images = out / f"seed{seed}" / "test_125" / "images"
    if not images.exists():
        raise RuntimeError(f"missing prediction directory: {images}")
    return images


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        x = np.asarray(im.convert("RGB").resize((256, 256), Image.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def sobel(x: torch.Tensor) -> torch.Tensor:
    gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=x.dtype).view(1, 1, 3, 3)
    gy = gx.transpose(2, 3)
    x = x.mean(1, keepdim=True)
    return torch.sqrt(F.conv2d(x, gx, padding=1).square() + F.conv2d(x, gy, padding=1).square() + 1e-12)


def metric(pred: Path, target: Path) -> dict[str, float]:
    p, t = image_tensor(pred), image_tensor(target)
    pm, tm = p.mean(1) < 0.90, t.mean(1) < 0.90
    tp = float((pm & tm).sum()); fp = float((pm & ~tm).sum()); fn = float((~pm & tm).sum())
    return {"F1": 2 * tp / (2 * tp + fp + fn + 1e-8), "IoU": tp / (tp + fp + fn + 1e-8),
            "L1": float(F.l1_loss(p, t)), "EdgeLoss": float(F.l1_loss(sobel(p), sobel(t)))}


def evaluate(root: Path, seed: int, split: str, targets: dict[str, Path], image_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for uid, target in targets.items():
        matches = list(image_dir.glob(f"{uid}*_fake_B.png"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one prediction for {uid}; got {matches}")
        row: dict[str, object] = {"seed": seed, "split": split, "subject_id": uid, **metric(matches[0], target)}
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Local directory containing the official pix2pix checkout and released assets.")
    ap.add_argument("--python", default=sys.executable, help="Python executable for the official pix2pix environment.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 2026, 9317])
    args = ap.parse_args()
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for split, expected in (("valid43", 43), ("test42", 42)):
        pairs, targets = pair_split(args.root, split)
        if len(targets) != expected:
            raise RuntimeError(f"{split} count {len(targets)} != {expected}")
        for seed in args.seeds:
            images = run_inference(args.root, seed, split, pairs, args.python)
            rows = evaluate(args.root, seed, split, targets, images)
            all_rows.extend(rows)
            mean = {k: float(np.mean([float(r[k]) for r in rows])) for k in ("F1", "IoU", "L1", "EdgeLoss")}
            summaries.append({"seed": seed, "split": split, "checkpoint_epoch": 125, **mean,
                              "checkpoint_path": str(args.root / "runs" / f"seed{seed}" / "checkpoints" / f"seed{seed}" / "125_net_G.pth")})
    out = args.root / "final_evaluation"
    out.mkdir(exist_ok=True)
    with (out / "per_subject_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "split", "subject_id", "F1", "IoU", "L1", "EdgeLoss"]); w.writeheader(); w.writerows(all_rows)
    with (out / "per_seed_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    print(out / "per_seed_summary.csv")


if __name__ == "__main__":
    main()
