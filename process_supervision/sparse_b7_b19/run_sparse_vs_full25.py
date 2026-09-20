"""Full-25 versus Sparse-3-Stage Process Control v1, one seed per run."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, sys, time
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
V2 = REPORT / "direct_vs_process_mini_poc_v2.json"
IMAGE_SIZE, STEPS = 256, 4000
SEEDS = (3407, 2026, 9317)
INIT = {
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


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for k in sorted(state): h.update(k.encode()); h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def image(path):
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(a.transpose(2, 0, 1).copy()) * 2 - 1


def rows_for(split, expected):
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f: rows = list(csv.DictReader(f))
    rows = sorted([r for r in rows if r.get("split", r.get("original_split")) == split], key=lambda r: r["sample_uid"])
    good = [r for r in rows if r.get("process_only_complete", "True").lower() == "true" and r.get("face_path") and all(r.get(f"step_{k:02d}_path") for k in range(1, 26))]
    out = good[:len(expected)]
    assert [r["sample_uid"] for r in out] == expected
    return out


def load_subjects(rows):
    faces, targets = [], []
    for r in rows:
        faces.append(image(ROOT / r["face_path"]))
        targets.append(torch.stack([image(ROOT / r[f"step_{k:02d}_path"]) for k in range(1, 26)]))
    return torch.stack(faces), torch.stack(targets)


def metric(pred, target):
    pm, tm = pred.mean(1) < .90, target.mean(1) < .90
    tp = (pm & tm).sum().float(); fp = (pm & ~tm).sum().float(); fn = (~pm & tm).sum().float()
    return {"f1": float((2 * tp / (2 * tp + fp + fn + 1e-8)).cpu()), "iou": float((tp / (tp + fp + fn + 1e-8)).cpu()), "l1": float(F.l1_loss(pred, target).cpu())}


def evaluate(model, rows, faces, targets, device, sobel):
    model.eval(); out = []
    with torch.inference_mode():
        for i, r in enumerate(rows):
            pred = model(faces[i:i+1].to(device), torch.ones(1, device=device)); target = targets[i, 24:25].to(device)
            m = metric(pred, target); m.update(sample_uid=r["sample_uid"], edge_loss=float(F.l1_loss(sobel(pred), sobel(target)).cpu())); out.append(m)
    return out


def grid(path, images, titles, rows, cols):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.55, rows * 1.65)); ax = np.asarray(ax).reshape(rows, cols)
    for i, (im, title) in enumerate(zip(images, titles)):
        ax.flat[i].imshow(((im.detach().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()); ax.flat[i].set_title(title, fontsize=5); ax.flat[i].axis("off")
    for i in range(len(images), rows * cols): ax.flat[i].axis("off")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)


def old_grid_crop(seed, row, column):
    candidates = [ROOT / "rebuild_experiment/visualizations/direct_vs_process_mini_poc_v2/mini_poc_v2_all8_b25_grid.png", ROOT / "rebuild_experiment/visualizations" / f"direct_vs_process_mini_poc_seed{seed}" / "all8_b25_grid.png"]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None: return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
    with Image.open(source) as im:
        im = im.convert("RGB"); w, h = im.size
        left = w * (column / 4 + .015); right = w * ((column + 1) / 4 - .015)
        crop = im.crop((int(left), int(h * (row / 8 + .02)), int(right), int(h * ((row + 1) / 8 - .045)))).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        a = np.asarray(crop, dtype=np.float32) / 255.0
    return torch.from_numpy(a.transpose(2, 0, 1).copy()) * 2 - 1


def bootstrap(values):
    a = np.asarray(values, float); rng = np.random.default_rng(3407); b = np.array([rng.choice(a, len(a), replace=True).mean() for _ in range(10000)])
    return {"mean": float(a.mean()), "ci95_low": float(np.quantile(b, .025)), "ci95_high": float(np.quantile(b, .975))}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, required=True); a = p.parse_args(); seed = a.seed
    if seed not in SEEDS: raise ValueError(f"seed must be one of {SEEDS}")
    seed_all(seed)
    exp = ROOT / "rebuild_experiment/experiments" / f"full25_vs_sparse_process_seed{seed}"; viz = ROOT / "rebuild_experiment/visualizations" / f"full25_vs_sparse_process_seed{seed}"; exp.mkdir(parents=True, exist_ok=True); viz.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    v2 = json.loads(V2.read_text(encoding="utf-8")); train_expected, valid_expected = v2["train_uids"], v2["valid_uids"]
    train_rows, valid_rows = rows_for("train", train_expected), rows_for("valid", valid_expected); assert len(train_rows) == 32 and len(valid_rows) == 8
    ordered_report = json.loads(ORDERED_REPORT[seed].read_text(encoding="utf-8")); initial = INIT[seed]; initial_file_sha = sha(initial); assert initial_file_sha == ordered_report["initial_checkpoint_sha256"]
    payload = torch.load(initial, map_location="cpu"); state = payload["model"]; initial_state_sha = state_sha(state)
    with (ORDERED_EXP[seed] / "subject_sampling_manifest.csv").open(encoding="utf-8", newline="") as f: ordered_sampling = list(csv.DictReader(f))
    assert len(ordered_sampling) == STEPS
    train_faces, train_targets = load_subjects(train_rows); valid_faces, valid_targets = load_subjects(valid_rows); device = torch.device("cuda"); start = time.time()
    model = ProcessConditionedGenerator(); model.load_state_dict(state); assert state_sha(model.state_dict()) == initial_state_sha; model = model.to(device); sobel = SobelMagnitude().to(device); opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.5, .999)); exposure = {7: 0, 19: 0}; sampling = []; curve = []
    for step in range(1, STEPS + 1):
        ref = ordered_sampling[step - 1]; uid = ref["direct_uid"]; assert uid == ref["process_uid"]; idx = train_expected.index(uid); aux_stage = 7 if step % 2 == 1 else 19; exposure[aux_stage] += 1
        sampling.append({"optimizer_step": step, "ordered_uid": uid, "sparse_uid": uid, "sparse_aux_stage": aux_stage, "sparse_t_process": aux_stage / 25.0})
        face = train_faces[idx:idx+1].to(device); y25 = train_targets[idx, 24:25].to(device); yaux = train_targets[idx, aux_stage-1:aux_stage].to(device); opt.zero_grad(set_to_none=True)
        lf = balanced_region_l1_edge(model(face, torch.ones(1, device=device)), y25, lambda_edge=.1, sobel=sobel); la = balanced_region_l1_edge(model(face, torch.tensor([aux_stage/25.], device=device)), yaux, lambda_edge=.1, sobel=sobel); total = lf["total_loss"] + la["total_loss"]; total.backward(); opt.step()
        if step == 1 or step % 100 == 0: curve.append({"step": step, "final_loss": float(lf["total_loss"].detach().cpu()), "sparse_aux_loss": float(la["total_loss"].detach().cpu()), "total_loss": float(total.detach().cpu()), "elapsed_seconds": time.time() - start, "gpu_peak_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20)})
    audit = Path(os.environ.get("AUDIT_RUN_DIR", str(exp))); write_csv(exp / "training_curve.csv", curve); write_csv(exp / "subject_sampling_manifest.csv", sampling); write_csv(exp / "sparse_step_exposure.csv", [{"stage": k, "exposure_count": exposure[k]} for k in (7, 19)]); write_csv(audit / "12_used_samples.csv", [{"sample_uid": r["sample_uid"], "split": "train", "usage": "training", "count": 25} for r in train_rows] + [{"sample_uid": r["sample_uid"], "split": "valid", "usage": "evaluation", "count": 25} for r in valid_rows]); write_csv(audit / "trainable_parameters.csv", [{"parameter_name": f"sparse_generator_parameters.{n}", "shape": str(list(x.shape)), "numel": x.numel(), "requires_grad": x.requires_grad, "optimizer_group": "sparse_generator_parameters"} for n, x in model.named_parameters()]); assert all(x["ordered_uid"] == x["sparse_uid"] for x in sampling)
    sparse = evaluate(model, valid_rows, valid_faces, valid_targets, device, sobel); ordered_metric_path = ORDERED_EXP[seed] / ("per_subject_b25_metrics_v2.csv" if seed == 3407 else "per_subject_b25_metrics.csv")
    with ordered_metric_path.open(encoding="utf-8", newline="") as f: old = {r["sample_uid"]: r for r in csv.DictReader(f)}
    with (REPORT / f"ordered_vs_shuffled_seed{seed}.json").open(encoding="utf-8") as f: shuffled_report = json.load(f)
    paired = []
    for s in sparse:
        o = old[s["sample_uid"]]; paired.append({"sample_uid": s["sample_uid"], "direct_f1": float(o["direct_f1"]), "direct_iou": float(o["direct_iou"]), "direct_l1": float(o["direct_l1"]), "ordered_f1": float(o["process_f1"]), "ordered_iou": float(o["process_iou"]), "ordered_l1": float(o["process_l1"]), "sparse_f1": s["f1"], "sparse_iou": s["iou"], "sparse_l1": s["l1"], "sparse_edge_loss": s["edge_loss"]})
    # Attach the already audited shuffled aggregate only at the summary level; per-subject shuffled values remain in its audited CSV.
    with (ROOT / "rebuild_experiment/experiments" / f"ordered_vs_shuffled_process_seed{seed}" / "ordered_vs_shuffled_per_subject.csv").open(encoding="utf-8", newline="") as f: sh = {r["sample_uid"]: r for r in csv.DictReader(f)}
    for row in paired:
        row["shuffled_f1"] = float(sh[row["sample_uid"]]["shuffled_f1"]); row["shuffled_iou"] = float(sh[row["sample_uid"]]["shuffled_iou"]); row["shuffled_l1"] = float(sh[row["sample_uid"]]["shuffled_l1"])
        row["delta_f1_ordered_minus_sparse"] = row["ordered_f1"] - row["sparse_f1"]; row["delta_iou_ordered_minus_sparse"] = row["ordered_iou"] - row["sparse_iou"]; row["delta_l1_ordered_minus_sparse"] = row["ordered_l1"] - row["sparse_l1"]; row["delta_f1_sparse_minus_direct"] = row["sparse_f1"] - row["direct_f1"]; row["delta_iou_sparse_minus_direct"] = row["sparse_iou"] - row["direct_iou"]; row["delta_l1_sparse_minus_direct"] = row["sparse_l1"] - row["direct_l1"]
    write_csv(exp / "full25_vs_sparse_per_subject.csv", paired)
    images, titles = [], []
    for i, r in enumerate(valid_rows):
        with torch.inference_mode(): pred = model(valid_faces[i:i+1].to(device), torch.ones(1, device=device)).cpu()[0]
        images += [valid_faces[i], valid_targets[i, 24], old_grid_crop(seed, i, 2), pred, old_grid_crop(seed, i, 3)]; titles += [r["sample_uid"], "GT B25", f"Direct {paired[i]['direct_f1']:.3f}", f"Sparse {paired[i]['sparse_f1']:.3f}", f"Ordered {paired[i]['ordered_f1']:.3f}"]
    grid(viz / "full25_vs_sparse_all8.png", images, titles, 8, 5)
    if seed == 3407:
        order = np.argsort([x["delta_f1_ordered_minus_sparse"] for x in paired]);
        for name, ids in (("full25_best4_over_sparse.png", order[-4:][::-1]), ("full25_worst4_over_sparse.png", order[:4])):
            images, titles = [], []
            for j in ids:
                x = paired[int(j)]; images += [valid_faces[int(j)], valid_targets[int(j), 24], old_grid_crop(seed, int(j), 2), model(valid_faces[int(j):int(j)+1].to(device), torch.ones(1, device=device)).cpu()[0], old_grid_crop(seed, int(j), 3)]; titles += [x["sample_uid"], "GT", f"D {x['direct_f1']:.3f}", f"S {x['sparse_f1']:.3f}", f"O {x['ordered_f1']:.3f}"]
            grid(viz / name, images, titles, 4, 5)
    ordered_mean = {k: float(np.mean([x[f"ordered_{k}"] for x in paired])) for k in ("f1", "iou", "l1")}; sparse_mean = {k: float(np.mean([x[f"sparse_{k}"] for x in paired])) for k in ("f1", "iou", "l1")}; direct_mean = {k: float(np.mean([x[f"direct_{k}"] for x in paired])) for k in ("f1", "iou", "l1")}; shuffled_mean = shuffled_report["shuffled_valid_mean"]; dfo = [x["delta_f1_ordered_minus_sparse"] for x in paired]; dio = [x["delta_iou_ordered_minus_sparse"] for x in paired]; dlo = [x["delta_l1_ordered_minus_sparse"] for x in paired]
    report = {"status": "COMPLETED_AUDITED", "risk": "LOW", "experiment": f"full25_vs_sparse_process_seed{seed}", "seed": seed, "gpu": torch.cuda.get_device_name(0), "gpu_peak_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20), "train_uids": train_expected, "valid_uids": valid_expected, "requested_train_count": 32, "actual_train_count": 32, "requested_valid_count": 8, "actual_valid_count": 8, "initial_checkpoint_sha256": initial_file_sha, "ordered_initial_checkpoint_sha256": ordered_report["initial_checkpoint_sha256"], "initial_sha_equal": initial_file_sha == ordered_report["initial_checkpoint_sha256"], "subject_sampling_identical": all(x["ordered_uid"] == x["sparse_uid"] for x in sampling), "optimizer_steps": STEPS, "b25_exposure": STEPS, "auxiliary_exposure": STEPS, "sparse_exposure": exposure, "sparse_exposure_balanced": exposure[7] == exposure[19], "direct_valid_mean": direct_mean, "sparse_valid_mean": sparse_mean, "shuffled_valid_mean": shuffled_mean, "ordered_valid_mean": ordered_mean, "delta_ordered_minus_sparse": {"f1": float(np.mean(dfo)), "iou": float(np.mean(dio)), "l1": float(np.mean(dlo))}, "delta_sparse_minus_direct": {"f1": sparse_mean["f1"] - direct_mean["f1"], "iou": sparse_mean["iou"] - direct_mean["iou"], "l1": sparse_mean["l1"] - direct_mean["l1"]}, "wins_ordered_over_sparse": {"f1": sum(x > 0 for x in dfo), "iou": sum(x > 0 for x in dio), "l1": sum(x < 0 for x in dlo)}, "bootstrap_ordered_minus_sparse_f1": bootstrap(dfo), "bootstrap_ordered_minus_sparse_iou": bootstrap(dio), "test_read_count": 0, "cuhk_read_count": 0, "training": True, "elapsed_seconds": time.time() - start, "paths": {"report_json": f"rebuild_experiment/reports/full25_vs_sparse_seed{seed}.json", "report_md": f"rebuild_experiment/reports/full25_vs_sparse_seed{seed}.md", "experiment": str(exp.relative_to(ROOT)), "visualizations": str(viz.relative_to(ROOT))}}
    (REPORT / f"full25_vs_sparse_seed{seed}.json").write_text(json.dumps(report, indent=2), encoding="utf-8"); (REPORT / f"full25_vs_sparse_seed{seed}.md").write_text("# Full-25 vs Sparse-3-Stage\n\n" + json.dumps(report, indent=2) + "\n", encoding="utf-8"); (audit / "metrics.csv").write_text("epoch,train_loss,valid_loss,valid_F1,valid_IoU,learning_rate,elapsed_time,GPU_memory,checkpoint_saved\n0,0,,0,0,0," + str(report["elapsed_seconds"]) + "," + str(report["gpu_peak_memory_mb"]) + ",false\n", encoding="utf-8"); print(f"EFFECTIVE_SEED={seed}"); print(json.dumps({"status": report["status"], "risk": report["risk"], "seed": seed, "ordered_minus_sparse": report["delta_ordered_minus_sparse"]}))


if __name__ == "__main__": main()
