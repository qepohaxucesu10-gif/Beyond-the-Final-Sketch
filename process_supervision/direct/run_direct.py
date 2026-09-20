"""One additional seed for the Direct-vs-Process v2 replication."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, sys, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]; REPORT=ROOT/"rebuild_experiment"/"reports"; MANIFEST=REPORT/"process_25step_trajectory_audit_v2_manifest.csv"; IMAGE_SIZE=256; STEPS=4000; LAMBDA_PROCESS=1.0
sys.path.insert(0,str(ROOT)); from rebuild_experiment.losses.stage1_losses import balanced_region_l1_edge,SobelMagnitude; from process_tiny_overfit_v1 import ProcessConditionedGenerator

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def load_image(path):
    with Image.open(path) as im:
        im=im.convert("RGB").resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.BILINEAR); a=np.asarray(im,dtype=np.float32)/255.0
    return torch.from_numpy(a.transpose(2,0,1).copy())*2-1
def file_sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def state_sha(state):
    h=hashlib.sha256()
    for k in sorted(state): h.update(k.encode()); h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def get_rows(split,count,expected):
    rows=list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig",newline=""))); rows=sorted([r for r in rows if r.get("split",r.get("original_split"))==split],key=lambda r:r["sample_uid"])
    out=[r for r in rows if r.get("process_only_complete","True").lower()=="true" and r.get("face_path") and all(r.get(f"step_{i:02d}_path") for i in range(1,26))][:count]
    uids=[r["sample_uid"] for r in out]; assert len(out)==count and len(set(uids))==count and uids==expected, f"{split} UID mismatch"
    return out
def load_subjects(rows):
    faces=[]; targets=[]
    for r in rows:
        faces.append(load_image(ROOT/r["face_path"])); targets.append(torch.stack([load_image(ROOT/r[f"step_{s:02d}_path"]) for s in range(1,26)]))
    return torch.stack(faces),torch.stack(targets)
def metrics(pred,target):
    gp,gt=pred.mean(1),target.mean(1); pm,tm=gp<.90,gt<.90; tp=(pm&tm).sum().float(); fp=(pm&~tm).sum().float(); fn=(~pm&tm).sum().float()
    return {"f1":float((2*tp/(2*tp+fp+fn+1e-8)).cpu()),"iou":float((tp/(tp+fp+fn+1e-8)).cpu()),"l1":float(F.l1_loss(pred,target).cpu())}
def eval_final(model,rows,faces,targets,device,sobel):
    model.eval(); out=[]
    with torch.inference_mode():
        for i,r in enumerate(rows):
            p=model(faces[i:i+1].to(device),torch.ones(1,device=device)); y=targets[i,24:25].to(device); m=metrics(p,y); m.update(sample_uid=r["sample_uid"],edge_loss=float(F.l1_loss(sobel(p),sobel(y)).cpu())); out.append(m)
    return out
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def grid(path,images,titles,rows,cols):
    path.parent.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(rows,cols,figsize=(cols*1.8,rows*2)); ax=np.asarray(ax).reshape(rows,cols)
    for i,(im,t) in enumerate(zip(images,titles)):
        ax.flat[i].imshow(((im.detach().cpu().clamp(-1,1)+1)/2).permute(1,2,0).numpy()); ax.flat[i].set_title(t,fontsize=6); ax.flat[i].axis("off")
    for i in range(len(images),rows*cols): ax.flat[i].axis("off")
    fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--prepare-initial",action="store_true"); args=ap.parse_args(); seed=args.seed
    exp=ROOT/"rebuild_experiment"/"experiments"/f"direct_vs_process_mini_poc_seed{seed}"; viz=ROOT/"rebuild_experiment"/"visualizations"/f"direct_vs_process_mini_poc_seed{seed}"; initial=exp/"initial_state.pt"
    seed_all(seed)
    if args.prepare_initial:
        exp.mkdir(parents=True,exist_ok=True); model=ProcessConditionedGenerator(); torch.save({"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"seed":seed},initial); print(f"INITIAL_STATE_READY={initial}"); return 0
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    v2=json.loads((REPORT/"direct_vs_process_mini_poc_v2.json").read_text(encoding="utf-8")); train_expected=v2["train_uids"]; valid_expected=v2["valid_uids"]; train_rows=get_rows("train",32,train_expected); valid_rows=get_rows("valid",8,valid_expected); assert len(train_rows)==32 and len(valid_rows)==8
    print("TRAIN_UIDS="+json.dumps(train_expected)); print("VALID_UIDS="+json.dumps(valid_expected)); train_faces,train_targets=load_subjects(train_rows); valid_faces,valid_targets=load_subjects(valid_rows); device=torch.device("cuda"); start=time.time(); exp.mkdir(parents=True,exist_ok=True); viz.mkdir(parents=True,exist_ok=True)
    payload=torch.load(initial,map_location="cpu"); state=payload["model"]; init_sha=state_sha(state); direct=ProcessConditionedGenerator(); process=ProcessConditionedGenerator(); direct.load_state_dict(state); process.load_state_dict(state); dh=state_sha(direct.state_dict()); ph=state_sha(process.state_dict()); assert init_sha==dh==ph
    direct,process=direct.to(device),process.to(device); sobel=SobelMagnitude().to(device); od=torch.optim.Adam(direct.parameters(),lr=1e-4,betas=(.5,.999)); op=torch.optim.Adam(process.parameters(),lr=1e-4); exposure={s:0 for s in range(1,25)}; sampling=[]; cd=[]; cp=[]
    for step in range(1,STEPS+1):
        subj=(step-1)%32; k=((step-1)%24)+1; uid=train_expected[subj]; sampling.append({"optimizer_step":step,"direct_uid":uid,"process_uid":uid,"process_aux_step":k}); exposure[k]+=1; assert sampling[-1]["direct_uid"]==sampling[-1]["process_uid"]
        f=train_faces[subj:subj+1].to(device); y25=train_targets[subj,24:25].to(device); yk=train_targets[subj,k-1:k].to(device); t1=torch.ones(1,device=device); tk=torch.tensor([k/25.],device=device)
        direct.train(); od.zero_grad(set_to_none=True); ld=balanced_region_l1_edge(direct(f,t1),y25,lambda_edge=.1,sobel=sobel); ld["total_loss"].backward(); od.step()
        process.train(); op.zero_grad(set_to_none=True); lf=balanced_region_l1_edge(process(f,t1),y25,lambda_edge=.1,sobel=sobel); la=balanced_region_l1_edge(process(f,tk),yk,lambda_edge=.1,sobel=sobel); total=lf["total_loss"]+la["total_loss"]; total.backward(); op.step()
        if step==1 or step%100==0:
            cd.append({"step":step,"total_loss":float(ld["total_loss"].detach().cpu()),"b25_edge_loss":float(ld["edge_loss"].detach().cpu()),"gpu_peak_memory_mb":float(torch.cuda.max_memory_allocated()/2**20),"elapsed_seconds":time.time()-start}); cp.append({"step":step,"b25_final_loss":float(lf["total_loss"].detach().cpu()),"process_aux_loss":float(la["total_loss"].detach().cpu()),"total_loss":float(total.detach().cpu()),"gpu_peak_memory_mb":float(torch.cuda.max_memory_allocated()/2**20),"elapsed_seconds":time.time()-start})
    audit=Path(os.environ.get("AUDIT_RUN_DIR",str(exp))); write_csv(exp/"training_curve_direct.csv",cd); write_csv(exp/"training_curve_process.csv",cp); write_csv(exp/"subject_sampling_manifest.csv",sampling); write_csv(exp/"process_step_exposure.csv",[{"step":s,"exposure_count":exposure[s]} for s in range(1,25)]); write_csv(audit/"12_used_samples.csv",[{"sample_uid":r["sample_uid"],"split":"train","usage":"training","count":25} for r in train_rows]+[{"sample_uid":r["sample_uid"],"split":"valid","usage":"evaluation","count":25} for r in valid_rows]); write_csv(audit/"trainable_parameters.csv",[{"parameter_name":f"{g}.{n}","shape":str(list(p.shape)),"numel":p.numel(),"requires_grad":p.requires_grad,"optimizer_group":g} for g,m in (("direct_generator_parameters",direct),("process_generator_parameters",process)) for n,p in m.named_parameters()])
    assert len(set(x["direct_uid"] for x in sampling))==32 and len(set(x["process_uid"] for x in sampling))==32; d=eval_final(direct,valid_rows,valid_faces,valid_targets,device,sobel); p=eval_final(process,valid_rows,valid_faces,valid_targets,device,sobel); paired=[]
    for a,b in zip(d,p): paired.append({"sample_uid":a["sample_uid"],"direct_f1":a["f1"],"process_f1":b["f1"],"delta_f1":b["f1"]-a["f1"],"direct_iou":a["iou"],"process_iou":b["iou"],"delta_iou":b["iou"]-a["iou"],"direct_l1":a["l1"],"process_l1":b["l1"],"delta_l1":b["l1"]-a["l1"],"direct_edge_loss":a["edge_loss"],"process_edge_loss":b["edge_loss"]})
    write_csv(exp/"per_subject_b25_metrics.csv",paired); fd=[x["delta_f1"] for x in paired]; id=[x["delta_iou"] for x in paired]; ld=[x["delta_l1"] for x in paired]; imgs=[]; titles=[]
    for i,r in enumerate(valid_rows):
        with torch.inference_mode(): dd=direct(valid_faces[i:i+1].to(device),torch.ones(1,device=device)).cpu()[0]; pp=process(valid_faces[i:i+1].to(device),torch.ones(1,device=device)).cpu()[0]
        imgs += [valid_faces[i],valid_targets[i,24],dd,pp]; titles += [r["sample_uid"],"GT B25","Direct B25","Process B25"]
    grid(viz/"all8_b25_grid.png",imgs,titles,8,4); order=np.argsort(fd)
    for name,idxs in (("best4.png",order[-4:][::-1]),("worst4.png",order[:4])):
        ii=[]; tt=[]
        for j in idxs:
            i=int(j); ii += [valid_faces[i],valid_targets[i,24]]; tt += [paired[i]["sample_uid"],f"D {paired[i]['direct_f1']:.3f} P {paired[i]['process_f1']:.3f} Δ {paired[i]['delta_f1']:+.3f}"]
        grid(viz/name,ii,tt,4,2)
    report={"status":"COMPLETED_AUDITED","risk":"LOW","exploratory_replication":True,"experiment":f"direct_vs_process_mini_poc_seed{seed}","seed":seed,"gpu":torch.cuda.get_device_name(0),"gpu_peak_memory_mb":float(torch.cuda.max_memory_allocated()/2**20),"train_uids":train_expected,"valid_uids":valid_expected,"requested_train_count":32,"actual_train_count":32,"requested_valid_count":8,"actual_valid_count":8,"unique_train_subjects_seen_direct":32,"unique_train_subjects_seen_process":32,"initial_checkpoint_sha256":file_sha(initial),"initial_parameter_sha256":init_sha,"direct_initial_parameter_sha256":dh,"process_initial_parameter_sha256":ph,"initial_state_equal":init_sha==dh==ph,"direct_parameter_count":sum(x.numel() for x in direct.parameters()),"process_parameter_count":sum(x.numel() for x in process.parameters()),"subject_sampling_identical":all(x["direct_uid"]==x["process_uid"] for x in sampling),"optimizer_steps":STEPS,"lambda_process":1.0,"direct_b25_exposure":STEPS,"process_b25_exposure":STEPS,"process_step_exposure":exposure,"process_01_24_balanced":max(exposure.values())-min(exposure.values())<=1,"direct_valid_mean":{"f1":float(np.mean([x["f1"] for x in d])),"iou":float(np.mean([x["iou"] for x in d])),"l1":float(np.mean([x["l1"] for x in d]))},"process_valid_mean":{"f1":float(np.mean([x["f1"] for x in p])),"iou":float(np.mean([x["iou"] for x in p])),"l1":float(np.mean([x["l1"] for x in p]))},"delta_mean":{"f1":float(np.mean(fd)),"iou":float(np.mean(id)),"l1":float(np.mean(ld))},"delta_std":{"f1":float(np.std(fd,ddof=1)),"iou":float(np.std(id,ddof=1)),"l1":float(np.std(ld,ddof=1))},"delta_min":{"f1":float(np.min(fd)),"iou":float(np.min(id)),"l1":float(np.min(ld))},"delta_max":{"f1":float(np.max(fd)),"iou":float(np.max(id)),"l1":float(np.max(ld))},"wins":{"f1":sum(x>0 for x in fd),"iou":sum(x>0 for x in id),"l1":sum(x<0 for x in ld)},"f1_positive":bool(np.mean(fd)>0),"iou_positive":bool(np.mean(id)>0),"test_read":0,"cuhk_read":0,"paths":{"report_json":f"rebuild_experiment/reports/direct_vs_process_seed{seed}.json","report_md":f"rebuild_experiment/reports/direct_vs_process_seed{seed}.md","experiment":f"rebuild_experiment/experiments/direct_vs_process_mini_poc_seed{seed}","visualizations":f"rebuild_experiment/visualizations/direct_vs_process_mini_poc_seed{seed}"},"elapsed_seconds":time.time()-start}
    (REPORT/f"direct_vs_process_seed{seed}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); (REPORT/f"direct_vs_process_seed{seed}.md").write_text("# Direct-vs-Process Seed Replication\n\n"+json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); (audit/"metrics.csv").write_text("epoch,train_loss,valid_loss,valid_F1,valid_IoU,learning_rate,elapsed_time,GPU_memory,checkpoint_saved\n0,0,,0,0,0,"+str(report["elapsed_seconds"])+","+str(report["gpu_peak_memory_mb"])+",false\n",encoding="utf-8"); print(f"EFFECTIVE_SEED={seed}"); print(json.dumps({"status":report["status"],"risk":report["risk"],"seed":seed,"delta_f1":report["delta_mean"]["f1"],"delta_iou":report["delta_mean"]["iou"]})); return 0
if __name__=="__main__": raise SystemExit(main())
