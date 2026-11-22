"""Masked four-head training and predicted-state physical output evaluation."""
import csv
import json
from itertools import islice
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .architectures import build_model
from .contact_data import (ContactPreparedDataset, DEFAULT_CONTACT_CACHE, contact_summary,
                           fit_contact_statistics, read_contact_prepared)
from .data import write_json
from .dual_rate_data import TARGET_NAMES, OrderedBatchSampler, feature_config, split_prepared
from .engine import choose_device, load_checkpoint, save_checkpoint, seed_everything

MODEL_NAME = "model_contact_gated"
LOSS_NAMES = ("position", "wrench", "slip", "contact", "zero_wrench")


def make_contact_loader(records, stats, batch_size, workers=0, training=False, seed=42):
    return DataLoader(ContactPreparedDataset(records, stats),
                      batch_sampler=OrderedBatchSampler(records, batch_size, training, seed),
                      num_workers=workers)


def masked_huber(prediction, target, valid):
    mask = valid.expand_as(prediction)
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    return torch.where(mask, error, 0.).sum()/mask.sum().clamp_min(1)


def contact_losses(model, result, batch, class_weights):
    """Ground-truth masks supervise raw heads, never the predicted hard gate."""
    target, valid = batch["target"], batch["target_valid"].bool()
    losses = {key: masked_huber(result[key], target[..., sl], valid[..., sl]) for key, sl in
              (("position", slice(0,3)), ("wrench", slice(3,9)), ("slip", slice(9,11)))}
    contact, cv = batch["contact"].long(), batch["contact_valid"].bool()
    bce = F.binary_cross_entropy_with_logits(result["contact_logits"], contact.to(target.dtype), reduction="none")
    weights = torch.as_tensor(class_weights, device=target.device, dtype=target.dtype)[contact]
    losses["contact"] = torch.where(cv, bce*weights, 0.).sum()/cv.sum().clamp_min(1)
    # Physical zero is -mean/std in normalized head space. Work explicitly in
    # physical units BEFORE any predicted gate; scale N and N*m independently.
    wrench = result["wrench"]*model.target_std[3:9]+model.target_mean[3:9]
    zero_valid = (cv & (contact == 0))[..., None]
    losses["zero_wrench"] = masked_huber(wrench/model.target_std[3:9], torch.zeros_like(wrench), zero_valid)
    return losses


def _dimension_metrics(pred, true, valid, squared=False):
    error = (pred-true)**2 if squared else np.abs(pred-true)
    count = valid.sum(0)
    values = (error*valid).sum(0)/np.maximum(count,1)
    if squared:
        values = np.sqrt(values)
    return {name: float(v) if n else None for name, v, n in zip(TARGET_NAMES,values,count)}


def run_contact_epoch(model, loader, device, stats, optimizer=None, max_steps=None,
                      position_weight=1., wrench_weight=1., slip_weight=1., contact_weight=1.,
                      zero_wrench_weight=1., progress_label="Run"):
    training = optimizer is not None
    model.train(training)
    state, sums, seen = {}, np.zeros(6), 0
    values = {k: [] for k in ("ids", "predictions", "raw_predictions", "targets", "target_valid", "contact_true",
              "contact_probability", "contact_state", "contact_valid", "position_valid", "slip_valid", "transition")}
    mean, std = np.asarray(stats["target_mean"]), np.asarray(stats["target_std"])
    weights = (position_weight,wrench_weight,slip_weight,contact_weight,zero_wrench_weight)
    total = min(len(loader),max_steps) if max_steps else len(loader)
    started = last_log = time.perf_counter()
    print(f"{progress_label}: 0/{total} batches",flush=True)
    iterator = loader if max_steps is None else islice(loader,max_steps)
    for step, incoming in enumerate(iterator,1):
        batch = {k: v.to(device) if isinstance(v,torch.Tensor) else v for k,v in incoming.items()}
        hidden, previous_contact = [], []
        for sequence, order in zip(batch["sequence"],batch["order"].tolist()):
            prev = state.get(sequence)
            continuous = prev is not None and prev[0]+1 == order
            hidden.append(prev[1] if continuous else torch.zeros(1,1,model.config.hidden_size,device=device))
            previous_contact.append(prev[2] if continuous else torch.full((1,),-1,device=device,dtype=torch.long))
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            raw = model.forward_sequence(batch["image"],batch["audio"],torch.cat(hidden,dim=1))
            losses = contact_losses(model,raw,batch,stats["contact_class_weights"])
            loss = sum(w*losses[k] for w,k in zip(weights,LOSS_NAMES))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite contact-aware loss")
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite contact-aware gradient")
                optimizer.step()
        with torch.no_grad():
            # Only PREDICTED previous states cross windows, even during training.
            result = model.gate_outputs(raw,torch.cat(previous_contact))
        for i,(sequence,order) in enumerate(zip(batch["sequence"],batch["order"].tolist())):
            state[sequence] = (order,raw["hidden"][:,i:i+1].detach(),result["last_contact"][i:i+1].detach())
        n = batch["target"].shape[0]*batch["target"].shape[1]
        sums += n*np.array([loss.item()]+[losses[k].item() for k in LOSS_NAMES])
        seen += n
        raw_prediction = torch.cat([raw[k] for k in ("position","wrench","slip")],-1)
        if not torch.isfinite(raw_prediction).all():
            raise FloatingPointError("Nonfinite contact-aware prediction")
        values["ids"].extend(f"{s}/event_{j:02d}" for s in batch["id"] for j in range(batch["target"].shape[1]))
        values["raw_predictions"].append(raw_prediction.detach().cpu().numpy().reshape(-1,11)*std+mean)
        values["predictions"].append(result["physical_prediction"].cpu().numpy().reshape(-1,11))
        values["targets"].append(batch["target"].cpu().numpy().reshape(-1,11)*std+mean)
        values["target_valid"].append(batch["target_valid"].cpu().numpy().reshape(-1,11))
        values["contact_true"].append(batch["contact"].cpu().numpy().reshape(-1))
        values["contact_valid"].append(batch["contact_valid"].cpu().numpy().reshape(-1))
        for k in ("contact_probability","contact_state","position_valid","slip_valid","transition"):
            values[k].append(result[k].detach().cpu().numpy().reshape(-1))
        now = time.perf_counter()
        if step == 1 or step == total or now-last_log > 10:
            print(f"{progress_label}: {step}/{total}, loss={sums[0]/seen:.6f}, events={seen}, seconds={now-started:.1f}",flush=True)
            last_log = now
    if not seen:
        raise ValueError("No contact-aware events evaluated")
    values = {k: v if k == "ids" else np.concatenate(v) for k,v in values.items()}
    cv, truth, predicted = values["contact_valid"].astype(bool),values["contact_true"].astype(bool),values["contact_state"].astype(bool)
    tp = int((cv & truth & predicted).sum()); tn = int((cv & ~truth & ~predicted).sum())
    fp = int((cv & ~truth & predicted).sum()); fn = int((cv & truth & ~predicted).sum())
    metrics = {"loss":float(sums[0]/seen), **{k+"_loss":float(sums[i+1]/seen) for i,k in enumerate(LOSS_NAMES)},
               "samples":seen,"batches":step,"raw_mae":_dimension_metrics(values["raw_predictions"],values["targets"],values["target_valid"]),
               "raw_rmse":_dimension_metrics(values["raw_predictions"],values["targets"],values["target_valid"],True),
               "contact": {"tp":tp,"tn":tn,"fp":fp,"fn":fn,"accuracy":(tp+tn)/max(tp+tn+fp+fn,1),
                           "precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"f1":2*tp/max(2*tp+fp+fn,1),
                           "specificity":tn/(tn+fp) if tn+fp else None,
                           "balanced_accuracy":.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None},
               "predicted_transitions": {str(int(k)):int(n) for k,n in zip(*np.unique(values["transition"],return_counts=True))}}
    # Evaluate force against physical zero on unloaded samples. Position/slip
    # errors use GT label validity, so a classifier cannot hide mistakes by
    # declaring all outputs invalid. Coverage is reported separately.
    physical_truth = values["targets"].copy()
    physical_truth[~truth,3:9] = 0.
    physical_valid = values["target_valid"].copy()
    physical_valid[:,3:9] = cv[:,None]
    metrics["mae"] = _dimension_metrics(values["predictions"],physical_truth,physical_valid)
    metrics["rmse"] = _dimension_metrics(values["predictions"],physical_truth,physical_valid,True)
    for k in ("position","slip"):
        sl = slice(0,3) if k == "position" else slice(9,11)
        label_valid = values["target_valid"][:,sl].all(-1)
        metrics[k+"_output_coverage"] = float(values[k+"_valid"][label_valid].mean()) if label_valid.any() else None
    unloaded = cv & ~truth
    for key,name in (("raw_predictions","no_contact_raw_wrench_mae"),("predictions","no_contact_gated_wrench_mae")):
        metrics[name] = dict(zip(TARGET_NAMES[3:9],np.abs(values[key][unloaded,3:9]).mean(0).tolist())) if unloaded.any() else None
    return metrics,values


def save_contact_predictions(path, values):
    with Path(path).open("w",newline="",encoding="utf-8-sig") as f:
        w = csv.writer(f)
        extras = ("contact_true","contact_probability","contact_state","position_valid","slip_valid","transition")
        w.writerow(["id",*extras]+[prefix+name for prefix in ("true_","pred_","raw_pred_","label_valid_") for name in TARGET_NAMES])
        for i,identifier in enumerate(values["ids"]):
            w.writerow([identifier]+[values[k][i].item() for k in extras]+
                       [v for k in ("targets","predictions","raw_predictions","target_valid") for v in values[k][i].tolist()])


def train_contact_prepared(args):
    if args.use_contact:
        raise ValueError("--use-contact is a legacy ground-truth input option; the new model PREDICTS contact")
    if args.strict_timing:
        raise ValueError("Exposure timestamps unavailable; strict timing cannot be certified")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"last.pt").exists() and not args.resume:
        raise ValueError("Output has last.pt; choose a new run or --resume")
    prepared = args.prepared_root or DEFAULT_CONTACT_CACHE
    args.prepared_root = str(prepared)
    records,manifest = read_contact_prepared(prepared)
    split = split_prepared(records,args.seed,args.val_fraction,args.test_fraction,args.max_groups)
    report = json.loads((Path(prepared)/"report.json").read_text(encoding="utf-8"))
    write_json(output/"data_report.json",report)
    if args.audit_only:
        print({k:{"windows":len(v),"contact":contact_summary(v)} for k,v in split.items()},flush=True)
        return
    config,checkpoint = feature_config(),None
    if args.resume:
        model,checkpoint,config = load_checkpoint(args.resume,device)
        if model.model_name != MODEL_NAME:
            raise ValueError("Resume model mismatch; new architecture needs a new training run")
        saved_channels = list(model.config.disabled_mic_channels)
        if args.disable_mic_channels is not None and sorted(args.disable_mic_channels) != saved_channels:
            raise ValueError("Resume microphone selection differs from checkpoint")
        args.disable_mic_channels = saved_channels
        for name in ("seed","batch_size","lr","weight_decay","val_fraction","test_fraction","max_groups",
                     "max_steps","max_eval_steps","position_weight","wrench_weight","slip_weight","contact_weight","zero_wrench_weight"):
            if getattr(args,name) != checkpoint["args"][name]:
                raise ValueError(f"Resume option mismatch: {name}")
        for option,field in (("contact_off_threshold","contact_off_threshold"),("contact_on_threshold","contact_on_threshold")):
            requested = getattr(args,option)
            if requested is not None and requested != getattr(model.config,field):
                raise ValueError("Resume contact thresholds differ from checkpoint")
        for k in split:
            if ([r["id"] for r in split[k]] != checkpoint["split_ids"][k]
                    or [r["target"] for r in split[k]] != checkpoint["split_labels"][k]
                    or [r["supervision_sha256"] for r in split[k]] != checkpoint["split_supervision"][k]):
                raise ValueError("Contact dataset/split/supervision changed; start a new run")
        stats = checkpoint["stats"]
    else:
        args.disable_mic_channels = sorted(args.disable_mic_channels or [])
        stats = fit_contact_statistics(split["train"])
        model_config = {"disabled_mic_channels":args.disable_mic_channels}
        for field in ("contact_off_threshold","contact_on_threshold"):
            if getattr(args,field) is not None:
                model_config[field] = getattr(args,field)
        model = build_model(MODEL_NAME,model_config).to(device)
        model.set_normalization(stats)
    args.model = MODEL_NAME
    args.contact_off_threshold = model.config.contact_off_threshold
    args.contact_on_threshold = model.config.contact_on_threshold
    loaders = {k:make_contact_loader(v,stats,args.batch_size,args.workers,k=="train",args.seed) for k,v in split.items()}
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=5)
    generator = torch.Generator().manual_seed(args.seed)
    first,best = 1,float("inf")
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        first,best = checkpoint["epoch"]+1,checkpoint["best_loss"]
        loaders["train"].batch_sampler.epoch = checkpoint["epoch"]
        if not (output/"best.pt").exists():
            import shutil
            shutil.copy2(Path(args.resume).parent/"best.pt",output/"best.pt")
    if first > args.epochs:
        raise ValueError("Increase --epochs beyond the completed epoch")
    write_json(output/"normalization.json",stats);write_json(output/"splits.json",split)
    write_json(output/"config.json",{"training":vars(args),"model_name":MODEL_NAME,"model":model.config.to_dict(),
                                     "features":config.to_dict(),"device":str(device)})
    kw = {k+"_weight":getattr(args,k+"_weight") for k in LOSS_NAMES}
    print(f"Model: {MODEL_NAME}; enabled microphones: {[i+1 for i in model.active_mic_indices]}",flush=True)
    for epoch in range(first,args.epochs+1):
        train,_ = run_contact_epoch(model,loaders["train"],device,stats,optimizer,args.max_steps,
                                    progress_label=f"Epoch {epoch} train",**kw)
        val,_ = run_contact_epoch(model,loaders["val"],device,stats,max_steps=args.max_eval_steps,
                                  progress_label=f"Epoch {epoch} val",**kw)
        scheduler.step(val["loss"])
        improved = val["loss"] < best; best = min(best,val["loss"])
        with (output/"history.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps({"epoch":epoch,"train":train,"val":val},allow_nan=False)+"\n")
        for path in ([output/"best.pt"] if improved else [])+[output/"last.pt"]:
            save_checkpoint(path,model,optimizer,scheduler,epoch,best,stats,config,args,split,generator,output/"best.pt")
    model,checkpoint,_ = load_checkpoint(output/"best.pt",device)
    test,values = run_contact_epoch(model,loaders["test"],device,stats,max_steps=args.max_eval_steps,
                                    progress_label="Test",**kw)
    write_json(output/"test_metrics.json",test);save_contact_predictions(output/"test_predictions.csv",values)
    write_json(output/"summary.json",{"model_name":MODEL_NAME,"device":str(device),"best_epoch":checkpoint["epoch"],
               "completed_epoch":args.epochs,"smoke_test":manifest["smoke_test"] or bool(args.max_groups or args.max_steps or args.max_eval_steps),
               "split_windows":{k:len(v) for k,v in split.items()},"contact_counts":{k:contact_summary(v) for k,v in split.items()},"test":test})
    print(json.dumps(test,ensure_ascii=False,allow_nan=False),flush=True)
