"""Clean two-cache contact/slip baselines and matched single-image anchor trials.

python -m image_exposure_experiments.gated_compare
No press-pretrained weights are used. Frozen REAL acoustic tokens accelerate
the controlled comparison; normal saved models retain their acoustic encoders.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from mvts.architectures import build_model
from mvts.dual_rate_data import OrderedBatchSampler
from mvts.engine import load_checkpoint, seed_everything
from .prepare import sha256
from .run import metrics, paired_bootstrap
from .multimodal_data import load_data, batch, MULTIMODAL_BASELINE
from .audio_cache import freeze_audio, forward_cached, prepare_audio_tokens
from .multimodal_anchor import train_variant, predict_anchors
from .train_anchor import causal_extrapolate


MODELS = ('model_contact_gated', 'model_slip_gated')
VARIANTS = {'image_exposure_nooffset': {'delay_ms': 0., 'averaged': False},
            'image_exposure_midpoint': {'delay_ms': 7.8, 'averaged': False}}


def clean_model(name, original):
    """Transfer only the clean two-cache model; initialize new gates explicitly."""
    seed_everything(42)
    model = build_model(name, {'disabled_mic_channels': [5]})
    own, source = model.state_dict(), original['model']
    compatible = {k: v for k,v in source.items() if k in own and own[k].shape == v.shape}
    if name == 'model_slip_gated':
        for prefix in ('visual_attention.', 'audio_attention.'):
            for suffix in ('in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias'):
                compatible[prefix+suffix] = source['attention.'+suffix]
    missing = model.load_state_dict(compatible, strict=False).missing_keys
    model.set_normalization(original['stats'])
    freeze_audio(model)
    return model.cuda(), {'transferred_state_entries': len(compatible), 'new_state_entries': missing,
                          'slip_attention_initialization': 'copy clean joint attention to both branches'}


def sequence_forward(model, b, hidden=None, history=None, exposure_delay_ms=0.):
    kwargs = {}
    if getattr(model, 'requires_timing', False):
        kwargs = {'delta_t': b['delta_t'], 'image_age': b['image_age']+exposure_delay_ms*.001,
                  'has_history': history}
    return forward_cached(model, b['image'], b['audio_tokens'], hidden=hidden, **kwargs)


@torch.no_grad()
def evaluate_baseline(model, data, stats):
    model.eval()
    raw = np.zeros_like(data['target'], dtype=np.float64)
    probability = np.full(raw.shape[:2], np.nan)
    gate_raw = raw.copy()
    coverage = np.ones(raw.shape[:2], bool)
    visual_weight = np.full(raw.shape[:2], np.nan)
    state = {}
    mean, std = np.asarray(stats['target_mean']), np.asarray(stats['target_std'])
    for idx in OrderedBatchSampler(data['records'], 8, False, 42):
        b = batch(data, idx, 'cuda')
        hidden, histories, previous = [], [], []
        for i in idx:
            r = data['records'][i]
            old = state.get(r['sequence'])
            present = old is not None and old[0]+1 == r['order']
            histories.append(present)
            hidden.append(old[1] if present else torch.zeros(1,1,model.config.hidden_size,device='cuda'))
            previous.append(old[2] if present else torch.full((1,),-1,device='cuda',dtype=torch.long))
        result = sequence_forward(model,b,torch.cat(hidden,dim=1),torch.tensor(histories,device='cuda'))
        y = torch.cat([result[k] for k in ('position','wrench','slip')],-1).cpu().numpy()*std+mean
        raw[idx] = y
        contact = model.model_name == 'model_contact_gated'
        if contact:
            g = model.gate_outputs(result,torch.cat(previous))
            probability[idx] = result['contact_probability'].cpu().numpy()
            coverage[idx] = g['position_valid'].cpu().numpy()
            gate_raw[idx] = g['physical_prediction'].cpu().numpy()
        else:
            gate_raw[idx] = y
            visual_weight[idx] = result['visual_weight'].cpu().numpy()
        for j,i in enumerate(idx):
            r=data['records'][i]
            c = g['last_contact'][j:j+1] if contact else torch.full((1,),-1,device='cuda',dtype=torch.long)
            state[r['sequence']] = (r['order'],result['hidden'][:,j:j+1].detach(),c.detach())
    return {'raw':raw, 'gated':gate_raw, 'contact_probability':probability,
            'position_valid':coverage, 'visual_weight':visual_weight}


def baseline_train(name, original, train, val, args, out):
    model, transfer = clean_model(name, original)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.baseline_lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.baseline_lr*.1)
    sampler=OrderedBatchSampler(train['records'],8,True,42)
    best, best_state, history = float('inf'), None, []
    started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();freeze_audio(model)
        state={};total=[]
        for idx in sampler:
            b=batch(train,idx,'cuda')
            hidden,history_flags=[],[]
            for i in idx:
                r=train['records'][i];old=state.get(r['sequence'])
                present=old is not None and old[0]+1==r['order']
                history_flags.append(present)
                hidden.append(old[1] if present else torch.zeros(1,1,model.config.hidden_size,device='cuda'))
            result=sequence_forward(model,b,torch.cat(hidden,dim=1),torch.tensor(history_flags,device='cuda'))
            target=b['target']
            loss=sum(F.smooth_l1_loss(result[k],target[...,sl]) for k,sl in
                     [('position',slice(0,3)),('wrench',slice(3,9)),('slip',slice(9,11))])
            if name=='model_contact_gated':
                loss=loss+F.binary_cross_entropy_with_logits(result['contact_logits'],b['contact'].float())
            else:
                loss=loss+F.smooth_l1_loss(result['slip_prior'],target[...,9:])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite baseline loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            total.append(float(loss.detach()))
            for j,i in enumerate(idx):
                r=train['records'][i];state[r['sequence']]=(r['order'],result['hidden'][:,j:j+1].detach())
        scheduler.step()
        values=evaluate_baseline(model,val,original['stats'])
        score=metrics(values['raw'][...,:3],val['target'])['xy_mean_mm']
        history.append({'epoch':epoch,'loss':float(np.mean(total)),'val_xy_mm':score,'elapsed_seconds':time.perf_counter()-started})
        if score<best:
            best,best_epoch=score,epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch==1 or epoch%5==0:
            print(name,'BASELINE',epoch,'valXY',round(score,5),'best',round(best,5),'seconds',round(time.perf_counter()-started,1),flush=True)
    model.load_state_dict(best_state)
    path=out/'image_exposure_baseline.pt'
    payload={'format_version':1,'model_name':name,'model_config':model.config.to_dict(),
             'feature_config':original['feature_config'],'stats':original['stats'],'model':best_state,
             'epoch':best_epoch,'split_ids':original['split_ids'],'args':{**original['args'],'output':str(out)},
             'experiment_type':'image_exposure_clean_two_cache_baseline','clean_initial_checkpoint':str(MULTIMODAL_BASELINE),
             'frozen_audio':True,'transfer':transfer,'validation_xy_mm':best}
    torch.save(payload,path)
    (out/'image_exposure_baseline_history.json').write_text(json.dumps(history,indent=2))
    values=evaluate_baseline(model,val,original['stats'])
    np.savez(out/'image_exposure_baseline_val.npz',**values)
    del model,opt;torch.cuda.empty_cache()
    return path,{'validation_xy_mm':best,'best_epoch':best_epoch,'seconds':time.perf_counter()-started,'transfer':transfer}


def gate_anchor(probabilities, raw_position, records, config):
    """Predicted contact hysteresis; never exclude rejected labels from errors."""
    states=np.zeros(len(records),bool)
    for seg in __import__('image_exposure_experiments.prepare',fromlist=['contiguous_segments']).contiguous_segments(records):
        previous=None
        for i in seg:
            p=probabilities[i]
            c=p>=config['contact_initial_threshold'] if previous is None else previous
            if p>=config['contact_on_threshold']:c=True
            if p<=config['contact_off_threshold']:c=False
            states[i],previous=c,c
    return np.where(states[:,None,None],raw_position,0.),np.repeat(states[:,None],raw_position.shape[1],axis=1)


def summarize(raw, target, records, gated=None, valid=None):
    result={'raw_position':metrics(raw[...,:3],target,records),
            'image_arrival':metrics(raw[:,:1,:3],target[:,:1],records),
            'raw_force_mae':np.abs(raw[...,3:9]-target[...,3:9]).mean(axis=(0,1)).tolist() if raw.shape[-1]>=9 else None,
            'raw_slip_mae':np.abs(raw[...,9:]-target[...,9:]).mean(axis=(0,1)).tolist() if raw.shape[-1]==11 else None}
    if raw.shape[-1]==11:
        result['first_event_force_mae']=np.abs(raw[:,0,3:9]-target[:,0,3:9]).mean(0).tolist()
        result['first_event_slip_mae']=np.abs(raw[:,0,9:]-target[:,0,9:]).mean(0).tolist()
    if gated is not None:
        result['gated_position']=metrics(gated[...,:3],target,records)
        result['position_output_coverage']=float(valid.mean())
        result['contact_false_negative_events']=int((~valid).sum())
        result['contact_specificity']=None
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/image_exposure_gated_compare')
    parser.add_argument('--source',default='runs/image_exposure_v1')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--baseline-lr',type=float,default=1e-4)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--weight-decay',type=float,default=1e-4)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=Path(args.output)
    if not out.name.startswith('image_exposure_') or out.exists():raise ValueError('Choose a new image_exposure_xxx output')
    out.mkdir(parents=True);torch.set_num_threads(4)
    source_model,original,_=load_checkpoint(MULTIMODAL_BASELINE,'cuda');source_model.eval()
    sources=[MULTIMODAL_BASELINE,*[Path(f'.cache/data_final_v6_{s}_contact/manifest.json') for s in ('slide','polyline')],
             Path('mvts/model_contact_gated/model_contact_gated.py'),Path('mvts/model_slip_gated/model_slip_gated.py')]
    hashes={str(p):sha256(p) for p in sources}
    protocol={'source_checkpoint':str(MULTIMODAL_BASELINE),'source_sha256':hashes,'training':vars(args),
        'inputs':'real images and real causal audio; microphone5 disabled; no press data or press-pretrained weights',
        'baseline':'new clean transferred contact/slip models;30epochs all10events with continuousGRU; acoustic encoder+projection fixed',
        'anchor':'30epochs positiononly eachimage GRUreset, firstaudioeventonly, predictedmotion extrapolation; acoustic branch fixed',
        'variants':VARIANTS,'selection':'validation current-time rawXY minimum; include baseline fallback; all choices fixed beforetest',
        'scope':'single seed exploratory on reused49trajectory holdout; allpositivecontact cannot assess background discrimination',
        'baseline_not_deployed':'new models initialized ONLY from clean2cache triplehead; conclusions do not directly measure existing press-pretrained deployed checkpoints'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Loading clean train/val inputs and real acoustic feature cache',flush=True)
    train,val=[load_data(args.source,s,original['stats']) for s in ('train','val')]
    for d in (train,val):
        prepare_audio_tokens(source_model,d,device='cuda')
        d.pop('audio',None)
    del source_model;torch.cuda.empty_cache()
    selections={}
    for name in MODELS:
        modelout=out/f'image_exposure_{name.removeprefix("model_")}'
        modelout.mkdir()
        basepath,base=baseline_train(name,original,train,val,args,modelout)
        variants={}
        for v,settings in VARIANTS.items():
            trained=train_variant(name,basepath,train,val,args,settings,modelout/v)
            variants[v]=trained['metadata']
        scores={'baseline':base['validation_xy_mm'],**{v:r['validation_xy_mm'] for v,r in variants.items()}}
        selected=min(scores,key=scores.get)
        selections[name]={'baseline':base,'variants':variants,'selected':selected,'selection_scores':scores,'output':str(modelout)}
        (modelout/'selection.json').write_text(json.dumps(selections[name],indent=2))
    (out/'selection.json').write_text(json.dumps(selections,indent=2))
    print('All validation choices locked; loading TEST now',flush=True)
    source_model,_,_=load_checkpoint(MULTIMODAL_BASELINE,'cuda');source_model.eval()
    test=load_data(args.source,'test',original['stats']);prepare_audio_tokens(source_model,test,device='cuda');test.pop('audio',None)
    del source_model;torch.cuda.empty_cache()
    results={}
    for name,sel in selections.items():
        modelout=Path(sel['output']);model,checkpoint,_=load_checkpoint(modelout/'image_exposure_baseline.pt','cuda')
        base=evaluate_baseline(model,test,original['stats']);np.savez(modelout/'image_exposure_baseline_test.npz',**base)
        modelresult={'baseline':summarize(base['raw'],test['target'],test['records'],base['gated'] if name=='model_contact_gated' else None,base['position_valid']),
                     'variants':{},'selected':sel['selected']}
        for v,settings in VARIANTS.items():
            variantout=modelout/v
            candidate,c,_=load_checkpoint(sel['variants'][v]['checkpoint_path'],'cuda')
            prediction=predict_anchors(candidate,test,original['stats'],'cuda',delay_ms=settings['delay_ms'])
            motion=c['anchor_config']
            raw=causal_extrapolate(prediction['anchors'],test['records'],test['image_ns'],test['host_ages'],settings['delay_ms'],motion['smoothing'])
            gated,valid=(None,None)
            if name=='model_contact_gated':gated,valid=gate_anchor(prediction['contact_probability'],raw,test['records'],candidate.config.to_dict())
            r=summarize(raw,test['target'],test['records'],gated,valid)
            r['first_event_force_mae']=np.abs(prediction['raw_first'][:,3:9]-test['target'][:,0,3:9]).mean(0).tolist()
            r['first_event_slip_mae']=np.abs(prediction['raw_first'][:,9:]-test['target'][:,0,9:]).mean(0).tolist()
            r['paired_bootstrap']=paired_bootstrap(base['raw'],raw,test['target'],test['records'])
            r['image_arrival_bootstrap']=paired_bootstrap(base['raw'][:,:1],raw[:,:1],test['target'][:,:1],test['records'])
            modelresult['variants'][v]=r
            np.savez(variantout/'image_exposure_test_predictions.npz',position=raw,gated_position=gated if gated is not None else raw,
                     target=test['target'],baseline=base['raw'],event_pc_ns=test['event_ns'],**prediction)
            del candidate
        results[name]=modelresult
        (modelout/'summary.json').write_text(json.dumps(modelresult,indent=2));del model;torch.cuda.empty_cache()
    summary={'protocol':protocol,'selection':selections,'results':results,
             'original_integrity':{p:sha256(p)==h for p,h in hashes.items()}}
    assert all(summary['original_integrity'].values())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({n:{'baseline':r['baseline']['raw_position']['xy_mean_mm'],'selected':r['selected'],
        'variants':{v:x['raw_position']['xy_mean_mm'] for v,x in r['variants'].items()}} for n,r in results.items()},indent=2),flush=True)


if __name__=='__main__':main()
