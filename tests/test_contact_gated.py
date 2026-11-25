import json

import numpy as np
from PIL import Image
import pytest
import torch

from mvts.architectures import build_model
from mvts.contact_data import (CONTACT_CACHE_VERSION, ContactPreparedDataset, add_contact_supervision,
                               fit_contact_statistics, load_contact_arrays, read_contact_prepared)
from mvts.contact_engine import contact_losses
from mvts.dual_rate_data import prepare_data, split_prepared


def stats():
    return {"target_mean":list(range(1,12)),"target_std":list(range(1,12)),
            "audio_mean":[0.]*6,"audio_std":[1.]*6}


def test_hysteresis_direction_unknown_history_and_physical_zero():
    model=build_model("model_contact_gated")
    model.set_normalization(stats())
    p=torch.tensor([[.1,.5,.8,.5,.2,.5,.9]])
    raw={"contact_probability":p,"position":torch.zeros(1,7,3),
         "wrench":torch.zeros(1,7,6),"slip":torch.zeros(1,7,2)}
    result=model.gate_outputs(raw)
    assert result["contact_state"].tolist()==[[0,0,1,1,0,0,1]]
    assert result["transition"].tolist()==[[-1,0,1,3,2,0,1]]
    assert result["slip_valid"].tolist()==[[False,False,False,True,False,False,False]]
    assert (result["wrench_physical"][0,[0,1,4,5]]==0).all()
    torch.testing.assert_close(result["wrench_physical"][0,2],torch.arange(4.,10.))
    first=model.gate_outputs({k:v[:,:1] if k!="contact_probability" else torch.tensor([[.9]]) for k,v in raw.items()})
    assert first["transition"].item()==-1 and first["contact_state"].item()==1
    assert not first["slip_valid"].item()  # First loaded slide is NOT a fabricated onset.
    with pytest.raises(ValueError,match="predicted states"):
        model.gate_outputs(raw,torch.tensor([.7]))


def test_streaming_matches_sequence_and_release_does_not_reset_gru():
    torch.manual_seed(3)
    model=build_model("model_contact_gated",{"dropout":0,"disabled_mic_channels":[5]}).eval()
    model.set_normalization(stats())
    image,audio=torch.randn(1,3,64,64),torch.randn(1,3,6,64,28)
    with torch.no_grad():
        model.contact_head[-1].weight.zero_();model.contact_head[-1].bias.fill_(3.)
        expected=model.gate_outputs(model.forward_sequence(image,audio[:,:2]))
        first=model.step(audio[:,0],image,frame_id="one")
        second=model.step(audio[:,1],frame_id="one")
        for key in ("position","wrench","slip","contact_probability","wrench_physical","contact_state"):
            torch.testing.assert_close(expected[key][:,0],first[key])
            torch.testing.assert_close(expected[key][:,1],second[key])
        assert first["transition"].item()==-1 and second["transition"].item()==3
        assert second["slip_valid"].item()
        model.contact_head[-1].bias.fill_(-3.)
        expected_release=model.forward(None,audio[:,2],hidden=second["hidden"],visual_tokens=second["visual_tokens"])
        release=model.step(audio[:,2])
        torch.testing.assert_close(release["hidden"],expected_release["hidden"])
        assert release["transition"].item()==2 and not release["wrench_physical"].any()
    assert not any("stream" in key for key in model.state_dict())
    model.reset_state()
    assert model._stream_contact is None and model._stream_hidden is None
    with pytest.raises(ValueError,match="image"):
        model.step(audio[:,0])


def test_four_heads_and_shared_encoders_receive_gradients():
    model=build_model("model_contact_gated",{"dropout":0}).eval()
    raw=model.forward_sequence(torch.randn(1,3,64,64),torch.randn(1,2,6,64,28))
    loss=sum(raw[k].square().mean() for k in ("position","wrench","slip","contact_logits"))
    loss.backward()
    for parameter in (model.contact_head[-1].weight,model.position_head[-1].weight,model.force_head[-1].weight,
                      model.slip_head[-1].weight,model.gru.weight_hh_l0,model.attention.in_proj_weight,
                      model.vision_backbone.stem[0].weight,model.audio_backbone.stem[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
    with pytest.raises(ValueError,match="not inputs"):
        model(torch.randn(1,3,64,64),torch.randn(1,6,64,28),torch.tensor([1]))


def test_background_loss_masks_and_zero_loss_train_ungated_wrench():
    model=build_model("model_contact_gated")
    model.set_normalization(stats())
    raw={key:torch.zeros(2,1,n,requires_grad=True) for key,n in (("position",3),("wrench",6),("slip",2))}
    raw["contact_logits"]=torch.zeros(2,1,requires_grad=True)
    target=torch.full((2,1,11),10.)
    valid=torch.ones(2,1,11,dtype=torch.bool);valid[0]=False
    batch={"target":target,"target_valid":valid,"contact":torch.tensor([[0],[1]]),
           "contact_valid":torch.ones(2,1,dtype=torch.bool)}
    losses=contact_losses(model,raw,batch,[1.,1.])
    sum(losses.values()).backward()
    assert raw["position"].grad[0].abs().sum()==0
    assert raw["slip"].grad[0].abs().sum()==0
    assert raw["wrench"].grad[0].abs().sum()>0  # Zero loss sees ungated physical mean.
    assert raw["contact_logits"].grad[0].item()>0 and raw["contact_logits"].grad[1].item()<0
    all_bg={**batch,"target_valid":torch.zeros_like(valid),"contact":torch.zeros(2,1,dtype=torch.long)}
    assert all(torch.isfinite(v) for v in contact_losses(model,raw,all_bg,[1.,1.]).values())


def test_slide_labels_are_all_contact_without_synthetic_onset():
    arrays={"target":np.zeros((10,11))}
    first,_=add_contact_supervision(arrays,"slide")
    following,_=add_contact_supervision(arrays,"slide",previous_available=True)
    assert first["contact"].tolist()==[1]*10
    assert first["previous_contact"].tolist()==[-1]+[1]*9
    assert not first["transition_valid"][0] and first["transition_valid"][1:].all()
    assert following["previous_contact"].tolist()==[1]*10


@pytest.fixture
def contact_cache(tmp_path):
    session=tmp_path/"raw"/"press"
    session.mkdir(parents=True)
    Image.new("RGB",(32,32)).save(session/"image.png")
    rows=[]
    for point in range(3):
        for kind in ("background","target"):
            key=f"{point}_{kind}"
            ft=np.arange(10,dtype=np.int64)*2_500_000+100_000_000
            at=ft[:,None]+(np.arange(4000)-4000)[None,:]*20_000
            forces=np.full((10,6),.001 if kind=="background" else 1.+point)
            np.savez(session/f"{key}.npz",adc=np.full((10,4000,6),1500,dtype=np.uint16),
                     audio_before=4000,audio_after=0,force_pc_ns=ft,estimated_pc_ns=at,
                     force_wrench=forces,actual_xy_mm=[float(point),2.])
            rows.append({"valid":True,"kind":kind,"mode":"dynamic","point":point,"sample_id":key,
                         "data":f"{key}.npz","image":"image.png","x_mm":point,"y_mm":2.,
                         "actual_xy_mm":[float(point),2.],"actual_depth_mm":-1000. if kind=="background" else .5,
                         "starts_in_contact":int(kind=="background"),  # Deliberately contradict current contact.
                         "force_sequence":[{"wrench":f.tolist(),"pc_est_ns":int(t)} for f,t in zip(forces,ft)]})
    (session/"samples.jsonl").write_text("\n".join(map(json.dumps,rows)),encoding="utf-8")
    prepare_data(tmp_path/"raw",tmp_path/"cache",contact_aware=True)
    return tmp_path/"cache"


def test_contact_cache_background_statistics_masks_integrity_and_splits(contact_cache):
    records,manifest=read_contact_prepared(contact_cache)
    assert manifest["version"]==CONTACT_CACHE_VERSION and len(records)==6
    for r in records:
        a=load_contact_arrays(r)
        contact=int("target" in r["id"])
        assert a["contact"].item()==contact
        assert a["target_valid"].all() if contact else not a["target_valid"].any()
        assert a["previous_contact"].item()==-1
    fitted=fit_contact_statistics(records)
    assert fitted["target_mean"][2]==.5  # Unloaded -1000 mm must never distort position normalization.
    assert fitted["contact_class_counts"]==[3,3]
    assert fitted["target_fit_counts"]==[3]*11
    split=split_prepared(records)
    assert all(sorted(r["contact_state"] for r in part)==[0,1] for part in split.values())
    sample=ContactPreparedDataset(records,fitted)[0]
    assert sample["audio"].shape==(1,6,64,28) and sample["contact"].shape==(1,)
    r=records[0];path=contact_cache/r["file"]
    with np.load(path) as z:
        changed={k:z[k].copy() for k in z.files}
    changed["contact"][:]=1
    np.savez(path,**changed)
    with pytest.raises(ValueError,match="supervision changed"):
        load_contact_arrays(r)


def test_contact_checkpoint_preserves_normalization_and_thresholds(tmp_path):
    from mvts.dual_rate_data import feature_config
    from mvts.engine import load_checkpoint
    model=build_model("model_contact_gated",{"contact_off_threshold":.3,"contact_on_threshold":.7})
    model.set_normalization(stats())
    checkpoint={"format_version":1,"model_name":model.model_name,"model_config":model.config.to_dict(),
                "feature_config":feature_config().to_dict(),"model":model.state_dict(),"stats":stats()}
    path=tmp_path/"contact.pt";torch.save(checkpoint,path)
    restored,_,_=load_checkpoint(path)
    assert restored.config.contact_off_threshold==.3 and restored.normalization_ready.item()
    torch.testing.assert_close(restored.target_mean,model.target_mean)
    checkpoint["stats"]={**stats(),"target_mean":[0.]*11}
    torch.save(checkpoint,path)
    with pytest.raises(ValueError,match="normalization differs"):
        load_checkpoint(path)


def test_contact_cache_standalone_verification(contact_cache):
    from verify_contact_cache import verify
    result = verify(contact_cache)
    assert result["passed"] and result["windows"] == result["events"] == 6
    assert result["contact_counts"] == {"no_contact": 3, "contact": 3}
    assert result["images_verified"] == 1
    assert "baseline_windows_preserved" not in result
