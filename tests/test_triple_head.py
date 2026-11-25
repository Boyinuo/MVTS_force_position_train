import numpy as np
import json
import pytest
import torch

from mvts.architectures import build_model
from mvts.dual_rate_data import (align_slide_interval, feature_config, CacheWriter,
                                OrderedBatchSampler, read_prepared, prepare_data)
from mvts.features import FeatureConfig, LogMel


def test_causal_alignment_displacement_and_history():
    # Linear measured trajectory at 400Hz, audio at 50kHz.
    ft = np.arange(0, 401_000_000, 2_500_000, dtype=np.int64)
    position = np.c_[ft/1e9*15, ft/1e9*-3, np.ones(len(ft))]
    at = np.arange(0, 401_000_000, 20_000, dtype=np.int64)
    adc = np.repeat(np.arange(len(at), dtype=np.float32)[:,None],6,axis=1)
    data = align_slide_interval(100_000_000,166_666_667,ft,np.zeros((len(ft),6)),position,at,adc)
    assert data["adc"].shape == (10,4000,6)
    assert data["event_pc_ns"][0] == 100_000_000
    assert data["event_pc_ns"][-1] < 166_666_667
    assert np.all(data["audio_last_pc_ns"] < data["event_pc_ns"])
    assert np.all(data["force_pc_ns"] <= data["event_pc_ns"])
    np.testing.assert_allclose(data["target"][:,9:], np.tile([.1,-.02],(10,1)),atol=1e-7)
    assert np.all(data["adc"][0,-1] == 4999)
    with pytest.raises(ValueError,match="history"):
        align_slide_interval(50_000_000,116_666_667,ft,np.zeros((len(ft),6)),position,at,adc)
    at[2000:] += 1_000_000
    with pytest.raises(ValueError,match="gap"):
        align_slide_interval(100_000_000,166_666_667,ft,np.zeros((len(ft),6)),position,at,adc)


def test_right_aligned_stft_includes_endpoint():
    c = feature_config()
    extractor = LogMel(c)
    assert c.audio_frames == 28
    a = torch.zeros(4000,6)
    a[-2] = 30000
    a[-3] = -30000
    # The final 32 samples must influence the spectrum.
    assert not torch.allclose(extractor(a),extractor(torch.zeros_like(a)))
    legacy = LogMel(FeatureConfig(samples=4000))
    torch.testing.assert_close(legacy(a), legacy(torch.zeros_like(a)))
    assert extractor(a).shape == (6,64,28)


def test_model_streaming_equivalence_cache_and_reset():
    torch.manual_seed(5)
    model = build_model("model_triple_head",{"dropout":0}).eval()
    image, audio = torch.randn(1,3,64,64),torch.randn(1,2,6,64,28)
    calls=[]
    hook = model.vision_backbone.register_forward_hook(lambda *x:calls.append(1))
    with torch.no_grad():
        expected=model.forward_sequence(image,audio)
        a=model.step(audio[:,0],image,frame_id="a")
        b=model.step(audio[:,1],frame_id="a")
    assert len(calls)==2  # Once for training-style sequence, once for stream.
    assert a["visual_tokens"].shape==(1,16,128)
    assert b["hidden"].shape==(1,1,256)
    for key,size in (("position",3),("wrench",6),("slip",2)):
        assert expected[key].shape==(1,2,size)
        torch.testing.assert_close(expected[key][:,0],a[key])
        torch.testing.assert_close(expected[key][:,1],b[key])
    model.reset_state()
    with pytest.raises(ValueError,match="image"):
        model.step(audio[:,0])
    assert not any("stream" in k for k in model.state_dict())
    hook.remove()


def test_three_heads_backward_through_shared_mic_encoder():
    model=build_model("model_triple_head",{"dropout":0}).eval()
    result=model.forward_sequence(torch.randn(1,3,64,64),torch.randn(1,2,6,64,28))
    sum(result[k].square().mean() for k in ("position","wrench","slip")).backward()
    for param in (model.vision_backbone.stem[0].weight,model.audio_backbone.stem[0].weight,
                  model.attention.in_proj_weight,model.gru.weight_hh_l0,model.slip_head[-1].weight):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum()>0
    assert model.audio_backbone.stem[0].in_channels==1


def test_sampler_never_reorders_or_duplicates_streams():
    records=[{"sequence":s,"order":i} for s,n in (("a",3),("b",1),("c",4),("d",2)) for i in range(n)]
    sampler=OrderedBatchSampler(records,2,True,seed=9)
    length=len(sampler)
    batches=list(sampler)
    assert len(batches)==length
    assert sorted(i for b in batches for i in b)==list(range(10))
    observed={}
    for batch in batches:
        assert len({records[i]["sequence"] for i in batch})==len(batch)
        for i in batch:
            r=records[i]
            assert r["order"]==observed.get(r["sequence"],-1)+1
            observed[r["sequence"]]=r["order"]


def test_press_preparation_uses_only_current_force(tmp_path):
    from PIL import Image
    session=tmp_path/"raw"/"press"
    session.mkdir(parents=True)
    Image.new("RGB",(32,32)).save(session/"frame.png")
    ft=np.arange(10,dtype=np.int64)*2_500_000+100_000_000
    at=ft[:,None]+(np.arange(4000)-4000)[None,:]*20_000
    forces=np.arange(60,dtype=np.float64).reshape(10,6)
    adc=np.random.default_rng(2).integers(1000,2000,(10,4000,6),dtype=np.uint16)
    np.savez(session/"event.npz",adc=adc,audio_before=4000,audio_after=0,
             force_pc_ns=ft,estimated_pc_ns=at,force_wrench=forces,actual_xy_mm=[2.,3.])
    r={"valid":True,"kind":"target","mode":"dynamic","point":1,"sample_id":"one",
       "data":"event.npz","image":"frame.png","x_mm":2,"y_mm":3,"actual_xy_mm":[2.,3.],
       "actual_depth_mm":.5,"force_sequence":[{"wrench":f.tolist(),"pc_est_ns":int(t)} for f,t in zip(forces,ft)]}
    (session/"samples.jsonl").write_text(json.dumps(r),encoding="utf-8")
    records,report=prepare_data(tmp_path/"raw",tmp_path/"cache")
    assert report["events"]==1 and records[0]["events"]==1 and records[0]["force_index"]==9
    with np.load(tmp_path/"cache"/records[0]["file"]) as z:
        np.testing.assert_array_equal(z["target"][0,3:9],forces[-1])
        np.testing.assert_array_equal(z["target"][0,9:],[0.,0.])
        assert z["event_pc_ns"].tolist()==[int(ft[-1])]
        assert z["audio"].shape==(1,6,64,28)
    mixed=[{"sequence":"press","order":0,"events":1},
           {"sequence":"slide","order":0,"events":10},
           {"sequence":"slide","order":1,"events":10}]
    for batch in OrderedBatchSampler(mixed,2,True):
        assert len({mixed[i]["events"] for i in batch})==1


def test_disabled_fifth_microphone_cannot_affect_features_or_gradients():
    model = build_model("model_triple_head", {"disabled_mic_channels": [5], "dropout": 0}).eval()
    audio = torch.randn(1,6,64,28,requires_grad=True)
    observed = []
    hook = model.audio_backbone.register_forward_pre_hook(lambda m,args:observed.append(tuple(args[0].shape)))
    encoded = model.encode_audio(audio)
    changed = audio.detach().clone()
    changed[:,4] = float("nan")  # A faulty disabled mic must not reach the encoder.
    with torch.no_grad():
        same = model.encode_audio(changed)
    torch.testing.assert_close(encoded.detach(),same)
    encoded.square().mean().backward()
    assert observed == [(5,1,64,28)]*2
    assert (audio.grad[:,4] == 0).all()
    assert all(audio.grad[:,i].abs().sum() > 0 for i in (0,1,2,3,5))
    hook.remove()


@pytest.mark.parametrize("channels", [[0],[7],[5,5],[1,2,3,4,5,6],[True]])
def test_invalid_microphone_selections(channels):
    from mvts.model_triple_head import ModelConfig
    with pytest.raises(ValueError,match="disabled_mic_channels"):
        ModelConfig(disabled_mic_channels=channels)


def test_disabled_microphones_survive_checkpoint_roundtrip(tmp_path):
    from mvts.engine import load_checkpoint
    model = build_model("model_triple_head", {"disabled_mic_channels": [5]})
    path = tmp_path/"masked.pt"
    torch.save({"format_version":1,"model_name":"model_triple_head", "model_config":model.config.to_dict(),
                "feature_config":feature_config().to_dict(),"model":model.state_dict()},path)
    restored,_,_ = load_checkpoint(path)
    assert restored.config.disabled_mic_channels == (5,)
    assert restored.active_mic_indices == (0,1,2,3,5)
    # Checkpoints predating microphone selection still enable all six channels.
    old_config = model.config.to_dict()
    old_config.pop("disabled_mic_channels")
    old_config.pop("image_only")
    legacy = build_model("model_triple_head",old_config)
    legacy.load_state_dict(model.state_dict())
    assert legacy.active_mic_indices == tuple(range(6))
    assert not legacy.config.image_only


def test_image_only_ignores_audio_and_backpropagates_visual_features():
    model = build_model("model_triple_head", {"image_only": True, "dropout": 0}).eval()
    image = torch.randn(1,3,64,64)
    audio = torch.full((1,2,6,64,28), float("nan"), requires_grad=True)
    memories = []
    hook = model.attention.register_forward_pre_hook(lambda m,args: memories.append(args[1].shape[1]))
    result = model.forward_sequence(image, audio)
    with torch.no_grad():
        camera_only = model.forward_sequence(image, sequence_length=2)
        single = model(image)
    for key, size in (("position",3), ("wrench",6), ("slip",2)):
        assert result[key].shape == (1,2,size)
        torch.testing.assert_close(result[key], camera_only[key])
        torch.testing.assert_close(result[key][:,0], single[key])
    sum(result[k].square().mean() for k in ("position","wrench","slip")).backward()
    assert audio.grad is None
    assert model.audio_backbone is None and model.audio_projection is None
    assert model.active_mic_indices == () and set(memories) == {16}
    for parameter in (model.vision_backbone.stem[0].weight, model.query.weight,
                      model.attention.in_proj_weight, model.gru.weight_hh_l0,
                      model.position_head[-1].weight, model.force_head[-1].weight, model.slip_head[-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    hook.remove()


def test_image_only_streaming_caches_visual_features_and_matches_sequence():
    model = build_model("model_triple_head", {"image_only": True, "dropout": 0}).eval()
    image = torch.randn(1,3,64,64)
    with torch.no_grad():
        expected = model.forward_sequence(image, sequence_length=2)
    calls = []
    hook = model.vision_backbone.register_forward_hook(lambda *args: calls.append(1))
    a = model.step(image=image, frame_id="a")
    b = model.step(frame_id="a")
    assert len(calls) == 1
    reset = model.step(image=image, reset=True)
    for key in ("position", "wrench", "slip"):
        torch.testing.assert_close(a[key], expected[key][:,0])
        torch.testing.assert_close(b[key], expected[key][:,1])
        torch.testing.assert_close(reset[key], a[key])
    with pytest.raises(ValueError, match="image"):
        model.step(frame_id="missing")
    model.reset_state()
    with pytest.raises(ValueError, match="image"):
        model.step()
    hook.remove()


def test_image_only_invalid_configuration_and_missing_audio():
    with pytest.raises(ValueError, match="image_only"):
        build_model("model_triple_head", {"image_only": True, "disabled_mic_channels": [5]})
    model = build_model("model_triple_head", {"image_only": True}).eval()
    image = torch.randn(1,3,64,64)
    for length in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="sequence_length"):
            model.forward_sequence(image, sequence_length=length)
    with pytest.raises(ValueError, match="sequence_length"):
        model.forward_sequence(image, torch.randn(1,2,6,64,28), sequence_length=3)
    model = build_model("model_triple_head").eval()
    with pytest.raises(ValueError, match="log-mel"):
        model(image)
    with pytest.raises(ValueError, match="sequence audio"):
        model.forward_sequence(image)


@pytest.mark.parametrize("image_only", [False, True])
def test_input_mode_train_resume_evaluate_and_predict(tmp_path, image_only):
    import subprocess
    import sys
    from pathlib import Path
    from PIL import Image
    from mvts.dual_rate_data import CACHE_VERSION
    from mvts.engine import load_checkpoint

    cache = tmp_path/"cache"
    cache.mkdir()
    image = cache/"image.png"
    Image.new("RGB", (64,64), (100,120,140)).save(image)
    records = []
    for i in range(3):
        target = np.full((2,11), float(i), dtype=np.float32)
        filename = f"{i}.npz"
        np.savez(cache/filename, audio=np.zeros((2,6,64,28), dtype=np.float32), target=target)
        records.append({"id": str(i), "file": filename, "image": "image.png", "kind": "slide",
                        "group": str(i), "sequence": str(i), "order": 0, "events": 2,
                        "target": target.tolist()})
    (cache/"manifest.json").write_text(json.dumps({"version": CACHE_VERSION,
        "features": feature_config().to_dict(), "smoke_test": True, "records": records}), encoding="utf-8")
    (cache/"report.json").write_text("{}", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]

    def run(*args, error=None):
        result = subprocess.run([sys.executable, *map(str,args)], cwd=root,
                                text=True, capture_output=True, timeout=120)
        if error:
            assert result.returncode != 0 and error in result.stderr, result.stdout+result.stderr
        else:
            assert result.returncode == 0, result.stdout+result.stderr

    output = tmp_path/"run"
    base = ["train.py", "--no-demo-tests", "--prepared-root", cache, "--output", output, "--device", "cpu",
            "--threads", 2, "--batch-size", 1, "--max-steps", 1, "--max-eval-steps", 1]
    run(*base, "--image-only" if image_only else "--no-image-only", "--epochs", 1)
    run(*base, "--resume", output/"last.pt", "--epochs", 2)
    config = json.loads((output/"config.json").read_text(encoding="utf-8"))
    assert config["model"]["image_only"] == config["training"]["image_only"] == image_only
    restored, checkpoint, _ = load_checkpoint(output/"last.pt")
    assert restored.config.image_only == image_only
    assert (restored.audio_backbone is None) == image_only
    assert checkpoint["epoch"] == 2
    run(*base, "--resume", output/"last.pt", "--epochs", 3,
        "--no-image-only" if image_only else "--image-only",
        error="Resume image_only differs")
    run(*base, "--image-only", "--disable-mic-channels", 5, error="cannot be combined")
    run(*base, "--image-only", "--model", "model_contact_gated", error="supported only by model_triple_head")
    run("evaluate.py", "--checkpoint", output/"best.pt", "--prepared-root", cache,
        "--device", "cpu", "--output", tmp_path/"evaluation")
    source = [] if image_only else ["--prepared-window", cache/"0.npz"]
    run("predict.py", "--checkpoint", output/"best.pt", "--image", image, *source, "--device", "cpu",
        "--state-out", tmp_path/"state.pt", "--output", tmp_path/"prediction.json")
    prediction = json.loads((tmp_path/"prediction.json").read_text(encoding="utf-8"))
    rows = [prediction] if image_only else prediction["predictions"]
    assert len(rows) == (1 if image_only else 2)
    for row in rows:
        assert len(row["prediction"]) == 11
        assert np.isfinite(list(row["prediction"].values())).all()
    assert (tmp_path/"state.pt").is_file()


@pytest.mark.parametrize("layout", ["attempts", "trajectories"])
def test_v2_slide_layout_and_source_filter(tmp_path, layout):
    from PIL import Image
    root = tmp_path/"raw"
    session = root/"selected_fast"
    trajectory = session/layout/"t0001"
    trajectory.mkdir(parents=True)
    (session/"dataset_summary.json").write_text(json.dumps({"schema":"planar_slide_intervals_v2"}),encoding="utf-8")
    (session/"samples.jsonl").write_text(json.dumps({"image_start":"unused_top_level_index"}),encoding="utf-8")
    (trajectory/"result.json").write_text(json.dumps({"status":"complete","trial":1}),encoding="utf-8")
    boundaries = np.rint(np.arange(5)*1e9/15).astype(np.int64)
    audio_times = np.arange(0,boundaries[-1],20_000,dtype=np.int64)
    force_times = np.arange(0,boundaries[-1],2_500_000,dtype=np.int64)
    groups = []
    for i in range(5):
        Image.new("RGB",(32,32)).save(trajectory/f"frame_{i}.png")
    for i,(start,end) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        ai = np.flatnonzero((audio_times>=start)&(audio_times<end))
        ft = force_times[(force_times>=start)&(force_times<end)]
        path = f"interval_{i}.npz"
        np.savez(trajectory/path,adc=np.full((len(ai),6),1400,dtype=np.uint16),
                 audio_pc_est_ns=audio_times[ai],sample_index=ai,force_pc_est_ns=ft,
                 force_wrench=np.ones((len(ft),6)),
                 force_local_xyz_mm=np.c_[ft/1e9*20,ft/1e9*-5,np.zeros(len(ft))],
                 force_depth_mm=np.ones(len(ft)))
        groups.append({"schema":"planar_slide_intervals_v2","frame_intervals":1,"valid":True,
                       "clipped_head":i==0,"clipped_tail":False,"data_file":path,
                       "first_frame":{"pc_ns":int(start),"file":f"frame_{i}.png"},
                       "last_frame":{"pc_ns":int(end),"file":f"frame_{i+1}.png"}})
    (trajectory/"groups.jsonl").write_text("\n".join(map(json.dumps,groups)),encoding="utf-8")
    old = root/"excluded_old"/"attempts"/"t0001"
    old.mkdir(parents=True)
    # This source is deliberately incomplete; selecting fast must avoid opening it.
    (old/"groups.jsonl").write_text(json.dumps({"schema":"planar_slide_intervals_v2"}),encoding="utf-8")
    records,report = prepare_data(root,tmp_path/"cache",slide_sources=["selected_fast"])
    assert len(records)==2 and report["events"]==20 and not report["rejected"]
    assert all(r["kind"]=="slide" and r["id"].startswith("selected_fast/") for r in records)
    assert report["selection"]["excluded_slide_sources"]==["excluded_old"]
    assert set(report["sources"])=={"selected_fast"}
    assert not any("source" in x for x in report["excluded"])
    with np.load(tmp_path/"cache"/records[0]["file"]) as z:
        # The first event at 133.33ms needs audio starting at 53.33ms:
        # it reaches past both current and previous image-interval boundaries.
        first_audio_index = int(z["audio_stop"][0])-4000
        assert audio_times[first_audio_index] < boundaries[1] < z["event_pc_ns"][0]
    with pytest.raises(ValueError,match="slide source selection"):
        prepare_data(root,tmp_path/"unknown",slide_sources=["missing"])
