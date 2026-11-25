"""Meaningful physical-time and causal-inference checks for the new model."""
import torch

from image_exposure_experiments.model import ExposurePositionModel, from_checkpoint


def make_inputs():
    torch.manual_seed(17)
    return (torch.randn(2, 4, 8), torch.full((2, 4), 0.066),
            torch.tensor([0., .006, .012]).expand(2, 4, 3).clone(), torch.randn(2, 4, 3))


def test_midpoint_propagation_is_in_physical_units():
    model = ExposurePositionModel(feature_dim=8, hidden_size=16, exposure_ms=15.6,
                                  target_mean=(100., 200., 300.), target_std=(2., 3., 4.), dropout=0.).eval()
    with torch.no_grad():
        model.anchor_head[-1].bias.copy_(torch.tensor([1., -1., .5]))
        # tanh(atanh(v / max_speed)) should yield this physical velocity.
        model.velocity_head[-1].bias.copy_(torch.atanh(torch.tensor([10., -20., 5.]) / 200.))
    features, dt, ages, base = make_inputs()
    result = model(features, dt, ages, base)
    expected_anchor = base + torch.tensor([2., -3., 2.])
    expected = expected_anchor[:, :, None] + torch.tensor([10., -20., 5.]) * (ages + .0078)[..., None]
    torch.testing.assert_close(result["position"], expected)
    torch.testing.assert_close(result["anchor"], expected_anchor)


def test_no_future_frame_or_age_changes_earlier_predictions():
    model = ExposurePositionModel(feature_dim=8, hidden_size=16, dropout=0.).eval()
    # Unzero the output heads, otherwise constant zero initialization could hide
    # a future-dependent implementation.
    torch.nn.init.normal_(model.anchor_head[-1].weight, std=.1)
    torch.nn.init.normal_(model.velocity_head[-1].weight, std=.1)
    features, dt, ages, base = make_inputs()
    reference = model(features, dt, ages, base)
    features[:, 2:] += 100.
    dt[:, 2:] *= 2
    ages[:, 2:] += .2
    changed = model(features, dt, ages, base)
    torch.testing.assert_close(reference["position"][:, :2], changed["position"][:, :2])


def test_chunked_stream_matches_full_sequence():
    model = ExposurePositionModel(feature_dim=8, hidden_size=16, dropout=0.).eval()
    torch.nn.init.normal_(model.anchor_head[-1].weight, std=.1)
    torch.nn.init.normal_(model.velocity_head[-1].weight, std=.1)
    features, dt, ages, base = make_inputs()
    full = model(features, dt, ages, base)
    first = model(features[:, :2], dt[:, :2], ages[:, :2], base[:, :2])
    second = model(features[:, 2:], dt[:, 2:], ages[:, 2:], base[:, 2:],
                   hidden=first["hidden"], previous_feature=first["last_feature"])
    torch.testing.assert_close(full["position"], torch.cat((first["position"], second["position"]), dim=1))


def test_zero_exposure_control_and_checkpoint_roundtrip():
    model = ExposurePositionModel(feature_dim=8, hidden_size=16, exposure_ms=0, dropout=0.).eval()
    model.velocity_head[-1].bias.data.fill_(.1)
    features, dt, ages, base = make_inputs()
    result = model(features, dt, ages, base)
    torch.testing.assert_close(result["position"][:, :, 0], base)
    restored = from_checkpoint(model.export_checkpoint()).eval()
    torch.testing.assert_close(restored(features, dt, ages, base)["position"], result["position"])


def test_readout_delay_is_additional_and_configurable():
    model = ExposurePositionModel(feature_dim=8, hidden_size=16, exposure_ms=15.6,
                                  readout_delay_ms=7.8, dropout=0.).eval()
    model.velocity_head[-1].bias.data.copy_(torch.atanh(torch.tensor([10., 0., 0.]) / 200.))
    features, dt, ages, base = make_inputs()
    result = model(features, dt, ages, base)
    expected = base[:, :, None] + torch.tensor([10., 0., 0.]) * (ages + .0156)[..., None]
    torch.testing.assert_close(result["position"], expected)
    assert from_checkpoint(model.export_checkpoint()).readout_delay_ms == 7.8
