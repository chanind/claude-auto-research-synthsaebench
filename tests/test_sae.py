"""Tests for LISTA-Matryoshka SAE with Decreasing K.

Tests cover: architecture naming, shapes, LISTA refinement, matryoshka losses,
TERM loss, frequency sorting, decreasing K schedule, save/load checkpoint,
and training/inference equivalence.
"""

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from sae_lens import SAE
from sae_lens.saes.sae import TrainStepInput

sys.path.insert(0, str(Path(__file__).parent.parent))
from autoresearch.sae import (
    FiringFrequencyTracker,
    ListaMatryoshkaSAE,
    ListaMatryoshkaSAEConfig,
    ListaMatryoshkaTrainingSAE,
    ListaMatryoshkaTrainingSAEConfig,
    _lerp,
    term_mean,
)


def _default_kwargs() -> dict[str, Any]:
    return {
        "d_in": 8,
        "d_sae": 16,
        "k": 4,
        "matryoshka_widths": [4, 8, 16],
        "n_refinement_steps": 1,
        "eta": 0.3,
        "dtype": "float32",
        "device": "cpu",
    }


def _build_sae(**overrides: Any) -> ListaMatryoshkaTrainingSAE:
    kwargs = _default_kwargs()
    kwargs.update(overrides)
    cfg = ListaMatryoshkaTrainingSAEConfig(**kwargs)
    return ListaMatryoshkaTrainingSAE(cfg)


@torch.no_grad()
def _random_params(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.data = torch.randn_like(param)


# ============ Architecture names ============


def test_training_architecture_name():
    cfg = ListaMatryoshkaTrainingSAEConfig(**_default_kwargs())
    assert cfg.architecture() == "lista_matryoshka_decreasingk_batchtopk"


def test_inference_architecture_name():
    cfg = ListaMatryoshkaSAEConfig(
        d_in=8, d_sae=16, n_refinement_steps=1, dtype="float32", device="cpu"
    )
    assert cfg.architecture() == "lista_matryoshka_decreasingk"


def test_inference_config_class():
    cfg = ListaMatryoshkaTrainingSAEConfig(**_default_kwargs())
    assert cfg.get_inference_config_class() == ListaMatryoshkaSAEConfig


# ============ Encode/decode shapes ============


def test_encode_decode_shapes():
    sae = _build_sae()
    _random_params(sae)
    x = torch.randn(4, 8)
    acts = sae.encode(x)
    assert acts.shape == (4, 16)
    out = sae.decode(acts)
    assert out.shape == (4, 8)


# ============ LISTA refinement ============


def test_zero_refinement():
    sae = _build_sae(n_refinement_steps=0)
    _random_params(sae)
    x = torch.randn(4, 8)
    acts = sae.encode(x)
    assert acts.shape == (4, 16)


def test_refinement_changes_activations():
    sae_no_refine = _build_sae(n_refinement_steps=0)
    sae_refine = _build_sae(n_refinement_steps=2, eta=0.5)
    _random_params(sae_no_refine)
    sd = sae_no_refine.state_dict()
    sd.pop("topk_thresholds", None)
    sae_refine.load_state_dict(sd, strict=False)

    x = torch.randn(8, 8)
    acts_no = sae_no_refine.encode(x)
    acts_yes = sae_refine.encode(x)
    assert not torch.allclose(acts_no, acts_yes)


def test_multiple_refinement_steps():
    sae = _build_sae(n_refinement_steps=3, eta=0.3)
    _random_params(sae)
    x = torch.randn(4, 8)
    output = sae.training_forward_pass(TrainStepInput(x, {}, None, 0, True))
    assert output.loss.isfinite()
    assert sae.topk_thresholds.shape == (4,)


# ============ Decreasing K schedule ============


def test_lerp_basic():
    assert _lerp(60.0, 25.0, 0, 100) == 60.0
    assert _lerp(60.0, 25.0, 100, 100) == 25.0
    assert _lerp(60.0, 25.0, 50, 100) == 42.5


def test_lerp_edge_cases():
    assert _lerp(60.0, 25.0, -10, 100) == 60.0
    assert _lerp(60.0, 25.0, 200, 100) == 25.0
    assert _lerp(60.0, 25.0, 50, 0) == 25.0  # n_steps=0 returns end


def test_k_schedule_enabled():
    sae = _build_sae(initial_k=12.0, transition_k_duration_steps=100)
    assert sae._k_schedule is True
    assert sae.get_k(0) == 12.0
    assert sae.get_k(100) == 4.0
    assert sae.get_k(50) == 8.0


def test_k_schedule_disabled():
    sae = _build_sae()
    assert sae._k_schedule is False
    assert sae.get_k(0) == 4.0
    assert sae.get_k(100) == 4.0


def test_k_schedule_applied_in_forward():
    sae = _build_sae(initial_k=12.0, transition_k_duration_steps=100)
    _random_params(sae)
    x = torch.randn(4, 8)

    # At step 0, k should be 12 (initial)
    output = sae.training_forward_pass(TrainStepInput(x, {}, None, 0, True))
    assert output.metrics["current_k"] == 12.0

    # At step 50, k should be 8
    output = sae.training_forward_pass(TrainStepInput(x, {}, None, 50, True))
    assert output.metrics["current_k"] == 8.0

    # At step 100, k should be 4 (target)
    output = sae.training_forward_pass(TrainStepInput(x, {}, None, 100, True))
    assert output.metrics["current_k"] == 4.0


def test_k_schedule_changes_activations():
    """Decreasing K should produce different L0 at different steps."""
    sae = _build_sae(initial_k=12.0, transition_k_duration_steps=100)
    _random_params(sae)
    x = torch.randn(32, 8)

    output_early = sae.training_forward_pass(TrainStepInput(x, {}, None, 0, True))
    l0_early = (output_early.feature_acts != 0).float().sum(dim=-1).mean()

    output_late = sae.training_forward_pass(TrainStepInput(x, {}, None, 100, True))
    l0_late = (output_late.feature_acts != 0).float().sum(dim=-1).mean()

    # Early should have higher L0 than late
    assert l0_early > l0_late


# ============ Training forward pass ============


def test_training_forward_pass_runs():
    sae = _build_sae()
    _random_params(sae)
    x = torch.randn(4, 8)
    output = sae.training_forward_pass(TrainStepInput(x, {"l1": 0.1}, None, 0, True))
    assert output.loss > 0
    assert output.loss.isfinite()
    assert output.sae_out.shape == x.shape
    assert output.feature_acts.shape == (4, 16)


def test_loss_keys():
    sae = _build_sae()
    _random_params(sae)
    x = torch.randn(4, 8)
    output = sae.training_forward_pass(TrainStepInput(x, {}, None, 0, True))
    assert "mse_loss" in output.losses
    assert "inner_recons_loss" in output.losses
    assert "auxiliary_reconstruction_loss" in output.losses


def test_metrics_logged():
    sae = _build_sae()
    _random_params(sae)
    x = torch.randn(4, 8)
    output = sae.training_forward_pass(TrainStepInput(x, {}, None, 0, True))
    assert "term_tilt" in output.metrics
    assert "current_k" in output.metrics


# ============ TERM loss ============


def test_term_mean_zero_tilt():
    loss = torch.randn(32).abs()
    result = term_mean(loss, 0.0)
    torch.testing.assert_close(result, loss.mean())


def test_term_mean_positive_tilt_upweights():
    loss = torch.randn(32).abs()
    tilted = term_mean(loss, 5.0)
    assert tilted >= loss.mean()


def test_term_training():
    sae = _build_sae(term_tilt=0.002)
    _random_params(sae)
    x = torch.randn(8, 8)
    output = sae.training_forward_pass(TrainStepInput(x, {}, None, 0, True))
    assert output.loss.isfinite()
    assert output.metrics["term_tilt"] == 0.002


# ============ Frequency sorting ============


def test_frequency_sorting():
    sae = _build_sae(use_frequency_sorted_matryoshka=True)
    _random_params(sae)
    x = torch.randn(8, 8)

    for i in range(10):
        sae.training_forward_pass(TrainStepInput(x, {}, None, i, True))

    indices = sae._get_sorted_indices()
    assert indices is not None
    assert indices.shape == (16,)


def test_no_frequency_sorting():
    sae = _build_sae(use_frequency_sorted_matryoshka=False)
    assert sae.firing_frequency_tracker is None
    assert sae._get_sorted_indices() is None


def test_frequency_tracker_ema():
    tracker = FiringFrequencyTracker(d_sae=4, ema_decay=0.9, device="cpu")
    # First update initializes
    acts1 = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    tracker.update(acts1)
    torch.testing.assert_close(
        tracker.frequencies, torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    )
    # Second update uses EMA
    acts2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    tracker.update(acts2)
    expected = torch.tensor([0.9, 0.1, 0.0, 0.9], dtype=torch.float64)
    torch.testing.assert_close(tracker.frequencies, expected)


# ============ Gradients ============


def test_gradients_flow():
    sae = _build_sae()
    _random_params(sae)
    x = torch.randn(4, 8)
    output = sae.training_forward_pass(TrainStepInput(x, {}, None, 0, True))
    output.loss.backward()

    assert sae.W_enc.grad is not None
    assert sae.W_dec.grad is not None
    assert sae.b_enc.grad is not None


def test_gradients_with_k_schedule():
    sae = _build_sae(initial_k=12.0, transition_k_duration_steps=100)
    _random_params(sae)
    x = torch.randn(4, 8)
    output = sae.training_forward_pass(TrainStepInput(x, {}, None, 50, True))
    output.loss.backward()
    assert sae.W_enc.grad is not None
    assert sae.W_dec.grad is not None


# ============ Save/load checkpoint ============


def test_save_load_checkpoint(tmp_path: Path):
    sae = _build_sae()
    _random_params(sae)

    x = torch.randn(4, 8)
    original_acts = sae.encode(x)
    original_out = sae.decode(original_acts)

    sae.save_model(tmp_path)

    loaded = _build_sae()
    loaded.load_weights_from_checkpoint(tmp_path)

    torch.testing.assert_close(loaded.W_enc, sae.W_enc)
    torch.testing.assert_close(loaded.W_dec, sae.W_dec)
    torch.testing.assert_close(loaded.b_enc, sae.b_enc)
    torch.testing.assert_close(loaded.b_dec, sae.b_dec)

    loaded_acts = loaded.encode(x)
    loaded_out = loaded.decode(loaded_acts)

    torch.testing.assert_close(loaded_acts, original_acts)
    torch.testing.assert_close(loaded_out, original_out)


# ============ Inference saving ============


def test_saves_as_inference(tmp_path: Path):
    sae = _build_sae(rescale_acts_by_decoder_norm=False)
    _random_params(sae)

    test_input = torch.randn(30, 8)
    for i in range(1000):
        sae.training_forward_pass(TrainStepInput(test_input, {}, None, i, True))

    sae.save_inference_model(tmp_path)
    loaded = SAE.load_from_disk(str(tmp_path))

    assert loaded.cfg.architecture() == "lista_matryoshka_decreasingk"
    assert isinstance(loaded, ListaMatryoshkaSAE)

    state_dict = loaded.state_dict()
    assert "thresholds" in state_dict
    assert "eta" in state_dict
    assert "topk_thresholds" not in state_dict
    assert "topk_threshold" not in state_dict

    sae.eval()
    torch.testing.assert_close(
        loaded(test_input),
        sae(test_input),
        rtol=1e-3,
        atol=1e-2,
    )


@pytest.mark.parametrize("rescale_acts_by_decoder_norm", [True, False])
def test_inference_output_matches_training(
    rescale_acts_by_decoder_norm: bool, tmp_path: Path
):
    sae = _build_sae(rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm)
    _random_params(sae)

    test_input = torch.randn(30, 8)
    for i in range(1000):
        sae.training_forward_pass(TrainStepInput(test_input, {}, None, i, True))

    sae.save_inference_model(tmp_path)
    loaded = SAE.load_from_disk(str(tmp_path))

    sae.eval()
    torch.testing.assert_close(
        loaded(test_input),
        sae(test_input),
        rtol=1e-3,
        atol=1e-2,
    )


def test_inference_roundtrip(tmp_path: Path):
    sae = _build_sae(n_refinement_steps=2, eta=0.5, rescale_acts_by_decoder_norm=False)
    _random_params(sae)

    test_input = torch.randn(30, 8)
    for i in range(100):
        sae.training_forward_pass(TrainStepInput(test_input, {}, None, i, True))

    sae.save_inference_model(tmp_path)
    loaded = SAE.load_from_disk(str(tmp_path))

    assert isinstance(loaded, ListaMatryoshkaSAE)
    assert loaded.cfg.n_refinement_steps == 2
    assert loaded.thresholds.shape == (3, 16)
    assert loaded.eta.shape == (2,)

    # Save again should be stable
    loaded.save_model(tmp_path / "resave")
    reloaded = SAE.load_from_disk(str(tmp_path / "resave"))
    torch.testing.assert_close(reloaded.encode(test_input), loaded.encode(test_input))


def test_inference_learnable_eta(tmp_path: Path):
    sae = _build_sae(
        n_refinement_steps=2,
        eta=0.5,
        learnable_eta=True,
        rescale_acts_by_decoder_norm=False,
    )
    _random_params(sae)

    assert sae.eta_params is not None
    sae.eta_params[0].data.fill_(0.7)
    sae.eta_params[1].data.fill_(0.3)

    test_input = torch.randn(30, 8)
    for i in range(1000):
        sae.training_forward_pass(TrainStepInput(test_input, {}, None, i, True))

    sae.save_inference_model(tmp_path)
    loaded = SAE.load_from_disk(str(tmp_path))

    assert isinstance(loaded, ListaMatryoshkaSAE)
    torch.testing.assert_close(loaded.eta[0], sae.eta_params[0].detach())
    torch.testing.assert_close(loaded.eta[1], sae.eta_params[1].detach())

    sae.eval()
    torch.testing.assert_close(
        loaded.encode(test_input),
        sae.encode(test_input),
        rtol=1e-3,
        atol=1e-2,
    )


def test_zero_refinement_saves(tmp_path: Path):
    sae = _build_sae(n_refinement_steps=0, rescale_acts_by_decoder_norm=False)
    _random_params(sae)

    test_input = torch.randn(30, 8)
    for i in range(1000):
        sae.training_forward_pass(TrainStepInput(test_input, {}, None, i, True))

    sae.save_inference_model(tmp_path)
    loaded = SAE.load_from_disk(str(tmp_path))

    assert isinstance(loaded, ListaMatryoshkaSAE)
    assert loaded.cfg.n_refinement_steps == 0
    assert loaded.thresholds.shape == (1, 16)
    assert loaded.eta.shape == (0,)

    sae.eval()
    torch.testing.assert_close(
        loaded(test_input),
        sae(test_input),
        rtol=1e-3,
        atol=1e-2,
    )


# ============ Inference SAE directly ============


def test_inference_sae_shapes():
    cfg = ListaMatryoshkaSAEConfig(
        d_in=8,
        d_sae=16,
        n_refinement_steps=1,
        dtype="float32",
        device="cpu",
    )
    sae = ListaMatryoshkaSAE(cfg)
    _random_params(sae)

    x = torch.randn(4, 8)
    acts = sae.encode(x)
    assert acts.shape == (4, 16)
    out = sae.decode(acts)
    assert out.shape == (4, 8)


def test_inference_fold_warns():
    cfg = ListaMatryoshkaSAEConfig(
        d_in=8,
        d_sae=16,
        n_refinement_steps=1,
        dtype="float32",
        device="cpu",
    )
    sae = ListaMatryoshkaSAE(cfg)
    with pytest.warns(match="not supported"):
        sae.fold_W_dec_norm()


# ============ Dead latent aux loss ============


def test_dead_latents_aux_loss():
    sae = _build_sae()
    x = torch.randn(4, 8)

    with torch.no_grad():
        out_no_dead = sae.training_forward_pass(TrainStepInput(x, {}, None, 0, True))
        out_all_dead = sae.training_forward_pass(
            TrainStepInput(x, {}, torch.ones(16, dtype=torch.bool), 0, True)
        )

    assert (
        out_all_dead.losses["auxiliary_reconstruction_loss"]
        > out_no_dead.losses["auxiliary_reconstruction_loss"]
    )


# ============ Matryoshka validation ============


def test_matryoshka_widths_auto_append():
    cfg = ListaMatryoshkaTrainingSAEConfig(
        d_in=8,
        d_sae=16,
        k=4,
        matryoshka_widths=[4, 8],
        dtype="float32",
        device="cpu",
    )
    sae = ListaMatryoshkaTrainingSAE(cfg)
    assert cfg.matryoshka_widths[-1] == 16


def test_matryoshka_widths_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        cfg = ListaMatryoshkaTrainingSAEConfig(
            d_in=8,
            d_sae=16,
            k=4,
            matryoshka_widths=[8, 4, 16],
            dtype="float32",
            device="cpu",
        )
        ListaMatryoshkaTrainingSAE(cfg)


def test_no_matryoshka_widths():
    """Without matryoshka widths, inner_recons_loss should be 0."""
    sae = _build_sae(
        matryoshka_widths=[16]
    )  # only d_sae, skip_final=True -> no inner losses
    _random_params(sae)
    x = torch.randn(4, 8)
    output = sae.training_forward_pass(TrainStepInput(x, {}, None, 0, True))
    assert output.losses["inner_recons_loss"].item() == 0.0


# ============ Full recipe config test ============


def test_full_recipe_config():
    """Test the actual full recipe config used in production."""
    cfg = ListaMatryoshkaTrainingSAEConfig(
        d_in=768,
        d_sae=4096,
        k=25,
        matryoshka_widths=[128, 512, 2048],
        detach_matryoshka_losses=True,
        use_frequency_sorted_matryoshka=True,
        n_refinement_steps=1,
        eta=0.3,
        term_tilt=0.002,
        initial_k=60.0,
        transition_k_duration_steps=195312,
        dtype="float32",
        device="cpu",
    )
    sae = ListaMatryoshkaTrainingSAE(cfg)
    assert sae.cfg.matryoshka_widths == [128, 512, 2048, 4096]
    assert sae.firing_frequency_tracker is not None
    assert sae._k_schedule is True
    x = torch.randn(2, 768)
    output = sae.training_forward_pass(TrainStepInput(x, {}, None, 0, True))
    assert output.loss.isfinite()
    assert output.metrics["current_k"] == 60.0
    output.loss.backward()


@pytest.mark.parametrize("rescale_acts_by_decoder_norm", [True, False])
def test_inference_matches_training_all_configs(
    rescale_acts_by_decoder_norm: bool,
    tmp_path: Path,
):
    """Comprehensive test: training eval and inference SAE match for all config combinations."""
    sae = _build_sae(
        rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
    )
    _random_params(sae)

    test_input = torch.randn(30, 8)
    for i in range(1000):
        sae.training_forward_pass(TrainStepInput(test_input, {}, None, i, True))

    sae.save_inference_model(tmp_path)
    loaded = SAE.load_from_disk(str(tmp_path))

    sae.eval()
    torch.testing.assert_close(
        loaded(test_input),
        sae(test_input),
        rtol=1e-3,
        atol=1e-2,
    )


def test_inference_matches_training_with_k_schedule(tmp_path: Path):
    """Verify K schedule doesn't affect inference (inference uses thresholds, not BatchTopK)."""
    sae = _build_sae(
        initial_k=12.0,
        transition_k_duration_steps=100,
        rescale_acts_by_decoder_norm=False,
    )
    _random_params(sae)

    test_input = torch.randn(30, 8)
    # Run enough steps past K transition so thresholds converge at the final K
    for i in range(1000):
        sae.training_forward_pass(TrainStepInput(test_input, {}, None, i, True))

    sae.save_inference_model(tmp_path)
    loaded = SAE.load_from_disk(str(tmp_path))

    sae.eval()
    torch.testing.assert_close(
        loaded(test_input),
        sae(test_input),
        rtol=1e-3,
        atol=1e-2,
    )
