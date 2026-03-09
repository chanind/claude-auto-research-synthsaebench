"""LISTA-Matryoshka SAE with Decreasing K for SynthSAEBench-16k.

Self-contained implementation using only SAELens.

Training architecture: "lista_matryoshka_decreasingk_batchtopk"
    - Matryoshka multi-scale training ([128, 512, 2048, 4096])
    - LISTA iterative refinement (T=1, eta=0.3)
    - Frequency sorting (EMA-tracked firing frequencies)
    - Detached matryoshka inner losses
    - TERM loss (tilt=0.002)
    - LR decay (final 1/3 of training)
    - Decreasing K schedule (initial_k -> k, linearly)

Inference architecture: "lista_matryoshka_decreasingk"
    - Per-step thresholds (JumpReLU-style) instead of BatchTopK
    - LISTA refinement with learned per-step eta
"""

import warnings
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from sae_lens import (
    SAE,
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    SAEConfig,
    TrainingSAE,
)
from sae_lens.registry import register_sae_class, register_sae_training_class
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import act_times_W_dec
from typing_extensions import override


# ============================================================================
# Utility modules
# ============================================================================


class FiringFrequencyTracker(nn.Module):
    """Track per-latent firing frequencies via EMA."""

    frequencies: torch.Tensor
    _initialized: torch.Tensor

    def __init__(self, d_sae: int, ema_decay: float = 0.99, device: str = "cpu"):
        super().__init__()
        self.ema_decay = ema_decay
        self.register_buffer(
            "frequencies", torch.zeros(d_sae, device=device, dtype=torch.float64)
        )
        self.register_buffer(
            "_initialized", torch.zeros(1, device=device, dtype=torch.bool)
        )

    @torch.no_grad()
    def update(self, feature_acts: torch.Tensor) -> None:
        flat = feature_acts.reshape(-1, feature_acts.shape[-1])
        batch_freq = (flat != 0).sum(dim=0).to(self.frequencies.dtype) / flat.shape[0]
        if not self._initialized.item():
            self.frequencies.copy_(batch_freq)
            self._initialized.fill_(True)
        else:
            self.frequencies.mul_(self.ema_decay).add_(
                batch_freq, alpha=1 - self.ema_decay
            )


# ============================================================================
# TERM loss helper
# ============================================================================


def term_mean(per_sample_loss: torch.Tensor, tilt: float) -> torch.Tensor:
    """Tilted Empirical Risk Minimization mean.

    Upweights hard (high-loss) samples via softmax reweighting.
    When tilt <= 0, returns standard mean.
    Uses normalized tilt: tilt * loss / mean(loss) for robustness.
    """
    if tilt <= 0:
        return per_sample_loss.mean()
    weights = torch.softmax(
        per_sample_loss.detach() * tilt / per_sample_loss.detach().mean(), dim=0
    )
    return (weights * per_sample_loss).sum()


# ============================================================================
# Linear interpolation utility
# ============================================================================


def _lerp(start: float, end: float, step: int, n_steps: int) -> float:
    """Linearly interpolate from start to end over n_steps."""
    if n_steps <= 0 or step >= n_steps:
        return end
    if step <= 0:
        return start
    return start + (end - start) * step / n_steps


# ============================================================================
# Training Config
# ============================================================================


@dataclass
class ListaMatryoshkaTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """Config for LISTA-Matryoshka training SAE with Decreasing K.

    Components:
    - Matryoshka multi-scale training with detached inner losses
    - LISTA iterative refinement
    - Frequency-sorted matryoshka prefixes
    - TERM loss
    - Decreasing K schedule
    """

    # LISTA refinement
    n_refinement_steps: int = 1
    eta: float = 0.3
    learnable_eta: bool = False

    # Matryoshka
    matryoshka_widths: list[int] = field(default_factory=list)
    detach_matryoshka_losses: bool = True
    skip_final_matryoshka_width: bool = True
    include_outer_loss: bool = True

    # Frequency sorting
    use_frequency_sorted_matryoshka: bool = False
    firing_frequency_ema_decay: float = 0.99

    # TERM loss
    term_tilt: float = 0.0

    # Decreasing K schedule
    initial_k: float | None = None
    transition_k_duration_steps: int | None = None

    @override
    @classmethod
    def architecture(cls) -> str:
        return "lista_matryoshka_decreasingk_batchtopk"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return ListaMatryoshkaSAEConfig


# ============================================================================
# Training SAE
# ============================================================================


class ListaMatryoshkaTrainingSAE(BatchTopKTrainingSAE):
    """LISTA-Matryoshka training SAE with Decreasing K.

    Combines: LISTA refinement, matryoshka multi-scale training,
    TERM loss, frequency sorting, and decreasing K schedule.
    """

    cfg: ListaMatryoshkaTrainingSAEConfig  # type: ignore

    def __init__(
        self,
        cfg: ListaMatryoshkaTrainingSAEConfig,
        use_error_term: bool = False,
    ):
        super().__init__(cfg, use_error_term)

        # ---- LISTA ----
        if cfg.learnable_eta:
            self.eta_params = nn.ParameterList(
                [
                    nn.Parameter(torch.tensor(cfg.eta))
                    for _ in range(cfg.n_refinement_steps)
                ]
            )
        else:
            self.eta_params: nn.ParameterList | None = None  # type: ignore[no-redef]

        # Per-step topk thresholds for inference conversion
        self.register_buffer(
            "topk_thresholds",
            torch.zeros(cfg.n_refinement_steps + 1, dtype=torch.double),
        )
        self._per_step_acts: list[torch.Tensor] = []

        # ---- Matryoshka ----
        if len(cfg.matryoshka_widths) == 0 or cfg.matryoshka_widths[-1] != cfg.d_sae:
            warnings.warn("Adding d_sae to matryoshka_widths as the final level.")
            cfg.matryoshka_widths.append(cfg.d_sae)
        for i in range(len(cfg.matryoshka_widths) - 1):
            if cfg.matryoshka_widths[i] >= cfg.matryoshka_widths[i + 1]:
                raise ValueError("matryoshka_widths must be strictly increasing.")

        # ---- Frequency sorting ----
        if cfg.use_frequency_sorted_matryoshka:
            self.firing_frequency_tracker = FiringFrequencyTracker(
                d_sae=cfg.d_sae,
                ema_decay=cfg.firing_frequency_ema_decay,
                device=cfg.device,
            )
        else:
            self.firing_frequency_tracker: FiringFrequencyTracker | None = None  # type: ignore[no-redef]

        # ---- K schedule ----
        self._k_schedule = (
            cfg.initial_k is not None and cfg.transition_k_duration_steps is not None
        )

    # ---- K schedule ----

    def get_k(self, step: int) -> float:
        if not self._k_schedule:
            return self.cfg.k
        assert self.cfg.initial_k is not None
        assert self.cfg.transition_k_duration_steps is not None
        return _lerp(
            self.cfg.initial_k,
            self.cfg.k,
            step,
            self.cfg.transition_k_duration_steps,
        )

    # ---- Eta access ----

    def _get_eta_for_step(self, refinement_step: int) -> float | torch.Tensor:
        if self.eta_params is not None:
            return self.eta_params[refinement_step]
        return self.cfg.eta

    # ---- Frequency sorting ----

    def _get_sorted_indices(self) -> torch.Tensor | None:
        if self.firing_frequency_tracker is None:
            return None
        return self.firing_frequency_tracker.frequencies.argsort(descending=True)

    # ---- Encoding ----

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norm: torch.Tensor | None = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norm = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norm

        feature_acts = self.activation_fn(hidden_pre)
        if self.training:
            self._per_step_acts = [feature_acts.detach()]

        for t in range(self.cfg.n_refinement_steps):
            # Reconstruct using W_dec for LISTA residual computation
            if self.cfg.rescale_acts_by_decoder_norm:
                assert W_dec_norm is not None
                recon = (feature_acts / W_dec_norm) @ self.W_dec
            else:
                recon = feature_acts @ self.W_dec
            residual = sae_in - recon

            # Encode residual with standard linear encoder
            delta = residual @ self.W_enc
            if self.cfg.rescale_acts_by_decoder_norm:
                assert W_dec_norm is not None
                delta = delta * W_dec_norm

            eta = self._get_eta_for_step(t)
            hidden_pre = hidden_pre + eta * delta
            feature_acts = self.activation_fn(hidden_pre)
            if self.training:
                self._per_step_acts.append(feature_acts.detach())

        return self.hook_sae_acts_post(feature_acts), hidden_pre

    # ---- Decoding ----

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        sae_out_pre = (
            act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            + self.b_dec
        )
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    # ---- Matryoshka decoding ----

    def iterable_decode(
        self, feature_acts: torch.Tensor
    ) -> Generator[torch.Tensor, None, None]:
        """Decode at each matryoshka prefix width, yielding partial reconstructions."""
        decoded = self.b_dec

        acts = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            acts = acts / self.W_dec.norm(dim=-1)

        sorted_indices = self._get_sorted_indices()
        prev_portion = 0

        widths = self.cfg.matryoshka_widths
        if self.cfg.skip_final_matryoshka_width:
            widths = widths[:-1]

        for i, portion in enumerate(widths):
            if portion > 0:
                if sorted_indices is not None:
                    idx = sorted_indices[prev_portion:portion]
                    current_delta = acts[:, idx] @ self.W_dec[idx]
                else:
                    current_delta = (
                        acts[:, prev_portion:portion] @ self.W_dec[prev_portion:portion]
                    )
                if self.cfg.detach_matryoshka_losses and i > 0:
                    decoded = decoded.detach()
                decoded = decoded + current_delta
                prev_portion = portion
            yield decoded

    # ---- TopK threshold tracking ----

    @override
    @torch.no_grad()
    def update_topk_threshold(self, acts_topk: torch.Tensor) -> None:
        """Update per-step thresholds from the per-step activations."""
        lr = self.cfg.topk_threshold_lr
        with torch.autocast(self.topk_thresholds.device.type, enabled=False):
            for step_idx, acts in enumerate(self._per_step_acts):
                positive_mask = acts > 0
                if positive_mask.any():
                    min_positive = (
                        acts[positive_mask].min().to(self.topk_thresholds.dtype)
                    )
                    self.topk_thresholds[step_idx] = (1 - lr) * self.topk_thresholds[
                        step_idx
                    ] + lr * min_positive
        self._per_step_acts = []

    # ---- Training forward pass ----

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        # Apply K schedule
        current_k = self.get_k(step_input.n_training_steps)
        self.activation_fn.k = current_k

        # Forward pass
        x = step_input.sae_in
        feature_acts, hidden_pre = self.encode_with_hidden_pre(x)
        sae_out = self.decode(feature_acts)
        sae_in = self.process_sae_in(x)

        # Update topk thresholds
        self.update_topk_threshold(feature_acts)

        # MSE loss with TERM
        per_sample_mse = self.mse_loss_fn(sae_out, sae_in).sum(dim=-1)
        mse_loss = term_mean(per_sample_mse, self.cfg.term_tilt)
        losses: dict[str, torch.Tensor] = {"mse_loss": mse_loss}

        # Dead latent aux loss
        aux_loss = self.calculate_topk_aux_loss(
            sae_in, sae_out, hidden_pre, step_input.dead_neuron_mask
        )
        losses["auxiliary_reconstruction_loss"] = aux_loss

        output = TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=mse_loss,
            losses=losses,
        )

        # Matryoshka inner losses
        self._matryoshka_train_step(step_input, output)

        # Metrics
        if step_input.is_logging_step:
            output.metrics["term_tilt"] = self.cfg.term_tilt
            output.metrics["current_k"] = current_k

        return output

    def _matryoshka_train_step(
        self,
        step_input: TrainStepInput,
        output: TrainStepOutput,
    ) -> None:
        """Compute matryoshka inner losses."""
        feature_acts = output.feature_acts
        sae_in = output.sae_in

        # Update frequency tracker
        if self.firing_frequency_tracker is not None:
            self.firing_frequency_tracker.update(feature_acts)

        if not self.cfg.include_outer_loss:
            output.losses = {}

        # Inner reconstruction losses with TERM
        inner_losses = []
        for _portion, partial_sae_out in zip(
            self.cfg.matryoshka_widths,
            self.iterable_decode(feature_acts),
        ):
            per_sample = self.mse_loss_fn(partial_sae_out, sae_in).sum(dim=-1)
            inner_losses.append(term_mean(per_sample, self.cfg.term_tilt))

        if inner_losses:
            output.losses["inner_recons_loss"] = torch.stack(inner_losses).sum()
        else:
            output.losses["inner_recons_loss"] = output.loss.new_tensor(0.0)

        # Sum all losses
        output.loss = torch.stack(list(output.losses.values())).sum()

    # ---- State dict processing ----

    @override
    @torch.no_grad()
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        # Extract topk thresholds before super() removes them
        topk_thresholds = state_dict.pop("topk_thresholds")

        # Get eta values
        if self.eta_params is not None:
            eta = torch.stack([p.detach().clone() for p in self.eta_params])
        else:
            eta = torch.full(
                (self.cfg.n_refinement_steps,),
                self.cfg.eta,
                dtype=self.W_enc.dtype,
                device=self.W_enc.device,
            )

        # Remove training-only params
        for key in list(state_dict.keys()):
            if key.startswith("eta_params."):
                del state_dict[key]

        # Save W_dec norm for threshold conversion (thresholds were computed in
        # scaled space: hidden_pre * W_dec_norm).
        original_W_dec_norm = None
        if self.cfg.rescale_acts_by_decoder_norm:
            original_W_dec_norm = state_dict["W_dec"].norm(dim=-1)

        # Remove other training-only keys
        for key in ["topk_threshold", "pinned_W_enc", "pinned_b_enc"]:
            state_dict.pop(key, None)
        for key in list(state_dict.keys()):
            if key.startswith("firing_frequency_tracker."):
                del state_dict[key]

        # Skip BatchTopKTrainingSAE's process_state_dict (which would add a single
        # threshold). Instead, call TrainingSAE's version directly.
        TrainingSAE.process_state_dict_for_saving_inference(self, state_dict)

        # Add inference params
        # Per-step thresholds: (n_steps+1, d_sae)
        thresholds = topk_thresholds.unsqueeze(1).expand(-1, self.cfg.d_sae).clone()
        thresholds = thresholds.to(dtype=self.W_enc.dtype)

        # If rescale_acts_by_decoder_norm was used, convert thresholds from scaled to
        # unscaled space.
        if original_W_dec_norm is not None:
            thresholds = thresholds / original_W_dec_norm.to(
                dtype=thresholds.dtype
            ).unsqueeze(0)

        state_dict["thresholds"] = thresholds
        state_dict["eta"] = eta.to(dtype=self.W_enc.dtype)


# ============================================================================
# Inference Config
# ============================================================================


@dataclass
class ListaMatryoshkaSAEConfig(SAEConfig):
    """Config for inference LISTA-Matryoshka SAE."""

    n_refinement_steps: int = 1

    @override
    @classmethod
    def architecture(cls) -> str:
        return "lista_matryoshka_decreasingk"

    @override
    def get_training_sae_cfg_class(self) -> type | None:
        return ListaMatryoshkaTrainingSAEConfig


# ============================================================================
# Inference SAE
# ============================================================================


class ListaMatryoshkaSAE(SAE["ListaMatryoshkaSAEConfig"]):
    """Inference version of LISTA-Matryoshka SAE.

    Uses per-step thresholds (JumpReLU-style) instead of BatchTopK,
    and stores learned eta values for refinement steps.
    """

    cfg: ListaMatryoshkaSAEConfig
    thresholds: nn.Parameter
    eta: nn.Parameter
    b_enc: nn.Parameter

    def __init__(
        self,
        cfg: ListaMatryoshkaSAEConfig,
        use_error_term: bool = False,
    ):
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self.thresholds = nn.Parameter(
            torch.zeros(
                self.cfg.n_refinement_steps + 1,
                self.cfg.d_sae,
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.eta = nn.Parameter(
            torch.ones(
                self.cfg.n_refinement_steps,
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    def _activation(self, hidden_pre: torch.Tensor, step: int) -> torch.Tensor:
        return torch.relu(hidden_pre) * (hidden_pre > self.thresholds[step])

    @override
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        feature_acts = self._activation(hidden_pre, step=0)

        for t in range(self.cfg.n_refinement_steps):
            recon = feature_acts @ self.W_dec
            residual = sae_in - recon
            delta = residual @ self.W_enc
            hidden_pre = hidden_pre + self.eta[t] * delta
            feature_acts = self._activation(hidden_pre, step=t + 1)

        return self.hook_sae_acts_post(feature_acts)

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        warnings.warn(
            "Folding W_dec norm is not supported for ListaMatryoshkaSAE "
            "(LISTA uses original W_dec norms for reconstruction), skipping."
        )


# ============================================================================
# Registry
# ============================================================================

register_sae_training_class(
    "lista_matryoshka_decreasingk_batchtopk",
    ListaMatryoshkaTrainingSAE,
    ListaMatryoshkaTrainingSAEConfig,
)

register_sae_class(
    "lista_matryoshka_decreasingk",
    ListaMatryoshkaSAE,
    ListaMatryoshkaSAEConfig,
)
