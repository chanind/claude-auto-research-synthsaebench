"""Training script for LISTA-Matryoshka SAE with Decreasing K.

Supports:
1. Default recipe validation (200M samples)
2. Component ablations (200M samples)
3. Multi-L0 sweep for paper plots (200M samples)

Usage:
    python train.py default               # Default recipe at K=25
    python train.py ablation <name>       # Single ablation
    python train.py ablations             # All ablations
    python train.py l0_sweep              # All L0s (15-45 step 5)
    python train.py <experiment_name>     # Specific experiment
"""

import json
import os
import sys
from pathlib import Path


# Must import sae module before anything else to register architectures
sys.path.insert(0, str(Path(__file__).parent))
import sae as _sae_module  # noqa: F401 - registers architectures

import torch
from sae_lens import LoggingConfig, SAE
from sae_lens.synthetic import (
    SyntheticModel,
    SyntheticSAERunner,
    SyntheticSAERunnerConfig,
)
from sae_lens.synthetic.evals import eval_sae_on_synthetic_data

from sae import ListaMatryoshkaTrainingSAE, ListaMatryoshkaTrainingSAEConfig

SPRINT_DIR = Path(__file__).parent
D_IN = 768
D_SAE = 4096
MODEL = "decoderesearch/synth-sae-bench-16k-v1"
BATCH_SIZE = 1024
LR = 3e-4
TRAINING_SAMPLES = 200_000_000
TOTAL_STEPS = TRAINING_SAMPLES // BATCH_SIZE  # 195312


def _make_config(
    k: int = 25,
    training_samples: int = TRAINING_SAMPLES,
    # Component toggles for ablations
    use_refinement: bool = True,
    n_refinement_steps: int = 1,
    eta: float = 0.3,
    use_matryoshka: bool = True,
    matryoshka_widths: list[int] | None = None,
    use_term: bool = True,
    term_tilt: float = 0.002,
    use_freq_sort: bool = True,
    use_lr_decay: bool = True,
    detach_matryoshka_losses: bool = True,
    use_decreasing_k: bool = True,
    initial_k: float = 60.0,
) -> tuple[
    ListaMatryoshkaTrainingSAEConfig,
    ListaMatryoshkaTrainingSAE,
    SyntheticSAERunnerConfig,
]:
    total_steps = training_samples // BATCH_SIZE
    lr_decay_steps = total_steps // 3 if use_lr_decay else 0

    if matryoshka_widths is None:
        matryoshka_widths = [128, 512, 2048] if use_matryoshka else []

    cfg = ListaMatryoshkaTrainingSAEConfig(
        d_in=D_IN,
        d_sae=D_SAE,
        k=k,
        # LISTA
        n_refinement_steps=n_refinement_steps if use_refinement else 0,
        eta=eta,
        # Matryoshka
        matryoshka_widths=matryoshka_widths,
        detach_matryoshka_losses=detach_matryoshka_losses,
        skip_final_matryoshka_width=True,
        include_outer_loss=True,
        # Frequency sorting
        use_frequency_sorted_matryoshka=use_freq_sort,
        # TERM
        term_tilt=term_tilt if use_term else 0.0,
        # Decreasing K
        initial_k=initial_k if use_decreasing_k else None,
        transition_k_duration_steps=total_steps if use_decreasing_k else None,
        # Standard settings
        dtype="float32",
        device="cuda",
    )

    sae_instance = ListaMatryoshkaTrainingSAE(cfg)

    runner_config = SyntheticSAERunnerConfig(
        synthetic_model=MODEL,
        sae=cfg,
        training_samples=training_samples,
        batch_size=BATCH_SIZE,
        lr=LR,
        lr_decay_steps=lr_decay_steps,
        device="cuda",
        eval_frequency=5_000,
        eval_samples=500_000,
        autocast_sae=True,
        autocast_data=True,
        logger=LoggingConfig(log_to_wandb=False),
    )

    return cfg, sae_instance, runner_config


def run_experiment(
    name: str,
    training_samples: int = TRAINING_SAMPLES,
    eval_inference: bool = True,
    **config_overrides: object,
) -> dict:
    output_dir = SPRINT_DIR / "output" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    cfg, sae_instance, runner_config = _make_config(
        training_samples=training_samples, **config_overrides
    )
    runner_config.output_path = str(output_dir)

    runner = SyntheticSAERunner(runner_config, override_sae=sae_instance)
    result = runner.run()

    results = {}
    if result.final_eval is not None:
        results = {
            "name": name,
            "mcc": result.final_eval.mcc,
            "f1": result.final_eval.classification.f1_score,
            "precision": result.final_eval.classification.precision,
            "recall": result.final_eval.classification.recall,
            "explained_variance": result.final_eval.explained_variance,
            "sae_l0": result.final_eval.sae_l0,
            "dead_latents": result.final_eval.dead_latents,
            "shrinkage": result.final_eval.shrinkage,
            "uniqueness": result.final_eval.uniqueness,
        }
        print(f"\n--- Training eval: {name} ---")
        for k_name, v in results.items():
            if k_name != "name":
                print(f"  {k_name}: {v}")

        # Also evaluate the inference SAE
        if eval_inference:
            try:
                inf_results = _eval_inference_sae(output_dir, name)
                results["inference_mcc"] = inf_results.get("mcc")
                results["inference_f1"] = inf_results.get("f1")
                results["inference_explained_variance"] = inf_results.get(
                    "explained_variance"
                )
                results["inference_sae_l0"] = inf_results.get("sae_l0")
            except Exception as e:
                print(f"  WARNING: Inference eval failed: {e}")
                import traceback

                traceback.print_exc()

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def _eval_inference_sae(output_dir: Path, name: str) -> dict:
    """Evaluate the saved inference SAE."""
    inference_dir = output_dir / "inference"
    if not inference_dir.exists():
        inference_dir = output_dir

    loaded = SAE.load_from_disk(str(inference_dir), device="cuda")
    model = SyntheticModel.from_pretrained(MODEL, device="cuda")

    eval_result = eval_sae_on_synthetic_data(
        sae=loaded,
        feature_dict=model.feature_dict,
        activations_generator=model.activation_generator,
        num_samples=500_000,
    )

    inf_results = {
        "mcc": eval_result.mcc,
        "f1": eval_result.classification.f1_score,
        "explained_variance": eval_result.explained_variance,
        "sae_l0": eval_result.sae_l0,
        "dead_latents": eval_result.dead_latents,
    }
    print(f"\n--- Inference eval: {name} ---")
    for k_name, v in inf_results.items():
        print(f"  {k_name}: {v}")

    return inf_results


# ============ Experiment definitions ============


EXPERIMENTS = {
    # Default recipe (with decreasing K)
    "default": dict(),
    # Component ablations (remove one at a time)
    "ablation_no_refinement": dict(use_refinement=False),
    "ablation_no_matryoshka": dict(use_matryoshka=False),
    "ablation_no_term": dict(use_term=False),
    "ablation_no_freq_sort": dict(use_freq_sort=False),
    "ablation_no_lr_decay": dict(use_lr_decay=False),
    "ablation_no_detach_matryoshka": dict(detach_matryoshka_losses=False),
    "ablation_no_decreasing_k": dict(use_decreasing_k=False),
    # Vanilla BatchTopK baseline (no enhancements)
    "baseline_batchtopk": dict(
        use_refinement=False,
        use_matryoshka=False,
        use_term=False,
        use_freq_sort=False,
        use_lr_decay=False,
        use_decreasing_k=False,
    ),
}

# L0 sweep experiments (paper L0 values: 15, 20, 25, 30, 35, 40, 45)
L0_VALUES = [15, 20, 25, 30, 35, 40, 45]

for l0 in L0_VALUES:
    EXPERIMENTS[f"l0_{l0}"] = dict(k=l0)

# Baseline L0 sweep (vanilla BatchTopK)
for l0 in L0_VALUES:
    EXPERIMENTS[f"baseline_l0_{l0}"] = dict(
        k=l0,
        use_refinement=False,
        use_matryoshka=False,
        use_term=False,
        use_freq_sort=False,
        use_lr_decay=False,
        use_decreasing_k=False,
    )


def run_and_save_all(experiment_names: list[str]) -> dict:
    all_results = {}
    for name in experiment_names:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            continue
        try:
            result = run_experiment(name, **EXPERIMENTS[name])
            all_results[name] = result
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback

            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # Save summary
    summary_path = SPRINT_DIR / "results_summary.json"
    # Load existing results if present
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Name':<35} {'F1':>8} {'MCC':>8} {'VarExpl':>8} {'L0':>6} {'Dead':>6}")
    print(f"{'-'*100}")
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<35} ERROR")
        elif res:
            print(
                f"{name:<35} {res.get('f1', 0):>8.4f} {res.get('mcc', 0):>8.4f} "
                f"{res.get('explained_variance', 0):>8.4f} "
                f"{res.get('sae_l0', 0):>6.1f} {res.get('dead_latents', 0):>6}"
            )
    print(f"{'='*100}")

    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <experiment|ablations|l0_sweep|all>")
        print(f"\nAvailable experiments:")
        for name in sorted(EXPERIMENTS.keys()):
            print(f"  {name}")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "default":
        run_and_save_all(["default"])
    elif arg == "ablations":
        names = (
            ["default"]
            + [n for n in EXPERIMENTS if n.startswith("ablation_")]
            + ["baseline_batchtopk"]
        )
        run_and_save_all(names)
    elif arg == "l0_sweep":
        names = [f"l0_{l0}" for l0 in L0_VALUES]
        run_and_save_all(names)
    elif arg == "baseline_l0_sweep":
        names = [f"baseline_l0_{l0}" for l0 in L0_VALUES]
        run_and_save_all(names)
    elif arg == "all":
        run_and_save_all(list(EXPERIMENTS.keys()))
    elif arg in EXPERIMENTS:
        run_and_save_all([arg])
    else:
        print(f"Unknown: {arg}")
        sys.exit(1)
