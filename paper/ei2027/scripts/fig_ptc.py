"""EMVA1288 photon-transfer validation figure for the EI 2027 paper.

Simulates uniform-field frame pairs with open-cam's EMVA model
(tools/emva_theory.simulate_uniform_stack, same noise picture as
tools/apply_emva_noise.py) using the iphone_8 camera recipe, then estimates
the system gain K and dark noise with the EMVA1288 photon-transfer method and
compares against the analytic model.

Run from the repository root:
    venv/bin/python paper/ei2027/scripts/fig_ptc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

from camera_model import load_camera_model  # noqa: E402
from emva_theory import (  # noqa: E402
    dsnu_offset_map,
    emva1288_prnu,
    prnu_gain_map,
    simulate_uniform_stack,
    temporal_variance_dn_squared,
)

OUT = REPO / "paper" / "ei2027" / "figures"
RECIPE = REPO / "config" / "camera_recipes" / "iphone_8.yaml"
SHAPE = (256, 256)
SEED = 1288


def main() -> None:
    cam = load_camera_model(RECIPE)
    emva = cam["noise"]["emva"]
    K = float(emva["overall_system_gain_K_e_per_DN"]) * float(emva.get("iso_gain_factor", 1.0)) ** -1
    sigma_d = float(emva["sigma_d_e"])
    black = float(emva["black_level_DN"])
    full_well = float(cam["noise"]["adc"]["full_well_e"])
    prnu = float(emva["prnu_std_fraction"])
    dsnu = float(emva["dsnu_std_e"])

    rng = np.random.default_rng(SEED)
    g_map = prnu_gain_map(SHAPE, prnu, rng)
    d_map = dsnu_offset_map(SHAPE, dsnu_std_e=dsnu, dark_mean_e=0.0, rng=rng)

    mu_levels = np.concatenate([[0.0], np.geomspace(5.0, 1.15 * full_well, 28)])
    rows = []
    for i, mu in enumerate(mu_levels):
        stack = simulate_uniform_stack(
            mu_e=float(mu),
            n_frames=2,
            prnu_map=g_map,
            dsnu_map=d_map,
            dark_mean_e=0.0,
            sigma_d_e=sigma_d,
            K_e_per_DN=K,
            black_level_DN=black,
            full_well_e=full_well,
            use_poisson=True,
            seed=SEED + i,
        )
        mean_dn = float(stack.mean())
        var_dn = float(np.var(stack[0] - stack[1]) / 2.0)  # EMVA1288 temporal variance from a frame pair
        pred = temporal_variance_dn_squared(min(mu, full_well), sigma_d, K, use_poisson=True)
        rows.append({"mu_e": float(mu), "mean_dn": mean_dn, "var_dn": var_dn, "pred_var_dn": pred})

    dark = rows[0]
    lin = [r for r in rows[1:] if 0.0 < r["mu_e"] <= 0.7 * full_well]
    x = np.array([r["mean_dn"] - dark["mean_dn"] for r in lin])
    y = np.array([r["var_dn"] - dark["var_dn"] for r in lin])
    slope = float(np.sum(x * y) / np.sum(x * x))  # zero-intercept LSQ fit, EMVA1288 sec. 6.3
    K_est = 1.0 / slope
    sigma_d_est = float(np.sqrt(dark["var_dn"]) * K_est)
    rel_err = [abs(r["var_dn"] - r["pred_var_dn"]) / r["pred_var_dn"] for r in lin]

    bright_mu = 0.5 * full_well
    dark_stack = simulate_uniform_stack(
        mu_e=0.0,
        n_frames=16,
        prnu_map=g_map,
        dsnu_map=d_map,
        dark_mean_e=0.0,
        sigma_d_e=sigma_d,
        K_e_per_DN=K,
        black_level_DN=black,
        full_well_e=full_well,
        use_poisson=True,
        seed=SEED + 100,
    )
    bright_stack = simulate_uniform_stack(
        mu_e=bright_mu,
        n_frames=16,
        prnu_map=g_map,
        dsnu_map=d_map,
        dark_mean_e=0.0,
        sigma_d_e=sigma_d,
        K_e_per_DN=K,
        black_level_DN=black,
        full_well_e=full_well,
        use_poisson=True,
        seed=SEED + 101,
    )
    prnu_est = emva1288_prnu(dark_stack, bright_stack).prnu_fraction

    summary = {
        "recipe": str(RECIPE.relative_to(REPO)),
        "K_config_e_per_DN": K,
        "K_estimated_e_per_DN": K_est,
        "K_rel_error": abs(K_est - K) / K,
        "sigma_d_config_e": sigma_d,
        "sigma_d_estimated_e": sigma_d_est,
        "prnu_config": prnu,
        "prnu_estimated": float(prnu_est),
        "max_rel_var_error_linear_range": float(np.max(rel_err)),
        "median_rel_var_error_linear_range": float(np.median(rel_err)),
        "n_levels": len(rows),
        "frame_shape": list(SHAPE),
        "full_well_e": full_well,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ptc_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    md = np.array([r["mean_dn"] - dark["mean_dn"] for r in rows[1:]])
    axes[0].loglog(md, [r["var_dn"] for r in rows[1:]], "o", ms=3, label="Monte Carlo (frame pair)")
    axes[0].loglog(md, [r["pred_var_dn"] for r in rows[1:]], "-", lw=1, label="EMVA model")
    axes[0].axhline(dark["var_dn"], color="0.5", ls=":", lw=1, label="dark variance")
    axes[0].set_xlabel(r"$\mu_y-\mu_{y.dark}$ [DN]")
    axes[0].set_ylabel(r"$\sigma_y^2$ [DN$^2$]")
    axes[0].set_title("Photon transfer", fontsize=9)
    axes[0].legend(fontsize=6)
    snr = [(r["mean_dn"] - dark["mean_dn"]) / np.sqrt(max(r["var_dn"], 1e-12)) for r in rows[1:]]
    mu_p = np.array([r["mu_e"] for r in rows[1:]])
    axes[1].loglog(mu_p, snr, "o", ms=3, label="Monte Carlo")
    axes[1].loglog(mu_p, mu_p / np.sqrt(mu_p + sigma_d**2), "-", lw=1, label=r"$\mu_e/\sqrt{\mu_e+\sigma_d^2}$")
    axes[1].set_xlabel(r"$\mu_e$ [e$^-$]")
    axes[1].set_ylabel("SNR")
    axes[1].set_title("Temporal SNR", fontsize=9)
    axes[1].legend(fontsize=6)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ptc.pdf")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
