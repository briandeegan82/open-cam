#!/usr/bin/env python3
"""Turn the scorecard and validation JSON into the paper's LaTeX tables, macros and figure.

    venv/bin/python paper/ei2027_iqlab/scripts/make_tables.py \
        --scorecard out/iq_lab/scorecard/scorecard.json \
        --validation out/iq_lab/validation/iqlab_validation.json

Writes tables/{numbers,validation,hdr,optics,skin}.tex and figures/fig_scorecard.pdf under
paper/ei2027_iqlab/. Every number quoted in the manuscript comes from these files.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]

HDR_NAMES = {"linear": "Linear", "dcg": "Dual conversion gain", "split_pixel": "Split pixel", "lofic": "LOFIC", "multi_exposure": "Multi-exposure"}


def tt(s: str) -> str:
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def fin(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def vals(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows if fin(r.get(key))], dtype=float)


def median_dr(rows: list[dict]) -> float:
    v = vals(rows, "dr_snr1_db")
    return float(np.median(v)) if v.size else 0.0


def med(rows, key, fmt="{:.1f}") -> str:
    v = vals(rows, key)
    return fmt.format(float(np.median(v))) if v.size else "--"


def rng(rows, key, fmt="{:.1f}") -> str:
    v = vals(rows, key)
    if not v.size:
        return "--"
    lo, hi = fmt.format(v.min()), fmt.format(v.max())
    return lo if lo == hi else f"{lo}--{hi}"


def table(cols: str, head: list[str], body: list[list[str]]) -> str:
    out = [r"\begin{tabular}{@{}" + cols + "@{}}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in body]
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def optics_label(key: str) -> str:
    parts = key.split(":")
    if parts[0] == "realistic":
        lens, ap = parts[1].replace(".dat", ""), parts[2].replace("f", "", 1)
        return f"{tt(lens)}, {ap} aperture"
    return f"{parts[0]}, {parts[1]}" if len(parts) > 1 else key


# ------------------------------------------------------------------------------------ validation
def validation_tables(val: dict, macros: dict) -> str:
    res, tol = val["results"], val["tolerances"]

    def worst(suite, key, tier=None, absval=True):
        v = [abs(c[key]) if absval else c[key] for c in res[suite] if tier is None or c.get("tier") == tier]
        return max(v) if v else float("nan")

    n = sum(len(v) for v in res.values())
    macros["valCases"] = str(n)
    macros["valFailures"] = str(len(val.get("failures", [])))
    rows = [
        [
            "Edge MTF50 bias, clean (\\%)",
            f"{worst('edge', 'mtf50_bias_pct', 'clean'):.2f}",
            f"{tol['edge']['clean']['mtf50_bias_pct']:g}",
        ],
        [
            "Edge MTF50 RMS, SNR 30 (\\%)",
            f"{worst('edge', 'mtf50_rms_pct', 'noisy'):.2f}",
            f"{tol['edge']['noisy']['mtf50_rms_pct']:g}",
        ],
        [
            "Edge acutance, clean",
            f"{worst('edge', 'acutance_abs', 'clean'):.4f}",
            f"{tol['edge']['clean']['acutance_abs']:g}",
        ],
        [
            "Texture MTF RMS, clean",
            f"{worst('texture', 'mtf_rms_abs', 'clean'):.4f}",
            f"{tol['texture']['clean']['mtf_rms_abs']:g}",
        ],
        [
            "Texture MTF RMS, noisy",
            f"{worst('texture', 'mtf_rms_abs', 'noisy'):.4f}",
            f"{tol['texture']['noisy']['mtf_rms_abs']:g}",
        ],
        [
            "SNR bias, 5--5000\\,e$^-$ (\\%)",
            f"{worst('snr', 'snr_bias_pct'):.2f}",
            f"{tol['snr']['all']['snr_bias_pct']:g}",
        ],
        ["DR at SNR\\,=\\,1 / 10 (dB)", f"{worst('dr', 'db_abs'):.2f}", f"{tol['dr']['all']['db_abs']:g}"],
        ["CDP (absolute)", f"{worst('cdp', 'abs'):.4f}", f"{tol['cdp']['all']['abs']:g}"],
        [
            "$\\Delta E_{00}$, 2\\,\\% noise",
            f"{worst('colour', 'de00_abs', 'noisy'):.3f}",
            f"{tol['colour']['noisy']['de00_abs']:g}",
        ],
        ["$\\Delta E_{00}$, noise-free", f"{worst('colour', 'de00_abs', 'clean'):.0e}", "$10^{-6}$"],
    ]
    clean_edge = [c for c in res["edge"] if c.get("tier") == "clean" and c["sigma_px"] == 0.5]
    macros["valSharpBias"] = f"{max(abs(c['mtf50_bias_pct']) for c in clean_edge):.1f}" if clean_edge else "--"
    return table("lrr", ["Quantity (worst case)", "Error", "Limit"], rows)


# ------------------------------------------------------------------------------------- scorecard
REBOUND_LIMIT = 0.5


def scorecard_tables(sc: dict, macros: dict) -> dict[str, str]:
    rows = [r for r in sc["rows"] if not r.get("error")]
    s = sc["settings"]
    macros.update(
        nRecipes=str(len(sc["rows"])),
        nScored=str(len(rows)),
        nFailed=str(len(sc["rows"]) - len(rows)),
        nGroups=str(len(sc["groups"])),
        scRes=f"{s['xres']}$\\times${s['yres']}",
        scSpp=str(s["pixelsamples"]),
    )
    for k in (
        "deMedian",
        "deBest",
        "deWorst",
        "deBestRecipe",
        "deWorstRecipe",
        "deUnderTwo",
        "drMedLinear",
        "drMedDcg",
    ):
        macros.setdefault(k, "--")
    for name in ("mtf", "mtfSensor", "glare", "drOne", "tex"):
        macros.setdefault(f"{name}Min", "--")
        macros.setdefault(f"{name}Max", "--")
    out = {"skin": table("lrrl", ["Recipe", "Mean", "Max", "CFA"], [["--", "--", "--", "--"]])}

    by_hdr = defaultdict(list)
    for r in rows:
        by_hdr[r.get("hdr", "linear")].append(r)
    body = []
    for h, rs in sorted(by_hdr.items(), key=lambda kv: median_dr(kv[1])):
        body.append(
            [
                HDR_NAMES.get(h, h.replace("_", " ")),
                str(len(rs)),
                rng(rs, "dr_snr1_db"),
                med(rs, "dr_snr10_db"),
                med(rs, "snr_at_18pct_db"),
                med(rs, "cdp_min_level_db_at_0p9", "{:.0f}"),
            ]
        )
        macros[f"drMed{h.replace('_', '').title()}"] = med(rs, "dr_snr1_db")
    out["hdr"] = table(
        "lrrrrr", ["Architecture", "$n$", "DR$_1$ range", "DR$_{10}$", "SNR$_{18}$", "CDP$_{0.9}$"], body
    )

    by_opt = defaultdict(list)
    for r in rows:
        by_opt[r["optics"]].append(r)
    body = [
        [
            optics_label(k),
            str(len(rs)),
            med(rs, "edge_mtf50_cy_px", "{:.3f}"),
            med(rs, "edge_acutance", "{:.2f}"),
            med(rs, "texture_acutance", "{:.2f}"),
            med(rs, "veiling_glare_pct", "{:.2f}"),
        ]
        for k, rs in sorted(by_opt.items(), key=lambda kv: -len(kv[1]))
    ]
    out["optics"] = table("lrrrrr", ["Optics", "$n$", "MTF50", "Edge acut.", "Tex. acut.", "Glare \\%"], body)

    # Sharpness by CFA: output luma (after the CCM) vs the dense channel before it; rebound flags
    # non-monotonic output MTFs whose MTF50 is not comparable across recipes.
    by_cfa = defaultdict(list)
    for r in rows:
        by_cfa[r.get("cfa", "")].append(r)
    body = [
        [
            tt(k),
            str(len(rs)),
            med(rs, "edge_mtf50_cy_px", "{:.3f}"),
            med(rs, "edge_mtf_rebound", "{:.2f}"),
            med(rs, "edge_mtf50_sensor_cy_px", "{:.3f}"),
            med(rs, "edge_acutance_sensor", "{:.2f}"),
        ]
        for k, rs in sorted(by_cfa.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]
    out["cfa"] = table(
        "lrrrrr", ["CFA", "$n$", "MTF50 (out)", "Rebound", "MTF50 (sensor)", "Edge acut. (sensor)"], body
    )
    flagged = [r for r in rows if fin(r.get("edge_mtf_rebound")) and r["edge_mtf_rebound"] > REBOUND_LIMIT]
    macros["nRebound"] = str(len(flagged))
    macros["reboundLimit"] = f"{REBOUND_LIMIT:g}"

    de = sorted((r for r in rows if fin(r.get("skin_de00_mean"))), key=lambda r: r["skin_de00_mean"])
    if de:
        pick = de[:4] + de[-4:] if len(de) > 8 else de
        out["skin"] = table(
            "lrrl",
            ["Recipe", "Mean", "Max", "CFA"],
            [
                [tt(r["recipe"]), f"{r['skin_de00_mean']:.2f}", f"{r['skin_de00_max']:.2f}", tt(r.get("cfa", ""))]
                for r in pick
            ],
        )
        v = vals(rows, "skin_de00_mean")
        macros.update(
            deMedian=f"{np.median(v):.2f}",
            deBest=f"{v.min():.2f}",
            deWorst=f"{v.max():.2f}",
            deBestRecipe=tt(de[0]["recipe"]),
            deWorstRecipe=tt(de[-1]["recipe"]),
            deUnderTwo=str(int((v < 2).sum())),
        )
    for key, name, f in (
        ("edge_mtf50_cy_px", "mtf", "{:.3f}"),
        ("edge_mtf50_sensor_cy_px", "mtfSensor", "{:.3f}"),
        ("veiling_glare_pct", "glare", "{:.2f}"),
        ("dr_snr1_db", "drOne", "{:.1f}"),
        ("texture_acutance", "tex", "{:.2f}"),
    ):
        v = vals(rows, key)
        if v.size:
            macros[f"{name}Min"], macros[f"{name}Max"] = f.format(v.min()), f.format(v.max())  # fmt: skip
    return out


def figure(sc: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in sc["rows"] if not r.get("error")]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)
    hdrs = sorted(
        {r.get("hdr", "linear") for r in rows},
        key=lambda h: median_dr([r for r in rows if r.get("hdr") == h]),
    )
    for i, h in enumerate(hdrs):
        v = vals([r for r in rows if r.get("hdr") == h], "dr_snr1_db")
        ax[0].scatter(np.full(v.size, i) + np.random.default_rng(i).uniform(-0.15, 0.15, v.size), v, s=8)
    ax[0].set_xticks(range(len(hdrs)), [HDR_NAMES.get(h, h).replace(" ", "\n") for h in hdrs], fontsize=6)
    ax[0].set_ylabel("DR at SNR = 1 (dB)")
    ma = [
        (r["edge_mtf50_cy_px"], r["edge_acutance"])
        for r in rows
        if fin(r.get("edge_mtf50_cy_px")) and fin(r.get("edge_acutance"))
    ]
    m, a = np.array(ma).T if ma else (np.array([]), np.array([]))
    ax[1].scatter(m, a, s=8)
    ax[1].set_xlabel("MTF50 (cy/px)")
    ax[1].set_ylabel("Edge acutance")
    v = vals(rows, "skin_de00_mean")
    ax[2].hist(v, bins=15)
    ax[2].set_xlabel(r"Skin $\Delta E_{00}$ (mean of 12)")
    ax[2].set_ylabel("Recipes")
    for x in ax:
        x.tick_params(labelsize=6)
        x.xaxis.label.set_size(7)
        x.yaxis.label.set_size(7)
    fig.savefig(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scorecard", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    args = ap.parse_args()
    (PAPER / "tables").mkdir(exist_ok=True)
    (PAPER / "figures").mkdir(exist_ok=True)
    macros: dict[str, str] = {}
    (PAPER / "tables" / "validation.tex").write_text(validation_tables(json.loads(args.validation.read_text()), macros))
    sc = json.loads(args.scorecard.read_text())
    for name, tex in scorecard_tables(sc, macros).items():
        (PAPER / "tables" / f"{name}.tex").write_text(tex)
    figure(sc, PAPER / "figures" / "fig_scorecard.pdf")
    lines = ["%% Generated by scripts/make_tables.py -- do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (PAPER / "tables" / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
