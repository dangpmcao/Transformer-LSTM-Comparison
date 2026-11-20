"""Derive report tables and figures from saved evaluations; never run a model.

Run from the checkout: python3 report/support/build_report_evidence.py
Copy comparisons use different fixed samples across architectures and are labelled
as such. Seed contrasts do not estimate uncertainty from this sampling difference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "report/support"
FIGURES = PROJECT / "report/figures"
SEEDS = (13, 37, 73)
SIZES = ("s", "m", "l")
SOURCES: dict[str, str] = {}


def read(relative: str) -> dict:
    path = PROJECT / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def describe(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    sd = float(np.std(x, ddof=1))
    radius = float(stats.t.ppf(.975, len(x)-1) * sd / np.sqrt(len(x)))
    return {"values": x.tolist(), "n": len(x), "mean": float(x.mean()),
            "sample_sd": sd, "ci95": [float(x.mean()-radius), float(x.mean()+radius)]}


def contrast(values: list[float], margin: float | None = None) -> dict:
    result = describe(values)
    x = np.asarray(values)
    sem = stats.sem(x)
    if margin is None:
        result["p_one_sided_positive"] = float(stats.t.sf(x.mean()/sem, len(x)-1))
    else:
        result["p_tost"] = float(max(stats.t.sf((x.mean()+margin)/sem, len(x)-1),
                                      stats.t.cdf((x.mean()-margin)/sem, len(x)-1)))
        radius = float(stats.t.ppf(.95, len(x)-1)*sem)
        result["ci90"] = [float(x.mean()-radius), float(x.mean()+radius)]
        result["equivalence_margin_pp"] = margin
    return result


def holm(rows: list[dict], key: str) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, rows[index][key]*(len(rows)-rank))
        rows[index]["p_holm"] = min(1., running)


def transformer(size: str, seed: int, view: str = "best_validation", campaign: str = "mb16") -> dict:
    path = f"transformer/evaluation/runs/proposal_21/tasks/{campaign}/{size}/seed{seed}/{view}/summary.json"
    d = read(path)
    assert d["status"] == "complete" and d["metrics"]["test"]["targets"] == 291710
    m = d["metrics"]
    b = m["blimp"]
    return {"size": size, "seed": seed, "architecture": "transformer", "source": path,
            "test_nll": m["test"]["nll"], "test_ppl": m["test"]["perplexity"],
            "context_gain": m["test"]["context_gain"],
            "blimp_accuracy_pp": 100*b["primary"]["overall"]["accuracy"],
            "equal_length_accuracy_pp": 100*b["sensitivities"]["equal_bpe_length_pairs"]["overall"]["accuracy"],
            "normalized_accuracy_pp": 100*b["sensitivities"]["length_normalized_log_probability"]["overall"]["accuracy"],
            "groups_pp": {k: 100*v["accuracy"] for k,v in b["primary"]["groups"].items()},
            "copy_natural": {length: m["copy"][f"natural_{length}"]["gain"] for length in [25,50,100,200]},
            "copy_random": {length: m["copy"][f"random_{length}"]["gain"] for length in [25,50,100,200]},
            "natural_copy_100": m["copy"]["natural_100"]["gain"],
            "natural_copy_100_control_gain": m["copy"]["natural_100"]["independent_span_gain"],
            "natural_copy_100_interior_gain": m["copy"]["natural_100"]["exclude_first_four_gain"],
            "same_target": m["same_target"]["gains_against_full"],
            "parameters": d["resources"]["parameters"],
            "training_seconds": d["resources"]["training_step_seconds"],
            "training_peak_gib": d["resources"]["training_peak_allocated_bytes"]/2**30,
            "estimated_matmul_flops": d["resources"]["estimated_dense_training_matmul_flops"],
            "validation_ppl": float(np.exp(d["validation_nll"]))}


def lstm(size: str, seed: int, view: str = "best_validation") -> dict:
    base = f"lstm/results/lstm_production_{size}_seed{seed}/{view}"
    t,b,c,a = [read(f"{base}/{metric}/summary.json") for metric in ["test","blimp","copy","same_target"]]
    train = read(f"lstm/runs/production/{size}/seed{seed}/summary.json")
    check = read(f"{base}/validation_check.json")
    assert t["status"] == "PASS" and t["targets"] == 291710 and b["primary"]["overall"]["pairs"] == 67000
    return {"size": size, "seed": seed, "architecture": "lstm", "source": base,
            "test_nll": t["mean_nll"], "test_ppl": t["perplexity"],
            "context_gain": t["context"]["context_gain"],
            "blimp_accuracy_pp": 100*b["primary"]["overall"]["accuracy"],
            "equal_length_accuracy_pp": 100*b["sensitivities"]["equal_bpe_length_pairs"]["overall"]["accuracy"],
            "normalized_accuracy_pp": 100*b["sensitivities"]["length_normalized_log_probability"]["overall"]["accuracy"],
            "groups_pp": {k: 100*v["accuracy"] for k,v in b["primary"]["groups"].items()},
            "copy_natural": {length: c["results"]["test_natural_text"][str(length)]["all_positions"]["copy_gain"] for length in [25,50,100,200]},
            "copy_random": {length: c["results"]["uniform_nonspecial_token_ids"][str(length)]["all_positions"]["copy_gain"] for length in [25,50,100,200]},
            "natural_copy_100": c["natural_copy_100"],
            "natural_copy_100_control_gain": c["results"]["test_natural_text"]["100"]["all_positions"]["control_gain"],
            "natural_copy_100_interior_gain": c["results"]["test_natural_text"]["100"]["exclude_first_four"]["copy_gain"],
            "same_target": {**a["truncation_gain"], "shuffle128": a["shuffle_gain"]},
            "parameters": train["parameters"], "training_seconds": train["train_seconds"],
            "training_peak_gib": train["peak_cuda_bytes"]/2**30,
            "validation_ppl": float(np.exp(check["recomputed_nll"]))}


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    rows = [loader(size,seed) for loader in (transformer,lstm) for size in SIZES for seed in SEEDS]
    aggregates = {}
    scalar = ["test_nll","test_ppl","context_gain","blimp_accuracy_pp","equal_length_accuracy_pp",
              "normalized_accuracy_pp","natural_copy_100","natural_copy_100_control_gain",
              "natural_copy_100_interior_gain","training_seconds","training_peak_gib"]
    for arch in ("transformer","lstm"):
        aggregates[arch] = {}
        for size in SIZES:
            subset = [r for r in rows if r["architecture"] == arch and r["size"] == size]
            group = {key:describe([r[key] for r in subset]) for key in scalar}
            group["same_target"] = {key:describe([r["same_target"][key] for r in subset]) for key in ["k4","k32","k128","shuffle128"]}
            group["groups_pp"] = {key:describe([r["groups_pp"][key] for r in subset]) for key in subset[0]["groups_pp"]}
            group["copy_natural"] = {length:describe([r["copy_natural"][length] for r in subset]) for length in [25,50,100,200]}
            group["copy_random"] = {length:describe([r["copy_random"][length] for r in subset]) for length in [25,50,100,200]}
            group["parameters"] = subset[0]["parameters"]
            aggregates[arch][size] = group
    delta = {size:{key:[next(r[key] for r in rows if r["architecture"]=="transformer" and r["size"]==size and r["seed"]==seed)
                       -next(r[key] for r in rows if r["architecture"]=="lstm" and r["size"]==size and r["seed"]==seed)
                       for seed in SEEDS] for key in ["context_gain","natural_copy_100","blimp_accuracy_pp"]} for size in SIZES}
    h1 = [{"size":size,"endpoint":key,**contrast(delta[size][key])} for size in SIZES for key in ["context_gain","natural_copy_100"]]
    holm(h1,"p_one_sided_positive")
    h2 = [{"size":size,**contrast(delta[size]["blimp_accuracy_pp"],2.)} for size in SIZES]
    holm(h2,"p_tost")
    h3 = [{"endpoint":key,**contrast((np.array(delta['l'][key])-np.array(delta['s'][key])).tolist())} for key in ["context_gain","natural_copy_100"]]
    holm(h3,"p_one_sided_positive")
    # Preserve the declared families without using caveated copy-test ordering.
    # p=1 is a conservative adjustment placeholder, not an observed test result.
    nominal_tests = {"h1": [dict(r) for r in h1], "h3": [dict(r) for r in h3]}
    for family in [h1, h3]:
        for row in family:
            row["holm_input_p"] = (row["p_one_sided_positive"]
                                   if row["endpoint"] == "context_gain" else 1.)
        holm(family,"holm_input_p")
        for row in family:
            row["inference_status"] = ("tested_from_recorded_seed_summaries"
                                       if row["endpoint"] == "context_gain"
                                       else "withheld_due_to_copy_manifest_mismatch")
            if row["endpoint"] != "context_gain":
                row["p_holm"] = None
    result = {"primary_transformer_campaign":"mb16", "checkpoint_view":"best_validation", "seeds":list(SEEDS),
              "note":"Derived from saved summaries, not a new evaluation. Copy spans differ across architectures; its seed-contrast tests are descriptive and omit evaluation-sampling mismatch uncertainty. Historical cross-host array identity and LSTM raw scores cannot be reverified here.",
              "rows":rows, "aggregates":aggregates, "h1":h1, "h2_tost_pp":h2, "h3":h3,
              "multiplicity_policy":"Retain all six H1 and two H3 family slots; assign untested copy slots p=1 for Holm adjustment only. Copy p-values and intervals in nominal_tests_for_audit_only are descriptive and not used for confirmation.",
              "nominal_tests_for_audit_only":nominal_tests, "source_sha256":SOURCES}
    (OUTPUT/"comparison_results.json").write_text(json.dumps(result,indent=2)+"\n")
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    colors={"transformer":"#0072B2","lstm":"#D55E00"}
    labels={"transformer":"Transformer","lstm":"LSTM"}
    size_labels={"s":"Small","m":"Medium","l":"Large"}
    fig,axs=plt.subplots(1,3,figsize=(8.3,2.35),layout="constrained")
    for ax,key,title,ylabel in zip(axs,["test_ppl","context_gain","blimp_accuracy_pp"],["Text prediction","Position-based context gain","Grammar judgments"],["Test perplexity","Gain (nats per token)","BLiMP accuracy (%)"]):
        for arch in colors:
            groups=[aggregates[arch][s][key] for s in SIZES]
            xs=np.arange(3)+(-.045 if arch=="transformer" else .045)
            ax.errorbar(xs,[g["mean"] for g in groups],yerr=[g["sample_sd"] for g in groups],color=colors[arch],marker="o",markersize=4,capsize=3,label=labels[arch])
            for x,g in zip(xs,groups):ax.scatter(np.full(3,x)+np.array([-.025,0,.025]),g["values"],s=9,color=colors[arch],alpha=.5)
        ax.set(xticks=range(3),xticklabels=[size_labels[s] for s in SIZES],title=title,ylabel=ylabel)
    axs[0].legend(frameon=False,fontsize=8)
    fig.savefig(FIGURES/"architecture_comparison.pdf");fig.savefig(FIGURES/"architecture_comparison.png",dpi=200);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8.3,2.3),layout="constrained")
    for arch in colors:
        for size,marker in zip(SIZES,["o","s","^"]):
            g=aggregates[arch][size]["same_target"]
            axs[0].plot(range(3),[g[k]["mean"] for k in ["k4","k32","k128"]],color=colors[arch],marker=marker,label=f"{labels[arch]} {size_labels[size].lower()}",markersize=4,alpha=.9)
        gs=[aggregates[arch][size]["same_target"]["shuffle128"] for size in SIZES]
        axs[1].errorbar(np.arange(3)+(-.035 if arch=="transformer" else .035),[g["mean"] for g in gs],yerr=[g["sample_sd"] for g in gs],color=colors[arch],marker="o",capsize=3,label=labels[arch])
    axs[0].set(xticks=range(3),xticklabels=["4","32","128"],xlabel="Number of preceding tokens retained",ylabel="Loss increase\n(nats per token)",title="Shortened context")
    axs[0].legend(frameon=False,fontsize=7,ncol=2)
    axs[1].axhline(0,color="gray",lw=.8,ls=":")
    axs[1].set(xticks=range(3),xticklabels=[size_labels[s] for s in SIZES],ylabel="Loss increase\n(nats per token)",title="Distant context shuffled")
    axs[1].legend(frameon=False,fontsize=8)
    fig.savefig(FIGURES/"context_intervention.pdf");fig.savefig(FIGURES/"context_intervention.png",dpi=200);plt.close(fig)
    for size in SIZES:
        print(size,{arch:{k:round(aggregates[arch][size][k]["mean"],4) for k in ["test_ppl","blimp_accuracy_pp","context_gain","natural_copy_100"]} for arch in colors})
    print('H1',[(r['size'],r['endpoint'],round(r['mean'],5),round(r['p_holm'],6) if r['p_holm'] is not None else None) for r in h1])
    print('H2',[(r['size'],round(r['mean'],5),round(r['p_holm'],6),r['ci90']) for r in h2])
    print('H3',[(r['endpoint'],round(r['mean'],5),round(r['p_holm'],6) if r['p_holm'] is not None else None) for r in h3])


if __name__ == "__main__":
    main()
