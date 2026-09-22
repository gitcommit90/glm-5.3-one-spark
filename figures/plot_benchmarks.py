#!/usr/bin/env python3
"""Regenerate the publication-quality benchmark summary from repository measurements."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "benchmark-summary"
PALETTE = {
    "blue": "#0F4D92", "blue2": "#3775BA", "green": "#8BCF8B",
    "green2": "#AADCA9", "red": "#B64342", "pink": "#E9A6A1",
    "gray": "#767676", "light": "#CFCECE", "dark": "#272727",
}
plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12,
    "axes.linewidth": 1.8, "axes.spines.right": False,
    "axes.spines.top": False, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

def label_bars(ax, bars, fmt="{:.1f}"):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h), (bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=10, fontweight="bold")

def finish(fig):
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

# Headline values are the measured tables in README; raw benchmark evidence lives under benchmarks/raw/ and docs/.
k=[4,5,6,7,8]; structured=[48.6,53.5,59.2,63.9,66.9]; prose=[28.8,29.9,29.0,25.8,23.9]; code=[37.7,40.1,37.5,38.3,38.0]
prompt=[8002,16002,100002]; prefill=[786.4,822.1,845.7]; cold=[10.175,19.464,118.250]; warm=[1.775,2.786,8.842]
fig,axes=plt.subplots(1,3,figsize=(16,4.8)); fig.suptitle("GLM-5.3 Flash EXL3 — measured on one DGX Spark",fontsize=18,fontweight="bold")
for vals,label,color,mark in [(structured,"Structured",PALETTE["blue"],"o"),(prose,"Prose",PALETTE["green"],"s"),(code,"Code",PALETTE["red"],"^")]: axes[0].plot(k,vals,label=label,color=color,marker=mark,lw=2.5,ms=7)
axes[0].axvline(5,color=PALETTE["gray"],ls="--",lw=1.5); axes[0].text(5.05,68,"default K=5",fontsize=9,color=PALETTE["gray"])
axes[0].set(title="DFlash2 speculative-depth sweep",xlabel="Draft depth K",ylabel="decode tokens/s",xticks=k); axes[0].legend(fontsize=9); axes[0].grid(axis="y",alpha=.18)
x=np.arange(3); bars=axes[1].bar(x,prefill,color=[PALETTE["light"],PALETTE["green"],PALETTE["blue"]],edgecolor="black",linewidth=1.2); label_bars(axes[1],bars,"{:.1f}")
axes[1].set(title="Cold prefill throughput",xlabel="Prompt tokens",ylabel="tokens/s",xticks=x,xticklabels=["8K","16K","100K"]); axes[1].grid(axis="y",alpha=.18)
w=.35; b1=axes[2].bar(x-w/2,cold,w,label="Cold",color=PALETTE["pink"],edgecolor="black"); b2=axes[2].bar(x+w/2,warm,w,label="Warm cached",color=PALETTE["blue"],edgecolor="black"); label_bars(axes[2],b1,"{:.1f}"); label_bars(axes[2],b2,"{:.1f}")
axes[2].set(title="Time to first token",xlabel="Prompt tokens",ylabel="seconds",xticks=x,xticklabels=["8K","16K","100K"]); axes[2].legend(fontsize=9); axes[2].grid(axis="y",alpha=.18)
fig.text(.5,.005,"Source: README measured tables; raw evidence: benchmarks/raw and docs/_cold_prefill_raw.json",ha="center",color=PALETTE["gray"],fontsize=9)
finish(fig)
