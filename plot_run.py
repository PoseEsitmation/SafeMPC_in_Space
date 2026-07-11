#!/usr/bin/env python3
"""
Visualise a SafeStuff CBF-CLF training run.

Usage:
    python plot_run.py <run_dir>                  # interactive overview
    python plot_run.py <run_dir> --save           # save overview + per-chart PNGs
    python plot_run.py <run_dir> --task 1 --save
"""
import argparse
import glob
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ── colour palette ─────────────────────────────────────────────────────────────
C_SAFE = "#2ecc71"
C_DANGER = "#e74c3c"
C_FILTER = "#f39c12"
C_IMIT = "#3498db"
C_CBF = "#9b59b6"
C_CLF = "#1abc9c"
C_NN = "#e67e22"
C_EVAL_EXPERT = "#27ae60"
C_VAL = "#8e44ad"
BG_RAND = "#fff0f0"
BG_TRAIN = "#f0fff4"

plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           9,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.3,
    "grid.linewidth":      0.5,
    "legend.framealpha":   0.85,
    "legend.fontsize":     8,
})


# ── helpers ────────────────────────────────────────────────────────────────────

def load(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    evts = ea.Scalars(tag)
    return (np.array([e.step for e in evts], dtype=float),
            np.array([e.value for e in evts], dtype=float))


def downsample(steps, vals, n=2000):
    if len(steps) <= n:
        return steps, vals
    idx = np.round(np.linspace(0, len(steps) - 1, n)).astype(int)
    return steps[idx], vals[idx]


def smooth(vals, w=50):
    if len(vals) < 2 * w:
        return vals
    k = np.exp(-0.5 * (np.arange(-w, w + 1) / (w / 2)) ** 2)
    k /= k.sum()
    return np.convolve(vals, k, mode="same")


def _savefig(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  saved → {path}")


# ── per-panel draw functions ───────────────────────────────────────────────────

def draw_safety_timeline(ax_rand, ax_train, d):
    """Broken-axis safety timeline: random phase (left) | training phase (right)."""
    rand_steps = d["rand_steps"]
    rand_margin = d["rand_margin"]
    tr_steps = d["tr_steps"]
    tr_margin = d["tr_margin"]
    sf_steps3 = d["sf_steps3"]
    sf_active = d["sf_active"]

    # --- random phase ---
    ax_rand.set_facecolor(BG_RAND)
    if len(rand_steps) > 0:
        ax_rand.fill_between(rand_steps, rand_margin, 0,
                             where=(rand_margin < 0),
                             color=C_DANGER, alpha=0.35, label="KOZ violation")
        ax_rand.plot(rand_steps, rand_margin,
                     color=C_DANGER, lw=0.6, alpha=0.7)
        ax_rand.axhline(0, color=C_DANGER, lw=1.2, ls="--", alpha=0.8,
                        label="KOZ boundary (0°)")
        n_viol = int((rand_margin < 0).sum())
        ax_rand.text(0.97, 0.96, f"{n_viol} violations\n(no filter)",
                     transform=ax_rand.transAxes, ha="right", va="top", fontsize=8,
                     color=C_DANGER, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax_rand.set_ylabel("θ margin (deg)")
    ax_rand.set_xlabel("Step (random phase)")
    ax_rand.set_title("Random Exploration  (no safety filter)",
                      fontsize=9, color=C_DANGER, fontweight="bold")
    ax_rand.legend(loc="lower left", fontsize=7)
    ax_rand.spines["right"].set_visible(False)

    # --- training phase ---
    ax_train.set_facecolor(BG_TRAIN)
    if len(tr_steps) > 0:
        ds_s, ds_m = downsample(tr_steps, tr_margin)
        ax_train.plot(ds_s, ds_m, color=C_SAFE, lw=0.7,
                      alpha=0.8, label="θ margin")
        ax_train.plot(ds_s, smooth(ds_m, 80), color=C_SAFE, lw=1.8, alpha=0.95,
                      label="smoothed")
    if len(sf_steps3) > 0:
        active_s, _ = downsample(sf_steps3[sf_active > 0.5],
                                 sf_steps3[sf_active > 0.5], n=3000)
        rug_y = tr_margin.max() * 0.95 if len(tr_margin) > 0 else 1.0
        ax_train.plot(active_s, np.full_like(active_s, rug_y),
                      "|", color=C_FILTER, alpha=0.25, ms=4, label="filter active")
    ax_train.axhline(0, color=C_DANGER, lw=1.2, ls="--", alpha=0.8)
    ax_train.set_xlabel("Step (training)")
    ax_train.set_title("Training  (safety filter ON)",
                       fontsize=9, color=C_SAFE, fontweight="bold")
    ax_train.tick_params(labelleft=False)
    ax_train.spines["left"].set_visible(False)
    ax_train.legend(loc="lower right", fontsize=7)

    # broken-axis slash decorators
    dv = 0.012
    for ax, side in [(ax_rand, "right"), (ax_train, "left")]:
        x = 1 if side == "right" else 0
        kw = dict(transform=ax.transAxes, color="grey", clip_on=False, lw=1.5)
        ax.plot((x - dv, x + dv), (-dv, +dv), **kw)
        ax.plot((x - dv, x + dv), (1 - dv, 1 + dv), **kw)


def draw_episode_stats(ax, d):
    tr_ep_s = d["tr_ep_s"]
    tr_ep_margin = d["tr_ep_margin"]
    tr_ep_ff_s = d["tr_ep_ff_s"]
    tr_ep_ff = d["tr_ep_ff"]
    tr_ep_filt_s = d["tr_ep_filt_s"]
    tr_ep_filt = d["tr_ep_filt"]

    if len(tr_ep_s) > 0:
        ax.bar(range(len(tr_ep_margin)), tr_ep_margin,
               color=C_SAFE, alpha=0.75, label="min θ margin")
        ax.axhline(0, color=C_DANGER, lw=1, ls="--")

    ax2 = ax.twinx()
    if len(tr_ep_ff_s) > 0:
        ax2.plot(range(len(tr_ep_ff)), tr_ep_ff * 100,
                 "o-", color=C_FILTER, lw=1.5, ms=4, label="filter fraction (%)")
    elif len(tr_ep_filt_s) > 0:
        ep_lens = d.get("ep_lens", np.array([]))
        if len(ep_lens) > 0:
            n = min(len(tr_ep_filt), len(ep_lens))
            frac = tr_ep_filt[:n] / np.maximum(ep_lens[:n], 1)
            ax2.plot(range(n), frac * 100,
                     "o-", color=C_FILTER, lw=1.5, ms=4, label="filter fraction (%)")

    ax2.set_ylabel("Filter fraction (%)", color=C_FILTER, fontsize=8)
    ax2.tick_params(axis="y", labelcolor=C_FILTER)
    ax2.spines["top"].set_visible(False)

    ax.set_title("Per-Episode Safety Statistics", fontsize=9)
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Min θ margin (deg)", color=C_SAFE, fontsize=8)
    ax.tick_params(axis="y", labelcolor=C_SAFE)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="upper right")


def draw_filter_activation(ax, d):
    """Filter activation breakdown — safety of the environment at a glance.

    Stacked bars: share of env steps per bin, split by what the QP filter did.
        green      inactive   proposed action already satisfied the CBF
        amber      corrected  QP projected the action; hard CBF held → safe
        red        fallback   hard CBF infeasible; least-unsafe action only
        dark red   failed     QP error; unfiltered action reached the env
    Fallback + failed are the steps where safety was NOT guaranteed.
    Overlay (right axis): % of steps actually inside the KOZ per bin.

    Falls back to the binary `filter_active` tag (inactive/corrected only) for
    runs that predate the per-step `filter_type` scalar.
    """
    steps, types = d["sf_steps6"], d["sf_type"]
    legacy = len(steps) == 0
    if legacy:
        steps, types = d["sf_steps3"], d["sf_active"]
    if len(steps) == 0:
        ax.text(0.5, 0.5, "no safety-filter data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="grey")
        ax.set_title("Filter Activation Breakdown", fontsize=9)
        return
    types = np.round(types).astype(int)

    n_bins = int(min(60, max(10, len(steps) // 200)))
    edges = np.linspace(steps.min(), steps.max() + 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = np.diff(edges) * 0.92
    bin_idx = np.clip(np.digitize(steps, edges) - 1, 0, n_bins - 1)

    CLASSES = [
        (0, "inactive (action already safe)", C_SAFE),
        (1, "corrected (QP projection, safe)", C_FILTER),
        (2, "fallback (CBF infeasible!)",      C_DANGER),
        (3, "QP failed (unfiltered!)",         "#4a0d0d"),
    ]
    if legacy:
        # Old runs only logged binary filter_active — fallback/failed steps are
        # folded into "corrected" and cannot be separated retroactively.
        CLASSES = [
            (0, "inactive (action already safe)", C_SAFE),
            (1, "active (type breakdown n/a — legacy run)", C_FILTER),
        ]
    counts = np.zeros((len(CLASSES), n_bins))
    for c, _, _ in CLASSES:
        np.add.at(counts[c], bin_idx[types == c], 1)
    tot = np.maximum(counts.sum(axis=0), 1)
    overall = counts.sum(axis=1) / max(counts.sum(), 1) * 100

    bottom = np.zeros(n_bins)
    for c, label, color in CLASSES:
        ax.bar(centers, counts[c] / tot * 100, width=width, bottom=bottom,
               color=color, alpha=0.85, linewidth=0,
               label=f"{label} — {overall[c]:.1f}%")
        bottom += counts[c] / tot * 100

    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of steps (%)")
    ax.set_xlabel("Env step")
    ax.set_title("Filter Activation Breakdown — how safety was maintained",
                 fontsize=9, fontweight="bold")

    # KOZ violation rate per bin (ground truth of "how safe the env was")
    ax2 = ax.twinx()
    if len(d["tr_steps"]) > 0:
        vidx = np.clip(np.digitize(d["tr_steps"], edges) - 1, 0, n_bins - 1)
        vsum = np.zeros(n_bins)
        vcnt = np.zeros(n_bins)
        np.add.at(vsum, vidx, (d["tr_margin"] < 0).astype(float))
        np.add.at(vcnt, vidx, 1)
        vrate = vsum / np.maximum(vcnt, 1) * 100
        ax2.plot(centers, vrate, "o-", color="#2c3e50", lw=1.6, ms=3.5,
                 label=f"KOZ violation rate — {vsum.sum()/max(vcnt.sum(),1)*100:.2f}% overall")
        ax2.set_ylim(bottom=0)
    ax2.set_ylabel("KOZ violation rate (%)", fontsize=8, color="#2c3e50")
    ax2.tick_params(axis="y", labelcolor="#2c3e50", labelsize=7)
    ax2.spines["top"].set_visible(False)

    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="center left",
              framealpha=0.9)


def draw_cbf_barrier(ax, d):
    sf_steps = d["sf_steps"]
    sf_H = d["sf_H"]
    sf_steps2 = d["sf_steps2"]
    sf_slack = d["sf_slack"]

    if len(sf_steps) > 0:
        ds_s, ds_H = downsample(sf_steps, sf_H)
        ax.plot(ds_s, ds_H, color=C_SAFE, lw=0.7, alpha=0.7)
        ax.plot(ds_s, smooth(ds_H, 100), color=C_SAFE, lw=2,
                label="H(x)  (extended barrier)")
    if len(sf_steps2) > 0:
        ds_s2, ds_sl = downsample(sf_steps2, sf_slack)
        ax.fill_between(ds_s2, ds_sl, alpha=0.3, color=C_DANGER,
                        label="CBF slack (>0 = violation)")
    ax.axhline(0, color=C_DANGER, lw=1, ls="--", alpha=0.7, label="H = 0")
    ax.set_title("CBF Barrier Value H(x)", fontsize=9)
    ax.set_xlabel("Env step")
    ax.set_ylabel("H(x)")
    ax.legend(fontsize=7)


def draw_policy_losses(ax, d):
    pol_s = d["pol_s"]
    pol_imit = d["pol_imit"]
    pol_s2 = d["pol_s2"]
    pol_cbf = d["pol_cbf"]
    pol_s3 = d["pol_s3"]
    pol_clf = d["pol_clf"]
    pol_s4 = d["pol_s4"]
    pol_cbf_vf = d["pol_cbf_vf"]
    pol_s6 = d["pol_s6"]
    pol_clf_vf = d["pol_clf_vf"]

    if len(pol_s) > 0:
        ax.semilogy(pol_s, pol_imit, color=C_IMIT, lw=1.5, label="L_imit")
    if len(pol_s2) > 0 and pol_cbf.max() > 0:
        ax.semilogy(pol_s2, np.clip(pol_cbf, 1e-9, None), color=C_CBF, lw=1.5,
                    label="L_cbf")
    if len(pol_s3) > 0 and pol_clf.max() > 0:
        ax.semilogy(pol_s3, np.clip(pol_clf, 1e-9, None), color=C_CLF, lw=1.5,
                    label="L_clf")

    ax2 = ax.twinx()
    if len(pol_s4) > 0:
        ax2.plot(pol_s4, pol_cbf_vf * 100, color=C_CBF, lw=1, ls=":", alpha=0.8,
                 label="CBF viol % (batch)")
        ax2.plot(pol_s6, pol_clf_vf * 100, color=C_CLF, lw=1, ls=":", alpha=0.8,
                 label="CLF viol % (batch)")
    ax2.set_ylabel("Batch violation (%)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.spines["top"].set_visible(False)

    ax.set_title("Policy Training Losses (log scale)", fontsize=9)
    ax.set_xlabel("Policy gradient step")
    ax.set_ylabel("Loss")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7)


def draw_rewards(ax, d):
    """Expert reward & KOZ violations.  NN-policy eval traces intentionally
    left out — the NN policy has its own dedicated charts (post-DAGGER
    validation)."""
    tr_ep_rew_s = d["tr_ep_rew_s"]
    tr_ep_rew = d["tr_ep_rew"]
    ev_rew_s = d["ev_rew_s"]
    ev_rew = d["ev_rew"]
    ev_koz_s = d["ev_koz_s"]
    ev_koz = d["ev_koz"]

    if len(tr_ep_rew_s) > 0:
        ax.plot(tr_ep_rew_s, tr_ep_rew, "o-", color=C_SAFE,
                lw=1.5, ms=4, label="Train env (expert+filter)")
    if len(ev_rew_s) > 0:
        ax.plot(ev_rew_s, ev_rew, "s-", color=C_EVAL_EXPERT,
                lw=1.5, ms=4, label="Eval env (hnet MPC)")

    ax2 = ax.twinx()
    plotted_koz = False
    if len(ev_koz_s) > 0:
        ax2.plot(ev_koz_s, ev_koz, "s--", color=C_EVAL_EXPERT,
                 lw=1, ms=4, alpha=0.6, label="Eval KOZ violations")
        plotted_koz = True
    if plotted_koz:
        ax2.set_ylabel("KOZ violations / episode", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.spines["top"].set_visible(False)

    ax.set_title("Reward & KOZ Violations", fontsize=9)
    ax.set_xlabel("Env step / Policy step")
    ax.set_ylabel("Episode reward")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7)


def draw_dagger_curriculum(ax, d):
    pol_dag_s = d["pol_dag_s"]
    pol_kappa = d["pol_kappa"]
    pol_dag_lcbf_s = d["pol_dag_lcbf_s"]
    pol_lcbf = d["pol_lcbf"]
    pol_dag_lclf_s = d["pol_dag_lclf_s"]
    pol_lclf = d["pol_lclf"]

    if len(pol_dag_s) > 0:
        ax.plot(pol_dag_s, pol_kappa, "o-", color="#2980b9",
                lw=1.5, ms=5, label="κ (expert mix)")
    ax2 = ax.twinx()
    if len(pol_dag_lcbf_s) > 0:
        ax2.plot(pol_dag_lcbf_s, pol_lcbf, "^-", color=C_CBF,
                 lw=1.5, ms=5, label="λ_cbf")
    if len(pol_dag_lclf_s) > 0:
        ax2.plot(pol_dag_lclf_s, pol_lclf, "v-", color=C_CLF,
                 lw=1.5, ms=5, label="λ_clf")
    ax2.set_ylabel("λ (loss weights)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.spines["top"].set_visible(False)

    ax.set_title("DAgger Curriculum", fontsize=9)
    ax.set_xlabel("Policy step")
    ax.set_ylabel("κ (expert mixing ratio)")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7)


def draw_dagger_validation(ax, d):
    """Post-DAGGER validation: does the *raw learned policy* stay safe once the
    QP filter is removed?  Answers whether DAGGER is teaching real safety or
    the filter is just papering over an unsafe policy.

    Filtered and unfiltered eval are run back-to-back after every DAGGER
    iteration on the same (dedicated, non-training) env and logged at the
    same policy step, so both series and `dag_iter` line up index-for-index.
    """
    u_koz = d["dv_u_koz"]
    f_koz = d["dv_f_koz"]
    u_margin = d["dv_u_margin"]

    n = len(u_koz)
    if n == 0:
        ax.text(0.5, 0.5, "no dagger_eval_unfiltered data yet",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="grey")
        ax.set_title("Post-DAGGER Validation (filter on/off)", fontsize=9)
        return

    it = np.arange(1, n + 1)
    width = 0.28
    ax.bar(it - width, u_koz, width, color=C_DANGER, alpha=0.85,
           label="unfiltered koz violations")
    if len(f_koz) == n:
        ax.bar(it, f_koz, width, color=C_SAFE, alpha=0.85,
               label="filtered koz violations")
    # KOZ hits during the unfiltered DAGGER rollouts themselves (baseline_22+):
    # nonzero early means the buffer is receiving near-KOZ avoidance labels;
    # this and the unfiltered validation bars should fall together.
    dag_rkoz = d.get("dag_rkoz", [])
    if len(dag_rkoz) == n:
        ax.bar(it + width, dag_rkoz, width, color="#e67e22", alpha=0.7,
               label="rollout koz (label source)")
    ax.set_xticks(it)
    ax.set_ylabel("KOZ violations / episode (mean)")

    ax2 = ax.twinx()
    if len(u_margin) == n:
        ax2.plot(it, u_margin, "D-", color="#8e44ad", lw=1.5, ms=5,
                  label="unfiltered min θ margin (deg)")
        ax2.axhline(0, color=C_DANGER, lw=1, ls="--", alpha=0.6)
    ax2.set_ylabel("Min θ margin (deg)", color="#8e44ad", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#8e44ad", labelsize=7)
    ax2.spines["top"].set_visible(False)

    ax.set_title("Post-DAGGER Validation — policy safety with filter OFF vs ON",
                 fontsize=9)
    ax.set_xlabel("DAgger iteration")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="upper right")


def draw_filter_reliance(ax, d):
    """Headline DAGGER result: the NN policy needs the safety filter less and
    less.  Plots the filter's intervention rate on the *policy's own* actions
    (dagger_eval_filtered/filter_fraction) per DAGGER iteration, with the
    unfiltered KOZ violations as context — both should fall together as the
    policy internalises the avoidance behaviour."""
    frac = np.asarray(d.get("dv_f_filtfrac", []), dtype=float)
    fb   = np.asarray(d.get("dv_f_fbfrac",   []), dtype=float)
    du   = np.asarray(d.get("dv_f_du",       []), dtype=float)
    ukoz = np.asarray(d.get("dv_u_koz",      []), dtype=float)

    n = len(frac)
    if n == 0:
        ax.text(0.5, 0.5, "no dagger_eval_filtered data yet",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="grey")
        ax.set_title("Filter Reliance Across DAGGER", fontsize=9)
        return

    it = np.arange(1, n + 1)
    ax.plot(it, frac * 100, "o-", color=C_DANGER, lw=2.2, ms=6,
            label="filter intervention rate (% of steps)", zorder=3)
    if len(fb) == n and np.any(fb > 0):
        ax.plot(it, fb * 100, "s--", color="#c0392b", lw=1.3, ms=4,
                label="fallback rate (CBF infeasible, %)")
    ax.axhline(0, color=C_SAFE, lw=1.2, ls="--", alpha=0.8)
    ax.set_ylim(bottom=-0.3)
    ax.set_xticks(it)
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("Filter usage on NN policy (% of steps)", color=C_DANGER)
    ax.tick_params(axis="y", labelcolor=C_DANGER)

    ax2 = ax.twinx()
    if len(ukoz) == n:
        ax2.bar(it, ukoz, 0.45, color="#95a5a6", alpha=0.45, zorder=1,
                label="unfiltered koz violations / ep")
    ax2.set_ylabel("Unfiltered KOZ violations", color="#7f8c8d", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#7f8c8d", labelsize=7)
    ax2.spines["top"].set_visible(False)

    ax.set_title("Filter Reliance Across DAGGER — policy needs the filter less",
                 fontsize=9, fontweight="bold")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="upper right")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)


def draw_dynamics_val(ax, d):
    val_s = d["val_s"]
    val_loss = d["val_loss"]

    if len(val_s) > 0:
        ax.semilogy(val_s, val_loss, color=C_VAL,
                    lw=1.8, label="Dynamics val loss")
    ax.set_title("Dynamics Model Validation Loss", fontsize=9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (log)")
    ax.legend(fontsize=7)


def draw_phase_comparison(fig, d, hp):
    """
    Full training timeline on a single continuous x-axis.

    Combined x  = step within phase:
        [0,          rand_steps)           → random phase
        [rand_steps, rand_steps+max_iter)  → MPC loop (training)

    Phase bands:
        0            .. rand_steps                 : random  (no filter)
        rand_steps   .. rand_steps+pol_start        : MPC-only (filter, no NN)
        rand_steps+pol_start .. rand_steps+dag_every: BC supervised
        rand_steps+dag_every .. (repeat every dag_every) : DAgger iterations
    """
    rand_steps = hp.get("init_rand_steps",    3000)
    pol_start = hp.get("policy_train_start", 3000)
    dag_every = hp.get("dagger_every",        5000)
    dag_n = hp.get("dagger_n_iter",          5)
    dyn_every = hp.get("dynamics_update_every", 1000)

    # ── stitch data onto combined x-axis ─────────────────────────────────────
    # already 0..rand_steps-1
    cx_rand = d["rand_steps"]
    cy_rand = d["rand_margin"]

    cx_tr = d["tr_steps"] + rand_steps             # shift training steps right
    cy_tr = d["tr_margin"]

    cx_filt = d["sf_steps3"][d["sf_active"] > 0.5] + \
        rand_steps   # filter-active steps

    # episode-level: use stored env_iter (already on env_iter axis)
    cx_ep_margin = d["tr_ep_s"] + rand_steps
    cy_ep_margin = d["tr_ep_margin"]

    # DAgger event steps (from eval_env which runs at same cadence as DAgger)
    # Use eval_env step markers as proxies
    dag_env_steps = d["ev_rew_s"] + rand_steps   # in combined coords

    total_x = rand_steps + (cx_tr.max() if len(cx_tr) > 0 else 1)

    # ── phase boundary x-coordinates (combined) ──────────────────────────────
    b_rand_end = rand_steps
    b_bc_start = rand_steps + pol_start
    b_dag_starts = [rand_steps + (i + 1) * dag_every for i in range(dag_n)]
    # clip to actual data length
    b_dag_starts = [x for x in b_dag_starts if x < total_x]

    # ── colour bands ──────────────────────────────────────────────────────────
    BANDS = [
        (0,          b_rand_end,  "#ffecec", "Random\n(no filter)"),
        (b_rand_end, b_bc_start,  "#fff8e8", "MPC + filter\n(no NN)"),
        (b_bc_start, b_dag_starts[0] if b_dag_starts else total_x,
         "#eaf4ff", "BC supervised"),
    ]
    dag_colours = ["#edfff0", "#d6f5da", "#b8ecc0", "#9de3a8", "#7dd990"]
    for i, bx in enumerate(b_dag_starts):
        end = b_dag_starts[i + 1] if i + 1 < len(b_dag_starts) else total_x
        BANDS.append((bx, end, dag_colours[i % len(dag_colours)],
                      f"DAgger\niter {i+1}"))

    # ── create subplots: θ-margin timeline + reward strip ────────────────────
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=fig.add_gridspec(1, 1)[0],
        height_ratios=[3, 1], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # ── top: theta_margin ─────────────────────────────────────────────────────
    for (x0, x1, bg, label) in BANDS:
        ax_top.axvspan(x0, x1, color=bg, alpha=1.0, zorder=0)
        mid = (x0 + x1) / 2
        ax_top.text(mid, ax_top.get_ylim()[1] if len(cy_rand) == 0 else
                    max(cy_rand.max() if len(cy_rand) > 0 else 0,
                        cy_tr.max() if len(cy_tr) > 0 else 0) * 1.05,
                    label, ha="center", va="top", fontsize=7,
                    color="#444", style="italic", clip_on=True)

    # KOZ boundary
    ax_top.axhline(0, color=C_DANGER, lw=1.4, ls="--", alpha=0.9, zorder=2,
                   label="KOZ boundary (0°)")

    # random phase: line + fill violations
    if len(cx_rand) > 0:
        ax_top.plot(cx_rand, cy_rand, color=C_DANGER,
                    lw=0.7, alpha=0.8, zorder=3)
        ax_top.fill_between(cx_rand, cy_rand, 0, where=(cy_rand < 0),
                            color=C_DANGER, alpha=0.45, zorder=4,
                            label=f"Violation ({int((cy_rand < 0).sum())} steps)")
        # red scatter dots at violation points
        vmask = cy_rand < 0
        ax_top.scatter(cx_rand[vmask], np.zeros(vmask.sum()),
                       color=C_DANGER, s=6, zorder=5, alpha=0.6)

    # training phase: thin raw + thick smooth
    if len(cx_tr) > 0:
        ds_s, ds_m = downsample(cx_tr, cy_tr)
        ax_top.plot(ds_s, ds_m, color=C_SAFE, lw=0.6, alpha=0.6, zorder=3)
        ax_top.plot(ds_s, smooth(ds_m, 80), color=C_SAFE, lw=2.0, alpha=0.95,
                    zorder=4, label="θ margin (training)")

    # filter activation rug — small ticks at top
    if len(cx_filt) > 0:
        ds_filt, _ = downsample(cx_filt, cx_filt, n=4000)
        y_top = (max(cy_rand.max() if len(cy_rand) > 0 else 0,
                     cy_tr.max() if len(cy_tr) > 0 else 0)) * 0.97
        ax_top.plot(ds_filt, np.full_like(ds_filt, y_top),
                    "|", color=C_FILTER, alpha=0.2, ms=4, zorder=3,
                    label="Filter correction")

    # DAgger iteration vertical lines
    for i, bx in enumerate(b_dag_starts):
        ax_top.axvline(bx, color="#2c7a4b", lw=1.5,
                       ls=":", alpha=0.9, zorder=5)
        ax_top.text(bx + total_x * 0.004, ax_top.get_ylim()[0] if False else -15,
                    f"DAgger {i+1}", fontsize=7, color="#2c7a4b",
                    rotation=90, va="bottom")

    # dynamics model training markers (every dyn_every within training phase)
    dyn_x = [rand_steps + k * dyn_every
             for k in range(1, int(total_x / dyn_every) + 1)
             if rand_steps + k * dyn_every < total_x]
    for dx in dyn_x:
        ax_top.axvline(dx, color="#95a5a6", lw=0.8,
                       ls="--", alpha=0.4, zorder=2)

    # phase boundary heavy line
    # ax_top.axvline(b_rand_end, color="#555", lw=2, alpha=0.7, zorder=6,               label="Filter ON")
    ax_top.axvline(b_bc_start, color="#2980b9", lw=1.5, ls="-.", alpha=0.7, zorder=6,
                   label="NN training starts")

    ax_top.set_ylabel("θ margin to KOZ (deg)", fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(loc="upper right", fontsize=7, ncol=2)
    ax_top.set_title(
        "Full Training Timeline — Phase Comparison  (safety filter effectiveness)",
        fontsize=11, fontweight="bold")
    ax_top.set_xlim(0, total_x)

    # ── bottom: eval rewards over the same timeline ──────────────────────────
    # Expert eval reward (env-step axis → combined coords).
    if len(d["ev_rew_s"]) > 0:
        ax_bot.plot(d["ev_rew_s"] + rand_steps, d["ev_rew"], "s-",
                    color=C_EVAL_EXPERT, lw=1.5, ms=4,
                    label="Eval env (hnet MPC)")
    # Post-DAgger NN-policy eval rewards — one point per DAgger iteration,
    # pinned to the DAgger event positions on the combined axis.
    dag_xs = b_dag_starts[:len(d["dv_f_rew"])]
    if len(dag_xs) > 0:
        ax_bot.plot(dag_xs, d["dv_f_rew"][:len(dag_xs)], "^-",
                    color=C_NN, lw=1.3, ms=5,
                    label="NN policy (filter ON)")
    dag_xs_u = b_dag_starts[:len(d["dv_u_rew"])]
    if len(dag_xs_u) > 0:
        ax_bot.plot(dag_xs_u, d["dv_u_rew"][:len(dag_xs_u)], "v--",
                    color=C_NN, lw=1.1, ms=5, alpha=0.55,
                    label="NN policy (filter OFF)")
    ax_bot.axhline(0, color="#888", lw=0.8, ls=":", alpha=0.7)

    # repeat phase backgrounds + DAgger markers on the reward strip
    for (x0, x1, bg, _) in BANDS:
        ax_bot.axvspan(x0, x1, color=bg, alpha=1.0, zorder=0)
    for bx in b_dag_starts:
        ax_bot.axvline(bx, color="#2c7a4b", lw=1.2, ls=":", alpha=0.7, zorder=5)

    ax_bot.set_ylabel("Episode reward", fontsize=8)
    ax_bot.tick_params(labelsize=7)
    ax_bot.legend(fontsize=7, loc="lower right", ncol=3)
    ax_bot.set_xlim(0, total_x)
    ax_bot.set_xlabel("Combined training step  (random phase → MPC + filter → DAgger)",
                      fontsize=9)

    # ── x-axis tick labels: show actual phase names ───────────────────────────
    tick_xs = [0, b_rand_end, b_bc_start] + b_dag_starts
    tick_lbs = (
        ["0",
         f"{b_rand_end}\n(filter ON)",
         f"{b_rand_end + pol_start}\n(NN training)"] +
        [f"{int(bx)}\n(DAgger {i+1})" for i, bx in enumerate(b_dag_starts)]
    )
    ax_bot.set_xticks(tick_xs)
    ax_bot.set_xticklabels(tick_lbs, fontsize=7)


def draw_filter_du(ax, d):
    sf_steps4 = d["sf_steps4"]
    sf_du = d["sf_du"]
    sf_steps5 = d["sf_steps5"]
    sf_V = d["sf_V"]

    if len(sf_steps4) > 0:
        ds_s, ds_du = downsample(sf_steps4, sf_du)
        ax.plot(ds_s, ds_du, color=C_FILTER, lw=0.7, alpha=0.6)
        ax.plot(ds_s, smooth(ds_du, 100), color=C_FILTER, lw=1.8,
                label="‖u_safe − u_MPC‖  (correction magnitude)")
    ax2 = ax.twinx()
    if len(sf_steps5) > 0:
        ds_s5, ds_V = downsample(sf_steps5, sf_V)
        ax2.plot(ds_s5, ds_V, color=C_CLF, lw=0.7, alpha=0.5)
        ax2.plot(ds_s5, smooth(ds_V, 100), color=C_CLF,
                 lw=1.5, label="V(x)  (CLF value)")
    ax2.set_ylabel("V(x)", fontsize=8, color=C_CLF)
    ax2.tick_params(axis="y", labelcolor=C_CLF, labelsize=7)
    ax2.spines["top"].set_visible(False)

    ax.set_title("Safety Filter Correction Magnitude & CLF Value", fontsize=9)
    ax.set_xlabel("Env step")
    ax.set_ylabel("‖Δu‖")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7)


# ── overview figure ────────────────────────────────────────────────────────────

def plot_overview(d, run_name, footer):
    fig = plt.figure(figsize=(18, 19))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"CBF-CLF Safety Filter Training  |  {run_name}",
                 fontsize=13, fontweight="bold", y=0.995)

    outer = gridspec.GridSpec(6, 1, figure=fig,
                              height_ratios=[1.8, 1.1, 1, 1, 1, 1], hspace=0.50)

    # Row 0 — safety timeline (broken x-axis)
    n_rand = max(len(d["rand_steps"]), 1)
    n_tr = max(len(d["tr_steps"]),   1)
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[n_rand, n_tr], wspace=0.05)
    ax_rand = fig.add_subplot(top_gs[0])
    ax_train = fig.add_subplot(top_gs[1], sharey=ax_rand)
    draw_safety_timeline(ax_rand, ax_train, d)

    # Row 1 — filter activation breakdown (headline safety chart), full width
    draw_filter_activation(fig.add_subplot(outer[1]), d)

    # Row 2 — episode stats | CBF barrier
    mid_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2], wspace=0.35)
    draw_episode_stats(fig.add_subplot(mid_gs[0]), d)
    draw_cbf_barrier(fig.add_subplot(mid_gs[1]), d)

    # Row 3 — policy losses | rewards
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[3], wspace=0.35)
    draw_policy_losses(fig.add_subplot(bot_gs[0]), d)
    draw_rewards(fig.add_subplot(bot_gs[1]), d)

    # Row 4 — DAgger curriculum | filter correction + val loss
    dag_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[4], wspace=0.38)
    draw_dagger_curriculum(fig.add_subplot(dag_gs[0]), d)
    draw_filter_du(fig.add_subplot(dag_gs[1]), d)
    draw_dynamics_val(fig.add_subplot(dag_gs[2]), d)

    # Row 5 — post-DAGGER validation | filter reliance (headline result)
    val_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[5], wspace=0.40)
    draw_dagger_validation(fig.add_subplot(val_gs[0]), d)
    draw_filter_reliance(fig.add_subplot(val_gs[1]), d)

    fig.text(0.5, 0.002, footer, ha="center",
             va="bottom", fontsize=9, color="grey")
    return fig


# ── individual chart saves ─────────────────────────────────────────────────────

CHARTS = [
    # (filename_stem, draw_fn,  figsize,    needs_broken_axis, needs_phase)
    ("01_safety_timeline",   None,                   (14, 5),  True,  False),
    ("02_episode_stats",     draw_episode_stats,     (7, 5),   False, False),
    ("03_cbf_barrier",       draw_cbf_barrier,       (7, 5),   False, False),
    ("04_policy_losses",     draw_policy_losses,     (7, 5),   False, False),
    ("05_rewards",           draw_rewards,           (7, 5),   False, False),
    ("06_dagger_curriculum", draw_dagger_curriculum, (7, 5),   False, False),
    ("07_filter_correction", draw_filter_du,         (7, 5),   False, False),
    ("08_dynamics_val",      draw_dynamics_val,      (7, 5),   False, False),
    ("09_phase_comparison",  None,                   (16, 8),  False, True),
    ("10_dagger_validation", draw_dagger_validation, (10, 5),  False, False),
    ("11_filter_activation", draw_filter_activation, (12, 5),  False, False),
    ("12_filter_reliance",   draw_filter_reliance,   (10, 5),  False, False),
]


def save_individual(d, run_name, out_dir, dpi, hp=None):
    os.makedirs(out_dir, exist_ok=True)
    if hp is None:
        hp = {}

    for stem, draw_fn, figsize, broken, phase in CHARTS:
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("white")

        if phase:
            draw_phase_comparison(fig, d, hp)
            fig.suptitle(f"Phase Comparison  |  {run_name}",
                         fontsize=10, fontweight="bold", y=1.01)
        elif broken:
            ax_rand = fig.add_subplot(1, 2, 1)
            ax_train = fig.add_subplot(1, 2, 2)
            ax_train.sharey(ax_rand)
            draw_safety_timeline(ax_rand, ax_train, d)
            fig.suptitle(f"Safety Timeline  |  {run_name}",
                         fontsize=10, fontweight="bold")
        else:
            ax = fig.add_subplot(1, 1, 1)
            draw_fn(ax, d)
            fig.suptitle(f"{run_name}", fontsize=9, color="grey", y=1.01)

        try:
            fig.tight_layout()
        except Exception:
            pass
        path = os.path.join(out_dir, f"{stem}.png")
        _savefig(fig, path, dpi)
        plt.close(fig)


# ── data loading ───────────────────────────────────────────────────────────────

def load_hparams(run_dir):
    """Parse hparams.csv into a plain dict with numeric values where possible."""
    hp = {}
    csv_path = os.path.join(run_dir, "hparams.csv")
    if not os.path.exists(csv_path):
        return hp
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("config"):
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            key, val = parts[0].strip(), parts[1].strip()
            try:
                hp[key] = int(val)
            except ValueError:
                try:
                    hp[key] = float(val)
                except ValueError:
                    hp[key] = val
    return hp


def load_all(ea, T):
    d = {}

    d["rand_steps"],  d["rand_margin"] = load(
        ea, f"random_phase/{T}/theta_margin_deg")
    d["rand_att_s"],  d["rand_att"] = load(
        ea, f"random_phase/{T}/attitude_error_deg")

    d["tr_steps"],  d["tr_margin"] = load(
        ea, f"train_env/{T}/theta_margin_deg")
    d["tr_ep_s"],   d["tr_ep_margin"] = load(
        ea, f"train_env/{T}/min_theta_margin_deg")
    d["tr_ep_filt_s"], d["tr_ep_filt"] = load(
        ea, f"train_env/{T}/filter_activations")
    d["tr_ep_ff_s"],   d["tr_ep_ff"] = load(
        ea, f"train_env/{T}/filter_fraction")
    d["tr_ep_koz_s"],  d["tr_ep_koz"] = load(
        ea, f"train_env/{T}/koz_violations")
    d["tr_ep_rew_s"],  d["tr_ep_rew"] = load(ea, f"train_env/{T}/reward")
    d["tr_ep_att_s"],  d["tr_ep_att"] = load(
        ea, f"train_env/{T}/att_err_final_deg")

    _, ep_lens = load(ea, f"train_env/{T}/episode_length")
    d["ep_lens"] = ep_lens

    d["sf_steps"],  d["sf_H"] = load(ea, f"safety/{T}/cbf_H")
    d["sf_steps2"], d["sf_slack"] = load(ea, f"safety/{T}/cbf_slack")
    d["sf_steps3"], d["sf_active"] = load(ea, f"safety/{T}/filter_active")
    d["sf_steps4"], d["sf_du"] = load(ea, f"safety/{T}/filter_du_norm")
    d["sf_steps5"], d["sf_V"] = load(ea, f"safety/{T}/clf_V")
    # Per-step activation type (SafetyFilter.TYPE_*): 0 inactive, 1 corrected,
    # 2 soft-CBF fallback, 3 QP failure.  Missing in runs before baseline_21.
    d["sf_steps6"], d["sf_type"] = load(ea, f"safety/{T}/filter_type")
    d["tr_ep_corr_s"], d["tr_ep_corr"] = load(ea, f"train_env/{T}/filter_corrected")
    d["tr_ep_fb_s"],   d["tr_ep_fb"] = load(ea, f"train_env/{T}/filter_fallback")
    d["tr_ep_fail_s"], d["tr_ep_fail"] = load(ea, f"train_env/{T}/filter_failed")

    d["pol_s"],   d["pol_imit"] = load(ea, "policy/loss_imit")
    d["pol_s2"],  d["pol_cbf"] = load(ea, "policy/loss_cbf")
    d["pol_s3"],  d["pol_clf"] = load(ea, "policy/loss_clf")
    d["pol_s4"],  d["pol_cbf_vf"] = load(ea, "policy/cbf_viol_frac")
    d["pol_s5"],  d["pol_cbf_mm"] = load(ea, "policy/cbf_mean_margin")
    d["pol_s6"],  d["pol_clf_vf"] = load(ea, "policy/clf_viol_frac")
    # Curriculum scalars moved from policy/* to dagger/* in baseline_21;
    # fall back to the old tags so dashboards for older runs still render.
    d["pol_dag_s"],      d["pol_kappa"] = load(ea, "dagger/kappa")
    if len(d["pol_dag_s"]) == 0:
        d["pol_dag_s"],  d["pol_kappa"] = load(ea, "policy/kappa")
    d["pol_dag_lcbf_s"], d["pol_lcbf"] = load(ea, "dagger/lambda_cbf")
    if len(d["pol_dag_lcbf_s"]) == 0:
        d["pol_dag_lcbf_s"], d["pol_lcbf"] = load(ea, "policy/lambda_cbf")
    d["pol_dag_lclf_s"], d["pol_lclf"] = load(ea, "dagger/lambda_clf")
    if len(d["pol_dag_lclf_s"]) == 0:
        d["pol_dag_lclf_s"], d["pol_lclf"] = load(ea, "policy/lambda_clf")
    # KOZ hits during the unfiltered DAGGER rollouts (baseline_22+): nonzero
    # early = the buffer is receiving near-KOZ avoidance labels; → 0 = learned.
    d["dag_rkoz_s"], d["dag_rkoz"] = load(ea, "dagger/rollout_koz")

    d["ev_rew_s"],  d["ev_rew"] = load(ea, f"eval_env/{T}/reward")
    d["pev_rew_s"], d["pev_rew"] = load(ea, f"policy_eval/{T}/reward")
    d["pev_koz_s"], d["pev_koz"] = load(ea, f"policy_eval/{T}/koz_violations")
    d["ev_koz_s"],  d["ev_koz"] = load(ea, f"eval_env/{T}/koz_violations")
    d["ev_att_s"],  d["ev_att"] = load(ea, f"eval_env/{T}/att_err_final_deg")

    d["val_s"],  d["val_loss"] = load(ea, f"val/{T}/loss")

    # Post-DAGGER validation: same NN policy, with vs. without the safety
    # filter, logged at the same policy step so the two line up 1:1 with
    # dagger/iter (see hnet_exp._eval_nn_policy call sites after dagger_update).
    d["dv_f_s"],   d["dv_f_koz"] = load(ea, f"dagger_eval_filtered/{T}/koz_violations")
    _,             d["dv_f_rew"] = load(ea, f"dagger_eval_filtered/{T}/reward")
    _,             d["dv_f_margin"] = load(ea, f"dagger_eval_filtered/{T}/min_theta_margin_deg")
    # Filter reliance of the NN policy (headline DAGGER result): fraction of
    # eval steps where the QP had to correct the action, and how hard.
    _,             d["dv_f_filtfrac"] = load(ea, f"dagger_eval_filtered/{T}/filter_fraction")
    _,             d["dv_f_fbfrac"]   = load(ea, f"dagger_eval_filtered/{T}/filter_fallback_frac")
    _,             d["dv_f_du"]       = load(ea, f"dagger_eval_filtered/{T}/filter_du_mean")

    d["dv_u_s"],   d["dv_u_koz"] = load(ea, f"dagger_eval_unfiltered/{T}/koz_violations")
    _,             d["dv_u_rew"] = load(ea, f"dagger_eval_unfiltered/{T}/reward")
    _,             d["dv_u_margin"] = load(ea, f"dagger_eval_unfiltered/{T}/min_theta_margin_deg")

    d["dag_iter_s"], d["dag_iter"] = load(ea, "dagger/iter")

    return d


def build_footer(d):
    n_rand = int((d["rand_margin"] < 0).sum()) if len(
        d["rand_margin"]) > 0 else "?"
    n_train = int(d["tr_ep_koz"].sum()) if len(d["tr_ep_koz"]) > 0 else "?"
    if len(d["tr_ep_ff"]) > 0:
        mf = f"{d['tr_ep_ff'].mean()*100:.1f}%"
    elif len(d["tr_ep_filt"]) > 0 and len(d["ep_lens"]) > 0:
        n = min(len(d["tr_ep_filt"]), len(d["ep_lens"]))
        mf = f"{(d['tr_ep_filt'][:n] / np.maximum(d['ep_lens'][:n], 1)).mean()*100:.1f}%"
    else:
        mf = "?"
    if len(d["dv_u_koz"]) > 0:
        dv = (f"  |  Post-DAGGER unfiltered violations: "
              f"{d['dv_u_koz'].sum():.0f} total over {len(d['dv_u_koz'])} checks "
              f"(worst min margin {d['dv_u_margin'].min():.2f}deg)")
    else:
        dv = ""
    if len(d["sf_type"]) > 0:
        t = np.round(d["sf_type"]).astype(int)
        ft = (f"  |  Filter fallback: {(t == 2).sum()} steps, "
              f"QP failures: {(t == 3).sum()} steps")
    else:
        ft = ""
    return (f"Random phase: {n_rand} KOZ violations  |  "
            f"Training: {n_train} KOZ violations (filter active)  |  "
            f"Avg filter fraction: {mf}{ft}{dv}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--save",  action="store_true",
                    help="save overview + per-chart PNGs")
    ap.add_argument("--task",  type=int, default=0)
    ap.add_argument("--dpi",   type=int, default=150)
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    run_name = os.path.basename(run_dir)

    pattern = os.path.join(run_dir, "events.out.tfevents.*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        sys.exit(f"No TFEvents file found in {run_dir}")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    print(f"Loading {matches[-1]} …")
    ea = EventAccumulator(matches[-1])
    ea.Reload()

    T = f"task_{args.task}"
    d = load_all(ea, T)
    hp = load_hparams(run_dir)
    footer = build_footer(d)

    if args.save:
        # overview
        fig = plot_overview(d, run_name, footer)
        overview_path = os.path.join(run_dir, "training_overview.png")
        _savefig(fig, overview_path, args.dpi)
        plt.close(fig)

        # individual charts
        charts_dir = os.path.join(run_dir, "charts")
        print(f"\nSaving individual charts to {charts_dir}/")
        save_individual(d, run_name, charts_dir, args.dpi, hp=hp)
        print("Done.")
    else:
        fig = plot_overview(d, run_name, footer)
        plt.tight_layout(rect=[0, 0.01, 1, 0.99])
        plt.show()


if __name__ == "__main__":
    main()
