"""The report's charts: matplotlib figures written out as inline SVG text.

Each function takes the numbers :func:`blinded.report.recompute` stored and
returns one ``<svg>`` element, so the page stays a single file with no
JavaScript. Text stays text (``svg.fonttype: none``) and ids are salted with a
fixed string, so the same numbers give the same page.
"""

from __future__ import annotations

import io
import re
from contextlib import contextmanager

import numpy as np

SEGMENT_COLOR = {"train": "#8c8c8c", "val": "#2a5d9f", "test": "#c0562c"}
SEGMENT_NAME = {"train": "train", "val": "validation", "test": "test"}
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e4e4"

# The look of a journal figure: a closed frame, ticks inward on all four sides with minor
# ticks between, no grid, black ink, thin lines and framed legends.
_RC = {
    "svg.fonttype": "none",
    "svg.hashsalt": "blinded",
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "normal",
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.edgecolor": INK,
    "legend.fontsize": 7.5,
    "patch.linewidth": 0.6,
    "hatch.linewidth": 0.6,
    "lines.linewidth": 1.2,
}


@contextmanager
def _figure(ncols: int = 1, width: float = 7.4, height: float = 2.9, sharey: bool = False):
    import matplotlib
    from matplotlib.figure import Figure

    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(width, height), layout="constrained")
        axes = fig.subplots(1, ncols, sharey=sharey, squeeze=False)[0]
        yield fig, list(axes)


def _svg(fig) -> str:
    import matplotlib

    buf = io.StringIO()
    with matplotlib.rc_context(_RC):
        fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None})
    text = buf.getvalue()
    text = re.sub(r"\s*<metadata>.*?</metadata>", "", text, flags=re.S)
    return text[text.index("<svg") :]


def _seconds(label: str) -> float:
    import pandas as pd

    return pd.Timedelta(label).total_seconds()


TICKS = ("100ms", "500ms", "1s", "5s", "10s", "30s", "60s", "1min", "5min")


def _horizon_axis(ax, labels: list[str]) -> None:
    shown = [h for h in labels if h in TICKS] or labels
    ax.set_xscale("log")
    ax.set_xticks([_seconds(h) for h in shown], shown)
    ax.minorticks_off()


def _whole_numbers(ax, n: int) -> None:
    from matplotlib.ticker import MaxNLocator, NullLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.5, n + 0.5)


def _num(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def ladder(points: list[dict]) -> str:
    """Every scored feature's validation value in order, released ones filled, best as a step."""
    with _figure() as (fig, (ax,)):
        n = np.arange(1, len(points) + 1)
        v = _num([p["value"] for p in points])
        released = np.array([bool(p["released"]) for p in points])
        best = _num([p["best_after"] for p in points])
        ax.step(n, best, where="post", color=SEGMENT_COLOR["val"], label="released best")
        ax.scatter(n[~released], v[~released], s=14, color="#b9b9b9", zorder=3,
                   label="not released")  # fmt: skip
        ax.scatter(n[released], v[released], s=22, color=SEGMENT_COLOR["val"], zorder=4,
                   label="released")  # fmt: skip
        _whole_numbers(ax, len(points))
        ax.set_xlabel("feature submission, in order")
        ax.set_ylabel("validation rank IC")
        ax.set_title("The Ladder: each feature against the released best")
        ax.legend(loc="lower right")
        return _svg(fig)


def attempts(points: list[dict], ylabel: str, title: str) -> str:
    """Each strategy's score in submission order, and the best so far as a step."""
    with _figure(height=2.7) as (fig, (ax,)):
        n = np.arange(1, len(points) + 1)
        v = _num([p["value"] for p in points])
        released = np.array([bool(p.get("released")) for p in points])
        best = np.fmax.accumulate(np.where(np.isfinite(v), v, -np.inf))
        best[~np.isfinite(best)] = np.nan
        flat = ~np.isfinite(v)
        ax.step(n, best, where="post", color="#c0562c", label="best so far")
        ax.scatter(n[~released & ~flat], v[~released & ~flat], s=16, color="#b9b9b9", zorder=3,
                   label="each strategy")  # fmt: skip
        if flat.any():
            ax.scatter(n[flat], np.zeros(flat.sum()), s=16, facecolors="none",
                       edgecolors="#9a9a9a", zorder=3, label="never traded")  # fmt: skip
        if released.any():
            ax.scatter(n[released], v[released], s=24, color=SEGMENT_COLOR["val"], zorder=4,
                       label="released")  # fmt: skip
        ax.axhline(0, color="#999999", lw=0.8)
        _whole_numbers(ax, len(points))
        ax.set_xlabel("strategy submission, in order")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        return _svg(fig)


def train_vs_val(points: list[dict]) -> str:
    """Train against validation rank IC for every accepted feature, with the line y = x."""
    with _figure(width=4.6, height=3.4) as (fig, (ax,)):
        x = _num([p["train"] for p in points])
        y = _num([p["val"] for p in points])
        released = np.array([bool(p["released"]) for p in points])
        ok = np.isfinite(x) & np.isfinite(y)
        lo = float(np.nanmin(np.r_[x[ok], y[ok], 0.0])) if ok.any() else 0.0
        hi = float(np.nanmax(np.r_[x[ok], y[ok], 0.0])) if ok.any() else 1.0
        pad = (hi - lo) * 0.06 or 0.01
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=0.8, ls=":",
                label="validation = train")  # fmt: skip
        ax.scatter(x[ok & ~released], y[ok & ~released], s=16, color="#b9b9b9",
                   label="not released")  # fmt: skip
        ax.scatter(x[ok & released], y[ok & released], s=24, color=SEGMENT_COLOR["val"],
                   label="released")  # fmt: skip
        ax.set_xlabel("train rank IC")
        ax.set_ylabel("validation rank IC")
        ax.set_title("Every accepted feature, train against validation")
        ax.legend(loc="upper left")
        return _svg(fig)


FAN_LONG, FAN_SHORT, BREAKEVEN = "#3a8d5b", "#8e5bb0", "#c62828"


def event_fan(fan_by_segment: dict, offsets: list[str], fill_ms: float, title: str, ylabel: str,
              xlabel: str, unit: str = "events", levels: dict | None = None,
              points: dict | None = None, x_unit_s: float = 1.0) -> str:  # fmt: skip
    """The markout fan: the mean move in the traded direction with two standard errors either side.

    One panel per segment, stacked on one time axis in seconds. The band is
    the mean plus and minus two standard errors, so it widens as single paths
    drift apart; longs and shorts are drawn apart; the dotted line is the
    fill. ``levels`` maps a segment to ``[(label, bps, linestyle), ...]``,
    the break-evens, and ``points`` to ``[(label, seconds, bps)]``, drawn as
    diamonds. A level more than three times the fan's height is named in the
    legend instead of drawn, so it cannot flatten the fan. ``x_unit_s`` is
    the axis unit in seconds, 3600 for hours on hourly bars.
    """
    import matplotlib
    from matplotlib.figure import Figure

    segs = list(fan_by_segment)
    levels, points = levels or {}, points or {}
    xs = np.array([_seconds(o) for o in offsets])
    fill_x = fill_ms / 1000
    if x_unit_s != 1.0:
        xs, fill_x = xs / x_unit_s, fill_x / x_unit_s
        points = {seg: [(n, x / x_unit_s, y) for n, x, y in pts] for seg, pts in points.items()}
    top = 0.0
    for f in fan_by_segment.values():
        mean, se = _num(f["mean_bps"]), _num(f["se_bps"])
        for y in (mean - 2 * se, mean + 2 * se, _num(f["long_bps"]), _num(f["short_bps"])):
            if np.isfinite(y).any():
                top = max(top, float(np.nanmax(np.abs(y))))
    top = top or 1.0
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(7.4, 0.5 + 2.4 * len(segs)), layout="constrained")
        axes = fig.subplots(len(segs), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
        for ax, seg in zip(axes, segs, strict=True):
            f = fan_by_segment[seg]
            color = SEGMENT_COLOR[seg]
            mean, se = _num(f["mean_bps"]), _num(f["se_bps"])
            ax.fill_between(xs, mean - 2 * se, mean + 2 * se, color=color, alpha=0.22, lw=0,
                            label="2 standard errors")  # fmt: skip
            ax.plot(xs, mean, color=color, lw=1.6, label=f"all ({f['n']:,})")
            ax.plot(xs, _num(f["long_bps"]), color=FAN_LONG, lw=0.8,
                    label=f"longs ({f['n_long']:,})")  # fmt: skip
            ax.plot(xs, _num(f["short_bps"]), color=FAN_SHORT, lw=0.8,
                    label=f"shorts ({f['n_short']:,})")  # fmt: skip
            ax.axhline(0, color=INK, lw=0.6)
            ax.axvline(fill_x, color=MUTED, lw=1.0, ls=":", label="fill")
            for name, x, y in points.get(seg, []):
                if y is not None:
                    ax.scatter([x], [y], marker="D", s=24, color=INK, zorder=5, label=name)
            for name, y, style in levels.get(seg, []):
                if y is None:
                    continue
                if abs(y) <= 3 * top:
                    ax.axhline(y, color=BREAKEVEN, lw=1.0, ls=style, label=name)
                else:
                    ax.plot([], [], color=BREAKEVEN, lw=1.0, ls=style,
                            label=f"{name}: {y:.1f} bps,\noff the scale")  # fmt: skip
            ax.set_title(f"{SEGMENT_NAME[seg]}: {f['n']:,} {unit}", color=color, loc="left")
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        axes[-1].set_xlabel(xlabel)
        axes[0].set_xlim(float(xs.min()), float(xs.max()))
        fig.supylabel(ylabel, fontsize=8.5, color=MUTED)
        fig.suptitle(title, fontsize=9, fontweight="normal")
        return _svg(fig)


def decay(decay_by_segment: dict, horizons: list[str], name: str) -> str:
    """Rank IC against horizon, one line per segment."""
    with _figure(width=4.6, height=3.0) as (fig, (ax,)):
        xs = [_seconds(h) for h in horizons]
        for seg, values in decay_by_segment.items():
            ax.plot(xs, _num(values), color=SEGMENT_COLOR[seg], marker="o", ms=3,
                    label=SEGMENT_NAME[seg])  # fmt: skip
        ax.axhline(0, color="#999999", lw=0.8)
        _horizon_axis(ax, horizons)
        ax.set_ylabel("rank IC")
        ax.set_title(f"Decay of {name}")
        ax.legend()
        return _svg(fig)


def windows(rows: list[dict], name: str, every: str) -> str:
    """Rank IC in each clock window, coloured by segment."""
    import pandas as pd

    with _figure(height=2.6) as (fig, (ax,)):
        n = np.arange(len(rows))
        v = _num([r["rank_ic"] for r in rows])
        colors = [SEGMENT_COLOR[r["segment"]] for r in rows]
        ax.bar(n, np.nan_to_num(v), width=0.8, color=colors)
        starts = [pd.Timestamp(r["start"]) for r in rows]
        ticks = [i for i, t in enumerate(starts) if t.minute == 0] or list(n)
        ticks = ticks[:: max(1, -(-len(ticks) // 12))]
        ax.set_xticks(ticks, [starts[i].strftime("%H:%M") for i in ticks])
        ax.axhline(0, color="#999999", lw=0.8)
        seen = []
        for r in rows:
            if r["segment"] not in seen:
                seen.append(r["segment"])
        test = [i for i, r in enumerate(rows) if r["segment"] == "test"]
        for seg in seen:
            if seg == "test":
                continue
            ax.bar([np.nan], [np.nan], color=SEGMENT_COLOR[seg], label=SEGMENT_NAME[seg])
        if test:
            # the locked test hatched behind its bars, as a journal figure marks a window
            ax.axvspan(test[0] - 0.5, test[-1] + 0.5, facecolor="none", hatch="////",
                       edgecolor=SEGMENT_COLOR["test"], alpha=0.45, lw=0, zorder=0,
                       label="locked test")  # fmt: skip
        ax.set_ylabel("rank IC")
        ax.set_title(f"Rank IC of {name} in each {every} window")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.2 * (hi - lo))  # room for the legend above the bars
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: labels[i] == "locked test")
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right",
                  ncols=len(seen))  # fmt: skip
        return _svg(fig)


def pnl(pnl_by_segment: dict, name: str) -> str:
    """Cumulative P&L before spread, after spread, and after spread and fees, per segment."""
    segs = list(pnl_by_segment)
    with _figure(ncols=len(segs), height=2.9, sharey=True) as (fig, axes):
        for ax, seg in zip(axes, segs, strict=True):
            p = pnl_by_segment[seg]
            h = p["hours"]
            ax.plot(h, p["mid"], color=INK, lw=0.9, ls=":", label="mid move only")
            ax.plot(h, p["after_spread"], color="#6c9bd2", lw=1.2, label="after the spread")
            ax.plot(h, p["net"], color="#c0562c", label="after spread and fees")
            ax.axhline(0, color="#999999", lw=0.8)
            ax.set_title(f"{SEGMENT_NAME[seg]}: {p['fills']:,} fills", color=SEGMENT_COLOR[seg])
            ax.set_xlabel("hours into the window")
        axes[0].set_ylabel("cumulative P&L, bps")
        axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(f"Where the P&L of {name} goes", fontsize=9, fontweight="normal")
        return _svg(fig)


def basket_pnl(pnl_by_segment: dict, name: str) -> str:
    """Cumulative net P&L of each leg of a basket as thin lines, and their sum bold, per
    segment."""
    segs = list(pnl_by_segment)
    with _figure(ncols=len(segs), height=2.9, sharey=True) as (fig, axes):
        for ax, seg in zip(axes, segs, strict=True):
            p = pnl_by_segment[seg]
            h = p["hours"]
            for i, leg in enumerate(p["legs"].values()):
                ax.plot(h, leg, color="#9aa7b8", lw=0.5, label="one leg" if i == 0 else None)
            ax.plot(h, p["net"], color=SEGMENT_COLOR[seg], lw=1.8, label="the basket")
            ax.axhline(0, color="#999999", lw=0.8)
            ax.set_title(f"{SEGMENT_NAME[seg]}: {p['fills']:,} fills", color=SEGMENT_COLOR[seg])
            ax.set_xlabel("hours into the window")
        axes[0].set_ylabel("cumulative net P&L, bps")
        axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(f"Net P&L of {name}, leg by leg", fontsize=9, fontweight="normal")
        return _svg(fig)


def funding(d7_by_segment: dict, name: str) -> str:
    """D7's two lines per segment: the funding the perpetual leg collected (positive) or paid,
    and what holding the position made on the marks, the basis change, both cumulative."""
    segs = list(d7_by_segment)
    with _figure(ncols=len(segs), height=2.9, sharey=True) as (fig, axes):
        for ax, seg in zip(axes, segs, strict=True):
            lines = d7_by_segment[seg]["lines"]
            h = lines["hours"]
            ax.plot(h, lines["funding_bps"], color=SEGMENT_COLOR[seg], lw=1.6,
                    label="funding received")  # fmt: skip
            ax.plot(h, lines["mid_bps"], color=INK, lw=0.9, ls=":", label="basis change (mid)")
            ax.axhline(0, color="#999999", lw=0.8)
            n = d7_by_segment[seg]["settlements"]
            ax.set_title(f"{SEGMENT_NAME[seg]}: {n:,} settlements", color=SEGMENT_COLOR[seg])
            ax.set_xlabel("hours into the window")
        axes[0].set_ylabel("cumulative, bps")
        axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(f"Funding against the basis: {name}", fontsize=9, fontweight="normal")
        return _svg(fig)


def locked_test_pnl(curves: dict, hours: float) -> str:
    """Cumulative net P&L on test: the brief's variant solid, the unrealistic ones dashed."""
    from blinded.markouts import downsample

    with _figure(height=2.9) as (fig, (ax,)):
        styles = {"brief": ("-", SEGMENT_COLOR["test"]), "latency 0": ("--", "#d99a7c"),
                  "latency 0, fees 0": (":", "#8c8c8c")}  # fmt: skip
        for name, (ls, color) in styles.items():
            curve = curves.get(name)
            if not curve:
                continue
            keep = downsample(len(curve), 1200)
            t = np.linspace(0, hours, len(curve))[keep]
            label = name if name == "brief" else f"{name} (unrealistic)"
            ax.plot(t, np.asarray(curve)[keep], ls=ls, color=color, label=label)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xlabel("hours into the test window")
        ax.set_ylabel("cumulative net P&L, bps")
        ax.set_title("The locked test")
        ax.legend(loc="lower left")
        return _svg(fig)


LATENCY_COLOR = "#f2c94c"
LONG_COLOR, SHORT_COLOR = "#2166ac", "#b2182b"


def lead_lag(item: dict) -> str:
    """The cross-correlation of the leader's last step with the follower's later steps.

    One line per segment against the lag in seconds. Zero is the decision;
    the shaded strip is the time to the fill, latency plus feed delay, where a
    follower move ends before a taker's order lands; the dotted line is the
    follower's first whole step after the fill.
    """
    lags = np.asarray(item["lags_ms"], dtype=float) / 1000
    fill, step = item["fill_ms"] / 1000, item["step_ms"] / 1000
    with _figure(height=3.0) as (fig, (ax,)):
        for seg, row in item["segments"].items():
            ax.plot(lags, _num(row["corr"]), color=SEGMENT_COLOR[seg], marker="o", ms=2.4,
                    lw=1.2, label=SEGMENT_NAME[seg])  # fmt: skip
        ax.axhline(0, color="#999999", lw=0.8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.axvspan(0, fill, color=LATENCY_COLOR, alpha=0.35, lw=0, label="before the fill")
        ax.axvline(fill + step, color=INK, lw=0.9, ls=":",
                   label="first whole step\nafter the fill")  # fmt: skip
        ax.set_xlim(lags.min() - step / 4, lags.max() + step / 4)
        for x, text, align in ((0.01, "follower first", "left"), (0.99, "leader first", "right")):
            ax.text(x, 0.97, text, transform=ax.transAxes, ha=align, va="top", color=MUTED,
                    fontsize=7.5)  # fmt: skip
        ax.set_xlabel("lag: the follower's step ends this long after the leader's, s")
        ax.set_ylabel("correlation of one-step returns")
        ax.set_title(f"Lead-lag: {item['leader']} then {item['follower']}")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        return _svg(fig)


def pair_spread(item: dict) -> str:
    """The spread less its walk-forward mean, against the bands it traded on, per segment.

    Under each panel, the position held in the spread: the share of each
    plotted interval spent long it less the share spent short.
    """
    import matplotlib
    from matplotlib.figure import Figure

    segs = list(item["spread"])
    z_in, z_out = float(item["params"]["z_in"]), float(item["params"]["z_out"])
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(7.4, 3.4), layout="constrained")
        grid = fig.subplots(2, len(segs), sharex="col", sharey="row", squeeze=False,
                            gridspec_kw={"height_ratios": [4, 1]})  # fmt: skip
        top = 0.0
        for col, seg in enumerate(segs):
            ax, held = grid[0][col], grid[1][col]
            sp = item["spread"][seg]
            h, dev, sd = np.asarray(sp["hours"]), _num(sp["dev_bps"]), _num(sp["sd_bps"])
            ax.plot(h, dev, color=INK, lw=0.6, label="spread less its mean")
            for k, ls, text in ((z_in, "--", f"+-{z_in:g} sd: in"),
                                (z_out, ":", f"+-{z_out:g} sd: out")):  # fmt: skip
                ax.plot(h, k * sd, color=SEGMENT_COLOR[seg], lw=0.9, ls=ls, label=text)
                ax.plot(h, -k * sd, color=SEGMENT_COLOR[seg], lw=0.9, ls=ls)
            finite = np.r_[dev[np.isfinite(dev)], z_in * sd[np.isfinite(sd)]]
            top = max(top, float(np.max(np.abs(finite))) if len(finite) else 0.0)
            ax.axhline(0, color="#999999", lw=0.8)
            ax.set_title(SEGMENT_NAME[seg], color=SEGMENT_COLOR[seg])
            pos = _num(sp["position"])
            held.fill_between(h, 0, pos, where=pos >= 0, color=LONG_COLOR, alpha=0.5, lw=0,
                              step="post", label="long the spread")  # fmt: skip
            held.fill_between(h, 0, pos, where=pos <= 0, color=SHORT_COLOR, alpha=0.5, lw=0,
                              step="post", label="short the spread")  # fmt: skip
            held.set_ylim(-1.05, 1.05)
            held.set_yticks([-1, 0, 1])
            held.set_xlabel("hours into the window")
        lim = top * 1.1 or 1.0
        grid[0][0].set_ylim(-lim, lim)
        grid[0][0].set_ylabel("bps")
        grid[1][0].set_ylabel("held")
        grid[0][-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        grid[1][-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        a, b = item["pair"]
        fig.suptitle(f"The spread of {a} over {b}, and its bands", fontsize=9,
                     fontweight="normal")  # fmt: skip
        return _svg(fig)


def _window_axis(ax, starts, most: int = 12) -> None:
    """Ticks on clock windows in order: at whole hours, at most ``most`` of them."""
    import pandas as pd

    times = [pd.Timestamp(s) for s in starts]
    ticks = [i for i, t in enumerate(times) if t.minute == 0] or list(range(len(times)))
    ticks = ticks[:: max(1, -(-len(ticks) // most))]
    days = len({t.date() for t in times}) > 1
    fmt = "%d %H:%M" if days else "%H:%M"
    ax.set_xticks(ticks, [times[i].strftime(fmt) for i in ticks])


def hedge_ratio(item: dict) -> str:
    """The hedge ratio held through each clock window, one step per window."""
    rows = item["windows"]
    with _figure(height=2.6) as (fig, (ax,)):
        n = np.arange(len(rows))
        beta = _num([r["beta"] for r in rows])
        for seg in dict.fromkeys(r["segment"] for r in rows):
            mask = np.array([r["segment"] == seg for r in rows])
            # the last window is held to its end, one window on
            x, y = np.r_[n[mask], n[mask][-1] + 1], np.r_[beta[mask], beta[mask][-1]]
            ax.step(x, y, where="post", color=SEGMENT_COLOR[seg], label=SEGMENT_NAME[seg])
        _window_axis(ax, [r["start"] for r in rows])
        a, b = item["pair"]
        ax.set_ylabel("hedge ratio")
        ax.set_title(f"Hedge ratio of {a} on {b}, refitted every {item['every']}")
        ax.legend(loc="upper right", ncols=3)
        return _svg(fig)


def half_life(item: dict) -> str:
    """How fast the spread reverted inside each clock window; open circles where it did not.

    The dashed line is what a random walk reads in a window as long.
    """
    rows = item["windows"]
    with _figure(height=2.6) as (fig, (ax,)):
        n = np.arange(len(rows))
        life = _num([r["half_life_s"] for r in rows])
        stuck = np.array([r["reverts"] is False for r in rows])
        finite = life[np.isfinite(life)]
        roof = float(finite.max()) * 2 if len(finite) else 1.0
        for seg in dict.fromkeys(r["segment"] for r in rows):
            mask = np.array([r["segment"] == seg for r in rows])
            ax.scatter(n[mask], life[mask], s=14, color=SEGMENT_COLOR[seg],
                       label=SEGMENT_NAME[seg])  # fmt: skip
        if stuck.any():
            ax.scatter(n[stuck], np.full(stuck.sum(), roof), s=18, facecolors="none",
                       edgecolors=INK, label="no reversion")  # fmt: skip
        walk = item.get("random_walk_half_life_s")
        if walk is not None:
            ax.axhline(walk, color=INK, lw=0.9, ls="--", label=f"random walk, {walk:.0f} s")
        ax.set_yscale("log")
        if not len(finite):
            # no window came back: only the open circles and the random walk to scale by
            ref = [v for v in (walk, roof if stuck.any() else None)
                   if v is not None and np.isfinite(v) and v > 0]  # fmt: skip
            lo, hi = (min(ref), max(ref)) if ref else (1.0, 1.0)
            ax.set_ylim(lo / 10, hi * 10)
        _plain_log(ax)
        _window_axis(ax, [r["start"] for r in rows], most=7)  # the legend is beside it
        ax.set_ylabel("half-life, s")
        a, b = item["pair"]
        ax.set_title(f"Half-life of the {a} over {b} spread in each {item['every']} window")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        return _svg(fig)


def _plain_log(ax) -> None:
    """Label a log axis 0.1, 0.2, 0.5, 1, ... instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


# -- a market maker's charts ---------------------------------------------------

MODEL_STYLE = {"RA": ("#2a5d9f", "-"), "PR": ("#c0562c", "-"), "T": ("#6b6b6b", "--"),
               "F": ("#3a8d5b", "--")}  # fmt: skip


def quotes(ex: dict, name: str) -> str:
    """A maker's half hour: the touch as two steps, its resting quotes as lines, its fills as
    markers (buys up, sells down), and the inventory below on its own axis."""
    import matplotlib
    from matplotlib.figure import Figure

    t = np.asarray(ex["seconds"], dtype=float) / 60
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(7.4, 4.2), layout="constrained")
        top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1], squeeze=True)
        top.step(t, _num(ex["bid"]), where="post", color="#9aa7b8", lw=0.8, label="best bid")
        top.step(t, _num(ex["ask"]), where="post", color="#d3a99a", lw=0.8, label="best ask")
        top.step(t, _num(ex["bid_quote"]), where="post", color=LONG_COLOR, lw=1.0, zorder=5,
                 label="your bid")  # fmt: skip
        top.step(t, _num(ex["ask_quote"]), where="post", color=SHORT_COLOR, lw=1.0, zorder=5,
                 label="your ask")  # fmt: skip
        fills = np.asarray(ex["fills"], dtype=float).reshape(-1, 4)
        for side, marker, color, label in ((1, "^", LONG_COLOR, "bought"),
                                           (-1, "v", SHORT_COLOR, "sold")):  # fmt: skip
            f = fills[fills[:, 1] == side]
            top.scatter(f[:, 0] / 60, f[:, 2], marker=marker, s=9, color=color, alpha=0.55,
                        lw=0, zorder=4, label=f"{label} ({len(f):,})")  # fmt: skip
        top.set_ylabel("price")
        top.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        bottom.step(t, _num(ex["inventory"]), where="post", color=INK, lw=1.0)
        bottom.axhline(0, color="#999999", lw=0.8)
        bottom.set_ylabel("inventory")
        bottom.set_xlabel("minutes into the excerpt")
        seg = SEGMENT_NAME.get(ex.get("segment"), "")
        fig.suptitle(f"{name} on {seg}: the half hour with the most fills, under "
                     f"{ex['model']}", fontsize=9, fontweight="normal")  # fmt: skip
        return _svg(fig)


def fill_markouts(fans: dict, offsets: list[str], fee_bps: float, scored: dict) -> str:
    """The mid's move after a maker's resting fills, in the fill's direction, per segment: one
    line per fill model, RA and PR solid, T and F dashed. The diamond at 0 is where the scored
    model's average fill price sat against the mid; the dashed red line is its break-even,
    the maker fee less what the fill captured: a fill is worth holding to an offset when the
    line is above it there."""
    import matplotlib
    from matplotlib.figure import Figure

    segs = list(fans)
    xs = np.array([_seconds(o) for o in offsets])
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(7.4, 0.5 + 2.4 * len(segs)), layout="constrained")
        axes = fig.subplots(len(segs), 1, sharex=True, squeeze=False)[:, 0]
        for ax, seg in zip(axes, segs, strict=True):
            for m, f in fans[seg].items():
                color, ls = MODEL_STYLE[m]
                ax.plot(xs, _num(f["mean_bps"]), color=color, ls=ls, lw=1.3,
                        label=f"{m} ({f['n']:,} fills)")  # fmt: skip
            ax.axhline(0, color=INK, lw=0.6)
            ax.axvline(0, color=MUTED, lw=0.8, ls=":")
            m = scored[seg]
            cap = fans[seg][m].get("captured_bps_per_unit")
            if cap is not None:
                ax.scatter([0], [-cap], marker="D", s=24, color=INK, zorder=5,
                           label=f"{m}'s fill against the mid")  # fmt: skip
                ax.axhline(fee_bps - cap, color=BREAKEVEN, lw=1.0, ls="--",
                           label=f"{m}'s break-even: fee less capture")  # fmt: skip
            ax.set_title(f"{SEGMENT_NAME[seg]}", color=SEGMENT_COLOR[seg], loc="left")
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        axes[-1].set_xlabel("seconds from the fill")
        fig.supylabel("mid move in the fill's direction, bps", fontsize=8.5, color=MUTED)
        fig.suptitle("After the fill: the markout of each fill model", fontsize=9,
                     fontweight="normal")  # fmt: skip
        return _svg(fig)


def maker_pnl(pnl_by_segment: dict, name: str, model_by_segment: dict) -> str:
    """Where a maker's P&L goes, per segment: cumulative capture at the fill, that plus the mid's
    move in the 10 s after each fill, and the net. The first gap is adverse selection; the
    second is inventory held past 10 s, the flatten and the fees."""
    segs = list(pnl_by_segment)
    with _figure(ncols=len(segs), height=2.9, sharey=True) as (fig, axes):
        for ax, seg in zip(axes, segs, strict=True):
            p = pnl_by_segment[seg]
            h = p["hours"]
            ax.plot(h, p["captured"], color="#6c9bd2", lw=1.2, label="captured at the fill")
            ax.plot(h, p["adverse"], color=INK, lw=0.9, ls=":",
                    label="after 10 s of\nadverse selection")  # fmt: skip
            ax.plot(h, p["net"], color="#c0562c", label="net")
            ax.axhline(0, color="#999999", lw=0.8)
            ax.set_title(f"{SEGMENT_NAME[seg]}, {model_by_segment[seg]}",
                         color=SEGMENT_COLOR[seg])  # fmt: skip
            ax.set_xlabel("hours into the window")
        axes[0].set_ylabel("cumulative, bps of one unit")
        axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(f"Where the P&L of {name} goes", fontsize=9, fontweight="normal")
        return _svg(fig)


def arb_excerpt(ex: dict, name: str, quote: str, hedge: str) -> str:
    """An arbitrage's half hour: both venues' touches as steps (the quote venue's in grey and
    rust, the hedge venue's in blue tones), the resting quotes, each fill on the quote venue
    and its hedge on the other, joined by a thin line from the fill to the hedge."""
    import matplotlib
    from matplotlib.figure import Figure

    t = np.asarray(ex["seconds"], dtype=float) / 60
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(7.4, 3.6), layout="constrained")
        ax = fig.subplots(1, 1)
        ax.step(t, _num(ex["bid"]), where="post", color="#9aa7b8", lw=0.8,
                label=f"{quote} bid")  # fmt: skip
        ax.step(t, _num(ex["ask"]), where="post", color="#d3a99a", lw=0.8,
                label=f"{quote} ask")  # fmt: skip
        ax.step(t, _num(ex["hedge_bid"]), where="post", color="#2a5d9f", lw=0.7, alpha=0.7,
                label=f"{hedge} bid")  # fmt: skip
        ax.step(t, _num(ex["hedge_ask"]), where="post", color="#7a3d8f", lw=0.7, alpha=0.7,
                label=f"{hedge} ask")  # fmt: skip
        ax.step(t, _num(ex["bid_quote"]), where="post", color=LONG_COLOR, lw=1.0, zorder=5,
                label="your bid")  # fmt: skip
        ax.step(t, _num(ex["ask_quote"]), where="post", color=SHORT_COLOR, lw=1.0, zorder=5,
                label="your ask")  # fmt: skip
        fills = np.asarray(ex["fills"], dtype=float).reshape(-1, 4)
        hedges = np.asarray(ex["hedges"], dtype=float).reshape(-1, 4)
        for side, marker, color, label in ((1, "^", LONG_COLOR, "bought"),
                                           (-1, "v", SHORT_COLOR, "sold")):  # fmt: skip
            f = fills[fills[:, 1] == side]
            ax.scatter(f[:, 0] / 60, f[:, 2], marker=marker, s=16, color=color, lw=0, zorder=6,
                       label=f"{label} on {quote} ({len(f):,})")  # fmt: skip
        if len(hedges):
            ax.scatter(hedges[:, 0] / 60, hedges[:, 2], marker="o", s=12, facecolor="none",
                       edgecolor=INK, lw=0.8, zorder=6,
                       label=f"hedged on {hedge} ({len(hedges):,})")  # fmt: skip
        k = min(len(fills), len(hedges))
        for (tf, _, pf, _), (th, _, ph, _) in zip(fills[:k], hedges[:k], strict=True):
            ax.plot([tf / 60, th / 60], [pf, ph], color=MUTED, lw=0.5, zorder=3)
        ax.set_ylabel("price")
        ax.set_xlabel("minutes into the excerpt")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        seg = SEGMENT_NAME.get(ex.get("segment"), "")
        fig.suptitle(f"{name} on {seg}: the half hour with the most fills, under "
                     f"{ex['model']}", fontsize=9, fontweight="normal")  # fmt: skip
        return _svg(fig)


def arb_pnl(pnl_by_segment: dict, name: str, model_by_segment: dict) -> str:
    """Where an arbitrage's P&L goes, per segment, in USD: the cumulative capture of the maker
    fills against the hedge venue's mid, that after the hedge delay, that after the hedge's
    slippage (the gross), and the net after both legs' fees and the rebalances."""
    segs = list(pnl_by_segment)
    with _figure(ncols=len(segs), height=2.9, sharey=True) as (fig, axes):
        for ax, seg in zip(axes, segs, strict=True):
            p = pnl_by_segment[seg]
            h = p["hours"]
            ax.plot(h, p["captured"], color="#6c9bd2", lw=1.2, label="captured at the fill")
            ax.plot(h, p["delayed"], color=INK, lw=0.9, ls=":", label="after the hedge delay")
            ax.plot(h, p["gross"], color="#7a3d8f", lw=0.9, ls="--",
                    label="after the hedge's slippage")  # fmt: skip
            ax.plot(h, p["net"], color="#c0562c", label="net of fees")
            ax.axhline(0, color="#999999", lw=0.8)
            ax.set_title(f"{SEGMENT_NAME[seg]}, {model_by_segment[seg]}",
                         color=SEGMENT_COLOR[seg])  # fmt: skip
            ax.set_xlabel("hours into the window")
        axes[0].set_ylabel("cumulative, USD")
        axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(f"Where the P&L of {name} goes", fontsize=9, fontweight="normal")
        return _svg(fig)


# -- forward paper (blinded.forward) ---------------------------------------------------------

FORWARD_COLOR = "#c0562c"
EXCLUDED_COLOR = "#b0b0b0"


def _hour_axis(ax, stamps: list[str]) -> np.ndarray:
    """Hours as numbers from the first, labelled by the hour, ``DD HH`` across days, at no more
    than 8 ticks; the axis names the day when there is one."""
    import pandas as pd
    from matplotlib.ticker import NullLocator

    t = pd.to_datetime(stamps, utc=True)
    x = ((t - t[0]) / pd.Timedelta("1h")).to_numpy(dtype=float)
    every = max(1, int(np.ceil(len(x) / 8)))
    one_day = len({s.date() for s in t}) == 1
    fmt = "%H" if one_day else "%d %H"
    ax.set_xticks(x[::every], [s.strftime(fmt) for s in t[::every]])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(f"hour, UTC, {t[0]:%Y-%m-%d}" if one_day else "day and hour, UTC")
    return x


def forward_hourly(hours: list[dict]) -> str:
    """Each hour's net and fills, the judged hours as bars and the excluded ones hatched at the
    height of what the pass made in them, which no statistic counts."""
    with _figure(ncols=2, height=2.9) as (fig, (a, b)):
        for ax, key, label in ((a, "net_usd", "net, USD"), (b, "fills", "fills")):
            x = _hour_axis(ax, [h["hour"] for h in hours])
            y = np.array([h[key] or 0 for h in hours], dtype=float)
            judged = np.array([h["status"] == "judged" for h in hours])
            ax.bar(x[judged], y[judged], width=0.8, color=FORWARD_COLOR, label="judged")
            if (~judged).any():
                # the whole column hatched, so an excluded hour that made nothing still shows
                for xi in x[~judged]:
                    ax.axvspan(xi - 0.45, xi + 0.45, facecolor="none", hatch="////",
                               edgecolor="#e2e2e2", lw=0)  # fmt: skip
                ax.bar(x[~judged], y[~judged], width=0.8, color="white", hatch="////",
                       edgecolor=EXCLUDED_COLOR, label="not judged")  # fmt: skip
            ax.axhline(0, color="#999999", lw=0.8)
            ax.set_ylabel(label)
        a.legend(loc="upper left")
        fig.suptitle("The forward pass, hour by hour", fontsize=9, fontweight="normal")
        return _svg(fig)


def forward_cumulative(hours: list[dict], ref: dict, verdict: dict) -> str:
    """The cumulative forward net over the judged hours against the locked test's: its mean
    an hour as the expected line and its 90% interval scaled to the hours as the band."""
    import json

    judged = [h for h in hours if h["status"] == "judged"]
    with _figure(height=2.9) as (fig, (ax,)):
        n = np.arange(1, len(judged) + 1)
        ax.plot(n, np.cumsum([h["net_usd"] for h in judged]), color=FORWARD_COLOR,
                label="forward, judged hours")  # fmt: skip
        ax.plot(n, ref["m_net"] * n, color=INK, ls="--", lw=0.9,
                label="the locked test's mean an hour")  # fmt: skip
        ci = json.loads(ref.get("ci_json") or "null")
        if ci:
            lo, hi = ci["usd_per_hour"]
            ax.fill_between(n, lo * n, hi * n, color="#dddddd", lw=0,
                            label="its 90% interval")  # fmt: skip
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xlabel("judged hours")
        ax.set_ylabel("cumulative net, USD")
        ax.set_title("Forward against the locked test")
        ax.legend(loc="upper left")
        return _svg(fig)


def forward_cusum(verdict: dict) -> str:
    """Both lower CUSUMs over the judged hours, each with its threshold drawn."""
    with _figure(height=2.6) as (fig, (ax,)):
        n = np.arange(1, len(verdict["cusum_net"]) + 1)
        h_net, h_fills = verdict["h_net"], verdict["h_fills"]
        ax.plot(n, verdict["cusum_net"], color=FORWARD_COLOR, label="hourly net")
        ax.plot(n, verdict["cusum_fills"], color="#2a5d9f", ls="--", label="hourly fills")
        ax.axhline(h_net, color=FORWARD_COLOR, lw=0.8, ls=":", label=f"h = {h_net:g}, net")
        ax.axhline(h_fills, color="#2a5d9f", lw=0.8, ls=":", label=f"h = {h_fills:g}, fills")
        top = max(h_net, h_fills) * 1.2
        ax.set_ylim(bottom=0, top=max(top, max(verdict["cusum_net"] + verdict["cusum_fills"],
                                               default=0) * 1.1))  # fmt: skip
        ax.set_xlabel("judged hours")
        ax.set_ylabel("CUSUM, sds of the locked hours")
        ax.set_title("The verdict: a CUSUM below the locked test")
        ax.legend(loc="upper left")
        return _svg(fig)


def forward_decomposition(pnl: dict, day: str) -> str:
    """Where the last judged day's P&L went, as the arbitrage report draws it: the capture at
    the fill, after the hedge delay, after the hedge's slippage, and net of fees."""
    with _figure(height=2.9) as (fig, (ax,)):
        h = pnl["hours"]
        ax.plot(h, pnl["captured"], color="#6c9bd2", lw=1.2, label="captured at the fill")
        ax.plot(h, pnl["delayed"], color=INK, lw=0.9, ls=":", label="after the hedge delay")
        ax.plot(h, pnl["gross"], color="#7a3d8f", lw=0.9, ls="--",
                label="after the hedge's slippage")  # fmt: skip
        ax.plot(h, pnl["net"], color=FORWARD_COLOR, label="net of fees")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xlabel("hours into the window")
        ax.set_ylabel("cumulative, USD")
        ax.set_title(f"Where the P&L of {day} went")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        return _svg(fig)


def forward_stock(rows: list, day: str) -> str:
    """The two venues' coin and USDT after every leg and transfer of the last judged day."""
    import pandas as pd

    with _figure(ncols=2, height=2.6) as (fig, (a, b)):
        t0 = rows[0][0]
        x = (np.array([r[0] for r in rows], dtype=float) - t0) / pd.Timedelta("1h").value
        for ax, (i, j), label in ((a, (2, 4), "coin"), (b, (3, 5), "USDT")):
            ax.step(x, [r[i] for r in rows], where="post", color=FORWARD_COLOR,
                    label="quote venue")  # fmt: skip
            ax.step(x, [r[j] for r in rows], where="post", color="#2a5d9f", ls="--",
                    label="hedge venue")  # fmt: skip
            ax.set_xlabel("hours into the window")
            ax.set_ylabel(label)
        a.legend(loc="best")
        fig.suptitle(f"The stock on {day}", fontsize=9, fontweight="normal")
        return _svg(fig)
