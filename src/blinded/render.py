"""Plain-text rendering of judge results."""

from __future__ import annotations

import pandas as pd

_MAIN = {
    "n": "n",
    "n_eff": "n_eff",
    "ic": "IC",
    "ic_se_nw": "se_nw",
    "ic_se_boot": "se_boot",
    "rank_ic": "rankIC",
    "rank_ic_se_nw": "se_nw",
    "rank_ic_se_boot": "se_boot",
    "rank_ic_t": "t_nw",
    "markout_bps": "mkout_bps",
    "tails_bps": "tails_bps",
}


def format_result(result) -> str:
    """The text a person reads: verdict first, then the tables."""
    lines = [
        f"{result.name}  [{result.status}]  {result.hypothesis}",
        f"window {result.window}  latency {_ms(result.latency)}  {_family(result)}",
        f"truncation test: {result.verdict if result.verdict else result.reason}",
    ]
    lines += [f"warning: {w}" for w in result.warnings]
    if not result.ok:
        return "\n".join(lines)
    main = result.horizons
    cells = [
        [f"{n:d}", f"{round(e):d}", *(f"{v:.4f}" for v in rest)]
        for n, e, *rest in main.select(list(_MAIN)).iter_rows()
    ]
    lines += ["", "by horizon (IC: Pearson, rankIC: Spearman; SEs allow for overlapping returns)"]
    lines += [table(main["horizon"].to_list(), list(_MAIN.values()), cells)]
    decay = result.decay
    rows = [[f"{v:.4f}" for v in decay[c]] for c in ("ic", "rank_ic")]
    lines += ["", "decay", table(["IC", "rankIC"], decay["horizon"].to_list(), rows)]
    lines += [
        "",
        f"turnover {result.turnover:.4f} sign changes per step, "
        f"lag-1 autocorrelation {result.autocorr:.4f}",
    ]
    if result.stability.height:
        lines += ["", _stability(result.stability)]
    return "\n".join(lines)


def table(index: list[str], columns: list[str], cells: list[list[str]]) -> str:
    """Rows of preformatted cells under right-aligned headers, two spaces apart."""
    left = max((len(i) for i in index), default=0)
    widths = [max([len(c), *(len(row[j]) for row in cells)]) for j, c in enumerate(columns)]

    def line(first: str, values: list[str]) -> str:
        return first.ljust(left) + "".join(
            f"  {v:>{w}}" for v, w in zip(values, widths, strict=True)
        )

    body = [line(i, row) for i, row in zip(index, cells, strict=True)]
    return "\n".join([line("", columns), *body])


def _stability(stab) -> str:
    ric = stab["rank_ic"]
    agree = ((ric > 0) == (ric.mean() > 0)).sum()
    head = f"stability at the primary horizon, {agree}/{stab.height} windows share the sign"
    starts = stab["start"].dt.strftime("%H:%M").to_list()
    return head + "\n" + table(["rankIC"], starts, [[f"{v:.3f}" for v in ric]])


def _ms(td: pd.Timedelta) -> str:
    return f"{td / pd.Timedelta('1ms'):g}ms"


def _family(result) -> str:
    """``directional on XRPUSDT``, ``pairs on A/B``; with an archetype,
    ``pairs_trading (relative value), pairs on A/B``."""
    family = result.family or "no family"
    out = f"{family} on {'/'.join(result.legs)}" if result.legs else family
    if getattr(result, "archetype", None):
        from blinded.archetypes import EDGES

        out = f"{result.archetype} ({EDGES[result.edge].title}), {out}"
    return out


_SIM = {
    "net_bps_per_hour": ("net/h", "{:.2f}"),
    "gross_bps_per_hour": ("gross/h", "{:.2f}"),
    "sharpe_per_sqrt_hour": ("SR/sqrt(h)", "{:.3f}"),
    "max_drawdown_bps": ("maxDD", "{:.1f}"),
    "fills": ("fills", "{:.0f}"),
    "fees_bps": ("fees", "{:.1f}"),
    "turnover_per_hour": ("turn/h", "{:.1f}"),
    "net_bps_per_fill": ("net/fill", "{:.3f}"),
    "gross_bps_per_fill": ("gross/fill", "{:.3f}"),
    "hit_rate": ("hit", "{:.3f}"),
}
_MAKER = {
    "units": ("units", "{:.4g}"),
    "captured_bps": ("captured", "{:.1f}"),
    "adverse_bps_10s": ("adverse 10s", "{:.1f}"),
    "post_only_rejected": ("rejected", "{:.0f}"),
}


def format_strategy(result) -> str:
    """A strategy's verdict and its sim rows, P&L in bps of notional."""
    lines = [
        f"{result.name}  [{result.status}]  requires {', '.join(result.requires) or 'nothing'}",
        f"window {result.window}  latency {_ms(result.latency)}  {_family(result)}",
        f"truncation test: {result.verdict if result.verdict else result.reason}",
    ]
    lines += [f"warning: {w}" for w in result.warnings]
    if not result.ok:
        return "\n".join(lines)
    rows = result.rows
    columns = _SIM | ({"gross_exposure": ("exposure", "{:.2f}")} if len(result.legs) > 1 else {})
    if "maker_share" in rows.columns:  # the maker-taker row's share of units that rested
        columns = columns | {"maker_share": ("maker share", "{:.2f}")}
    maker = "model" in rows.columns
    if maker:
        columns = {"model": ("model", "{}")} | _SIM | _MAKER
    cells = []
    for row in rows.iter_rows(named=True):
        cells.append([_cell(row.get(k), fmt) for k, (_, fmt) in columns.items()])
    unit = "one notional per leg" if len(result.legs) > 1 else "notional"
    what = "fill engine (bps of one unit" if maker else f"taker sim (bps of {unit}"
    lines += ["", f"{what}; per hour, per fill)"]
    lines += [table(rows["row"].to_list(), [h for h, _ in columns.values()], cells)]
    notes = [f"{r['row']}: {r['note']}" for r in rows.iter_rows(named=True) if r["note"]]
    lines += ["", *notes]
    return "\n".join(lines)


def _cell(value, fmt: str) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return fmt.format(value)
