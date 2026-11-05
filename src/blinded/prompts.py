"""What the agent reads: RULES.md, the appended system prompt, the iteration prompts and the
schema of the summary each session returns."""

from __future__ import annotations

PHASES = ("features", "strategies", "finalise")

SCHEMA = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "description": "the ideas submitted in this session",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "trial_ids": {"type": "array", "items": {"type": "integer"}},
                    "rationale": {"type": "string", "description": "one line"},
                },
                "required": ["name", "trial_ids", "rationale"],
            },
        },
        "next": {"type": "string", "description": "what you would try next, one line"},
        "phase_done": {
            "type": "boolean",
            "description": "true if you have no further idea worth submitting in this phase",
        },
        "final_trial_id": {
            "type": ["integer", "null"],
            "description": "finalise phase only: the trial id of your final strategy",
        },
    },
    "required": ["ideas", "next", "phase_done"],
}

SYSTEM_APPEND = """\
You are the proposer in a research loop for trading signals. A judge you cannot
influence scores everything you submit. Your working directory is a workspace;
it is the whole of what you may read and write. Files outside it are not yours,
and an attempt to read the validation or test data, by any route, halts the run.
Work with the files in the workspace, the `blinded` command and Python. There is
no network. Each session is fresh: what you know of earlier sessions is in
LEDGER.md and NOTES.md. Be concise in NOTES.md; it is read every session.
"""

_COMMON = """\
# RULES

{world}

## The contract

A feature is a function `panel -> polars expression` in `features/<name>.py`,
registered with `@bl.feature("one line: the hypothesis")`:

    import blinded as bl
    from blinded import features as F

    @bl.feature("spot closes a perp-spot basis gap within seconds")
    def basis_dev(panel):
        b = F.basis("XRPUSDT_PERP", "XRPUSDT", log=True) * 1e4
        return b - F.lag(F.roll_mean(b, 3000), 1)

`panel.frame` is a wide polars frame on the decision grid: `ts` and one column
per instrument and field, named `SYM.field` (`XRPUSDT.mid`, `XRPUSDT.bid_qty`,
`XRPUSDT_PERP.cum_signed_qty`, ...); `print(panel.frame.columns)` lists them.
Return a polars expression over those columns (preferred), a polars Series or
a pandas Series on the grid. The library `blinded.features` builds the usual
ones; list it with
`python -c "import blinded.features as F; print(F.__all__)"` and read one with
`python -c "import blinded.features as F; help(F.basis)"`.

A strategy is a function `(panel, features) -> position` in
`strategies/<name>.py`, registered with `@bl.strategy(requires=[...])`.
`features` maps each required feature name to its column expression. Positions
are in [-1, 1], the position you want to hold from the decision time plus the
latency. The built-ins are the usual way to turn a feature into a position:

    @bl.strategy(requires=["basis_dev"])
    def basis_threshold(panel, features):
        return bl.threshold(features["basis_dev"], z_in=2.0, z_out=0.5)

`bl.linear(f)` is the other; both scale `f` by its standard deviation over all
earlier 30-minute windows, never the current one.

Every function must be pure: same panel in, same values out; no files, no
network, no clock, no unseeded randomness.

## Commands

    blinded submit features/basis.py               # every feature and strategy in the file
    blinded submit strategies/s.py --name s        # one of them
    blinded eval features/basis.py --brief brief.yaml --window train   # local, not recorded

A refused Bash command does not mean Bash is off. Split it into plain
commands, without shell loops, one `blinded submit` per file, and carry on.

{submit}
{families}
## Good practice

- Trailing and expanding windows only. A full-sample mean, rank, scaler or fit
  uses the future at every earlier point.
- Returns are measured from the decision time plus the latency in the brief,
  so a signal that has decayed within the latency is worth nothing to a taker.
- Watch turnover: every position change pays the spread and the fee.
- One idea per feature. Name it for what it measures.
- Read LEDGER.md before proposing: an idea already tried, under any name,
  counts again.
"""

_PROTECTED_WORLD = """\
This workspace holds the training segment of the data (`data/train.parquet`,
read through `brief.yaml`) and nothing else. Validation and test are in a
vault you cannot read. DATA.md describes the data, LEDGER.md every trial so
far, BUDGET.md what is left of the run, NOTES.md your own notes from earlier
sessions."""

_PROTECTED_SUBMIT = """\
Each submission adds one ledger row per feature or strategy and returns JSON:
the status, the reason, train metrics in full, and for validation the
truncation verdict, `released` and `best`. A strategy may require any feature
that has been accepted, by name. Each evaluation of your code is limited to
120 s and 4 GB.

## What is checked, and why

- Look-ahead, by recomputation: your feature is computed on copies of the data
  cut at several times, and any past value that changes is a rejection
  (`rejected_lookahead`). The same test runs again on the validation panel.
- Determinism: two full runs must agree (`rejected_nondeterministic`).
- Imports outside numpy, pandas, polars, blinded, math, statistics, functools,
  itertools, collections, dataclasses and typing, and the calls open, exec,
  eval, compile, `__import__`, getattr, setattr, globals, locals, vars, and any
  `read_*`, `scan_*` or `write_*` of files, are rejected before anything runs
  (`rejected_static`): they could reach data the judge holds.
- An error returns its exception type and line, never the message.

## What you see of validation

Every accepted submission is scored on validation inside the vault: rank IC
at the primary horizon for a feature, Sharpe per square-root hour after the
brief's latency and fees for a strategy. You learn only `released`, true when
the score beat the released best of its kind by more than its own noise, and
`best`, the released best rounded to that noise. The first release of each
kind must beat zero. A strategy's noise is measured in blocks at least as
long as its time-weighted mean hold, since one hold is one bet: a strategy
whose holds average longer than a twentieth of the validation window is
never released. You never see the score itself, so tuning against
validation is not possible; tune on train, where everything is exact.

## How the run proceeds

1. Features phase. It ends after {features} sessions in a row without a
   feature release, after 60% of the iterations, or when you return
   `phase_done: true` in two sessions in a row. A session counts the same
   whether it submitted one idea or seven.
2. Strategies phase. It ends after {strategies} sessions in a row without a
   strategy release, or when the budget is spent.
3. Finalise. You name the final strategy by trial id; no new code. A trial
   that was never released is replaced by the released strategy with the
   best released score.
4. The final strategy is scored once on the test segment, by the driver.

The test segment is never scored on request, and any attempt to read the
vault, or the agent and driver directories beside the workspace, halts the run.
"""

_NAIVE_WORLD = """\
This workspace holds all of the data (`data/all.parquet`, read through
`brief.yaml`). DATA.md describes it, LEDGER.md every trial so far, BUDGET.md
what is left of the run, NOTES.md your own notes from earlier sessions."""

_NAIVE_SUBMIT = """\
Each submission adds one ledger row per feature or strategy and returns JSON
with its metrics on the window you ask for: `--window all` (the default),
`train`, `val`, `test` or `START/END`, for example
`blinded submit strategies/s.py --window test`. Metrics are exact, at zero
latency, with the brief's fees. Nothing is rejected for how it is computed.

## How the run proceeds

1. Features phase, for 60% of the iterations or until you return
   `phase_done: true` in two sessions in a row.
2. Strategies phase, until the budget is spent.
3. Finalise. You name your best strategy by trial id; no new code.
"""


_FAMILIES = """\
## Families

This run studies {n} kinds of strategy on the same data. Each session is for
one family, named in the prompt and in BUDGET.md: submit features and
strategies of that family, with `family=` in the decorator. One final
strategy is picked from every family's.
"""

_LADDER = """
The Ladder keeps a released best for each family and kind, and a family's
phase ends after {features} of its sessions in a row without a release of
one of its features ({strategies} for strategies). Submissions of another
family are judged and recorded, and count towards that family's trials only.
The count of trials the final Sharpe is deflated for spans every family.
"""

_DIRECTIONAL = """
### directional

Bets on an instrument's own next move, in {instruments}. A feature predicts
the forward return of its `target` and a strategy holds one position in the
instrument it `trades`; both default to {target}.

    @bl.feature("...", target="{example}")
    @bl.strategy(requires=["..."], trades="{example}")
"""

_LEAD_LAG = """
### lead_lag

Bets that a leader's recent move predicts a follower's later move. Leaders:
{leaders}. Followers: {followers}. A feature predicts a follower's forward
return and a strategy holds one position in a follower. `F.lead_ret(leader, k)`
is the leader's trailing return, `F.lead_gap(leader, follower, k)` that less
the follower's, the move it has not made yet, `F.lead_gap_z` the gap over
its trailing sd, and `F.lagged_corr(leader, follower, k, window)` how closely
the follower has lately followed the leader's move k earlier. DATA.md has
the cross-correlation of every leader and follower on train, with how much
of it is left after a taker's fill.

    @bl.feature("...", family="lead_lag", target="{example}")
    def gap(panel):
        return F.lead_gap("{leader}", "{example}", "1s")

    @bl.strategy(requires=["gap"], family="lead_lag", trades="{example}")
"""

_PAIRS = """
### pairs

Bets that the spread between two legs returns to its recent mean, in
{pairs}. A feature predicts the forward log return of A less that of B, one
notional of each, so a high value means the spread of A over B is expected
to rise. A strategy returns a position for each leg, in units of one
notional of that leg; each leg pays its own spread and fee. `F.hedge_ratio`,
`F.pair_spread` and `F.spread_z` fit the hedge ratio, the mean and the scale
on the window before each clock window, and `bl.pairs` is the built-in:

    @bl.feature("...", family="pairs", pair=("{a}", "{b}"))
    def a_b_z(panel):
        return F.spread_z("{a}", "{b}", window="2h", every="30m")

    @bl.strategy(family="pairs", pair=("{a}", "{b}"))
    def a_b_pairs(panel, features):
        return bl.pairs("{a}", "{b}", z_in=2.0, z_out=0.5, window="2h", every="30m")
"""

_BASKET = """
### basket

Bets that some names of a cross-section beat the rest, in {legs}. A feature
returns one value per leg, a mapping over `panel.legs`, and is scored by its
cross-sectional rank IC: at each grid point, the rank correlation across the
legs of the feature with each leg's forward return, averaged over the window.
A strategy returns a position for each leg. `F.each(panel.legs, fn)` builds a
mapping from a function of a symbol, `F.xs_demean`, `F.xs_zscore` and
`F.xs_rank` combine its legs row by row, and `bl.basket` is the built-in:

    @bl.feature("...", family="basket")
    def score(panel):
        return F.xs_zscore(F.each(panel.legs, lambda s: -F.ret(F.mid(s), "4h")))

    @bl.strategy(requires=["score"], family="basket")
    def long_short(panel, features):
        return bl.basket(features["score"], top=4, every="4h")
"""


_ARCHETYPES = """\
## Archetypes

This run studies archetypes grouped by edge, who pays the strategy: {edges}.
ARCHETYPES.md is the menu: what runs on this data, the decorator of each
with this brief's instruments, its built-ins, and what does not run and why.
Each session is for one edge, named in the prompt and BUDGET.md; submit
features and strategies with that edge's decorators.{ladder} Submitting a
refused archetype, or a bare `bl.feature` or `bl.strategy`, is an error row
and costs a submission. One final strategy is picked from every edge's.

An archetype's decorator takes the instruments the archetype trades and
does the rest; in this run it takes the place of `@bl.feature` and
`@bl.strategy` above:

    import blinded as bl
    from blinded import features as F

    x = bl.{example}

    @x.feature("one line: the hypothesis")
    def my_feature(panel):
        ...

    @x.strategy(requires=["my_feature"])
    def my_strategy(panel, features):
        ...

`@bl.{example}` on a function is the same as `@x.strategy()`.
"""

_ARCHETYPE_LADDER = """ The Ladder
keeps a released best for each edge and kind, shared by the archetypes of
the edge, and an edge's phase ends after {features} of its sessions in a row
without a release of one of its features ({strategies} for strategies)."""


def rules(mode: str, brief=None, budget=None, menu=None) -> str:
    """RULES.md for a protected or a naive run.

    ``budget`` gives the streak lengths; with a ``brief`` that has
    ``families``, a section describes each one, and with one that has
    ``archetypes``, a section points at the ``menu`` in ARCHETYPES.md.
    """
    streaks = {
        "features": getattr(budget, "feature_streak_sessions", 3),
        "strategies": getattr(budget, "strategy_streak_sessions", 3),
    }
    families = ""
    if brief is not None and brief.has_archetypes:
        families = archetypes_section(brief, mode, streaks, menu)
    elif brief is not None and brief.has_families:
        families = families_section(brief, mode, streaks)
    if brief is not None and brief.data.cost_model is not None:
        families = bars_section(brief) + families
    if brief is not None and "borrow" in brief.data.streams:
        families = _BORROW + families
    if mode == "naive":
        return _COMMON.format(world=_NAIVE_WORLD, submit=_NAIVE_SUBMIT, families=families)
    submit = _PROTECTED_SUBMIT.format(**streaks)
    return _COMMON.format(world=_PROTECTED_WORLD, submit=submit, families=families)


_BARS = """\
## Bars

The panel is bars: `SYM.mid` is the bar's close, `SYM.bid` and `SYM.ask` are
the close plus and minus a measured half spread, and a fill is the next bar's
close at that spread. There is no book inside a bar. {line}.

"""


def bars_section(brief) -> str:
    """The Bars section of RULES.md, for a file whose bid and ask are a declared model."""
    line = brief.data.cost_model.line()
    return _BARS.format(line=line[0].upper() + line[1:])


_BORROW = """\
## Borrow

Shorting spot means borrowing the coin, and the loan costs interest.
`SYM.borrow_rate` is what borrowing SYM costs a day, as a fraction of the
position's notional, as the rate was published. A short spot position pays the
rate in force at each step for as long as it is held, and the borrow is in
every net P&L, the Ladder's included.

"""


_MAKER = """
A market-making strategy returns quotes, not positions: `bl.quote(...)` on
the panel. The sim rests them, fills them against the recorded trades and the
recorded queue under four models, and the Ladder scores the worse of the two
queue models; you see `released` and `best` as for any strategy, never which
model. The train rows show all four. Keep the size small: the sim does not
know the market would react to a large quote.
"""


def archetypes_section(brief, mode: str, streaks: dict, menu) -> str:
    """The Archetypes section of RULES.md, which points at ARCHETYPES.md for the menu; where
    market making runs, a paragraph on what its strategies return and how they are scored."""
    from blinded.archetypes import EDGES, REGISTRY, example

    edges = ", ".join(EDGES[e].title for e in menu.edges)
    first = REGISTRY[menu.runnable[0]]
    ladder = _ARCHETYPE_LADDER.format(**streaks) if mode == "protected" else ""
    text = _ARCHETYPES.format(edges=edges, ladder=ladder, example=example(first, brief)[0])
    if "market_making" in menu.runnable:
        text += _MAKER
    return text


def families_section(brief, mode: str, streaks: dict) -> str:
    """The Families section of RULES.md, one paragraph per family in the brief."""
    text = _FAMILIES.format(n=len(brief.families))
    text += _LADDER.format(**streaks) if mode == "protected" else ""
    for fam in brief.families:
        if fam.name == "directional":
            text += _DIRECTIONAL.format(instruments=", ".join(fam.instruments),
                                        target=brief.target.instrument,
                                        example=fam.instruments[-1])  # fmt: skip
        elif fam.name == "lead_lag":
            follower = fam.followers[0]
            leader = next(x for x in fam.leaders if x != follower)
            text += _LEAD_LAG.format(leaders=", ".join(fam.leaders),
                                     followers=", ".join(fam.followers),
                                     example=follower, leader=leader)  # fmt: skip
        elif fam.name == "basket":
            text += _BASKET.format(legs=", ".join(fam.instruments))
        else:
            listed = ", ".join(f"{a} over {b}" for a, b in fam.pairs)
            text += _PAIRS.format(pairs=listed, a=fam.pairs[0][0], b=fam.pairs[0][1])
    return text


_ITERATION = """\
Read RULES.md, DATA.md, LEDGER.md, BUDGET.md and NOTES.md. You are in the
{phase} phase.{family} Propose and submit between one and three new {kind} with
`blinded submit`. Prefer ideas the ledger does not already contain.{extra}
Update NOTES.md with what you learned. End by returning the JSON summary,
with `phase_done` true only if you have no further {kind} worth trying."""

_FINALISE = """\
Read RULES.md, LEDGER.md, BUDGET.md and NOTES.md. This is the finalise phase.
Name the final strategy by its trial id from LEDGER.md: {which}. You may not
submit new code. Return the JSON summary with `final_trial_id` set (null if
there is no strategy you would stand behind), `ideas` empty and `phase_done`
true."""


def iteration(phase: str, mode: str = "protected", family=None) -> str:
    """The prompt for one session in ``phase``; ``family`` is the session's, in a run with
    families, or its :class:`~blinded.brief.Group` (an edge), in a run with archetypes."""
    if phase == "finalise":
        which = (
            "the strategy with the best numbers you have seen"
            if mode == "naive"
            else "the released strategy you expect to do best on unseen data"
        )
        if family is None:
            return _FINALISE.format(which=which)
        unit = "edge" if _is_edge(family) else "family"
        return _FINALISE.format(which=which + f", from any {unit}")
    kind = "features" if phase == "features" else "strategies"
    extra = ""
    if phase == "strategies":
        extra = (
            " A strategy uses accepted features by name in `requires`; you may"
            " also submit a feature it needs."
        )
    return _ITERATION.format(phase=phase.upper(), kind=kind, extra=extra, family=_for(family))


def _is_edge(group) -> bool:
    return bool(getattr(group, "members", ()))


def _for(family) -> str:
    if family is None:
        return ""
    if _is_edge(family):
        return (
            f" This session is for the {family.title} edge: {', '.join(family.members)}. "
            "Submit ideas with those decorators."
        )
    if family.name == "pairs":
        on = "the pairs " + ", ".join(f"{a} over {b}" for a, b in family.pairs)
    elif family.name == "lead_lag":
        on = f"leaders {', '.join(family.leaders)} and followers {', '.join(family.followers)}"
    else:
        on = ", ".join(family.instruments)
    return (
        f" This session is for the {family.name} family, on {on}: submit {family.name}"
        f' ideas, with family="{family.name}" in the decorator.'
    )
