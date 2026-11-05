"""Shared fixtures: a synthetic event file in the demo file's layout."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from blinded import brief as briefs
from blinded.panel import Panel

# the judge runs judged code only inside srt (blinded.isolate.fence); tests run it without,
# unless BLINDED_SRT names one, and the tests of the fence itself set and unset this
os.environ.setdefault("BLINDED_UNFENCED", "1")

EXAMPLE_BRIEF = Path(__file__).parents[1] / "examples" / "demo.brief.yaml"
TICK = 1e-4


def local(*parts: str) -> Path:
    """A path under BLINDED_LOCAL_DATA, which holds recorded runs and downloaded market data.
    The tests that read one skip when it is not there."""
    return Path(os.environ.get("BLINDED_LOCAL_DATA", "/nonexistent")).joinpath(*parts)


def _quotes(rng, times_s, fair, lag_s, sym):
    """Book rows that track ``fair`` delayed by ``lag_s`` seconds, one or two ticks wide."""
    price = fair(times_s - lag_s)
    bid = np.floor(price / TICK) * TICK
    ask = bid + TICK * rng.choice([1, 1, 1, 2], size=len(times_s))
    return pd.DataFrame(
        {
            "t": times_s,
            "side": None,
            "price": np.nan,
            "qty": np.nan,
            "bid0_px": bid,
            "ask0_px": ask,
            "bid0_qty": rng.lognormal(7, 1, len(times_s)).round(1),
            "ask0_qty": rng.lognormal(7, 1, len(times_s)).round(1),
            "symbol": sym,
            "is_trade": False,
        }
    )


def _trades(rng, book, rate, span_s, sym):
    """Trades at the touch of the book just before them; some share a timestamp."""
    times = np.sort(rng.uniform(0, span_s, rng.poisson(rate * span_s)))
    times[1::7] = times[0::7][: len(times[1::7])]
    idx = np.searchsorted(book.t.to_numpy(), times, side="right") - 1
    keep = idx >= 0
    times, idx = times[keep], idx[keep]
    state = book.iloc[idx].reset_index(drop=True)
    buy = rng.random(len(times)) < 0.5
    return pd.DataFrame(
        {
            "t": times,
            "side": np.where(buy, "B", "S"),
            "price": np.where(buy, state.ask0_px, state.bid0_px),
            "qty": rng.lognormal(4, 1.5, len(times)).round(1),
            "bid0_px": state.bid0_px,
            "ask0_px": state.ask0_px,
            "bid0_qty": state.bid0_qty,
            "ask0_qty": state.ask0_qty,
            "symbol": sym,
            "is_trade": True,
        }
    )


def make_synthetic(
    path: Path, minutes: int = 30, seed: int = 7, rates=(8, 6), trade_rate: float = 4
) -> Path:
    """Write a two-instrument L1 + trades file in the XRP layout.

    The perp quotes a random-walk fair price at once and the spot quotes it
    300 ms late, so the perp-spot basis predicts spot. ``rates`` are the two
    books' updates per second and ``trade_rate`` each one's trades.
    """
    rng = np.random.default_rng(seed)
    span_s = minutes * 60
    fine = np.arange(-1.0, span_s + 1.0, 0.01)
    walk = 1.4 * np.exp(np.cumsum(rng.normal(0, 2e-5, len(fine))))

    def fair(t):
        return walk[np.clip(((t + 1.0) / 0.01).astype(int), 0, len(walk) - 1)]

    rows = []
    for sym, rate, lag in (("XRPUSDT_PERP", rates[0], 0.0), ("XRPUSDT", rates[1], 0.3)):
        times = np.sort(rng.uniform(0, span_s, rng.poisson(rate * span_s)))
        book = _quotes(rng, times, fair, lag, sym)
        rows += [book, _trades(rng, book, trade_rate, span_s, sym)]
    frame = pd.concat(rows).sort_values("t", kind="stable")
    frame["mid"] = (frame.bid0_px + frame.ask0_px) / 2
    frame.index = pd.Timestamp("2024-11-28 02:00") + pd.to_timedelta(frame.pop("t"), unit="s")
    frame.index.name = "ts"
    frame["side"] = frame.side.astype("category")
    frame.to_parquet(path)
    return path


def brief_dict(data_path: Path) -> dict:
    raw = yaml.safe_load(EXAMPLE_BRIEF.read_text())
    raw["data"]["path"] = str(data_path)
    return raw


def write_brief(raw: dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture(scope="session")
def synthetic_brief(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic")
    data = make_synthetic(root / "events.parquet")
    raw = brief_dict(data)
    raw["stats"]["n_boot"] = 200
    return write_brief(raw, root / "brief.yaml")


@pytest.fixture(scope="session")
def long_brief(tmp_path_factory) -> Path:
    """Two hours of synthetic data: train 02:00-03:00, val 03:05-03:30, test 03:35-04:00."""
    root = tmp_path_factory.mktemp("long")
    data = make_synthetic(root / "events.parquet", minutes=120, seed=11)
    raw = brief_dict(data)
    raw["stats"]["n_boot"] = 200
    return write_brief(raw, root / "brief.yaml")


@pytest.fixture(scope="session")
def run_dir(long_brief, tmp_path_factory):
    """A run directory built from the two-hour file. Tests must not write to it."""
    from blinded import vault

    return vault.init(long_brief, tmp_path_factory.mktemp("runs") / "run")


@pytest.fixture(scope="session")
def pairs_brief(long_brief, tmp_path_factory) -> Path:
    """The two-hour file with a directional family on both instruments and a pairs family,
    perp over spot."""
    raw = yaml.safe_load(long_brief.read_text())
    raw["families"] = {"directional": {"instruments": ["XRPUSDT", "XRPUSDT_PERP"]},
                       "pairs": {"pairs": [["XRPUSDT_PERP", "XRPUSDT"]]}}  # fmt: skip
    return write_brief(raw, tmp_path_factory.mktemp("pairs") / "brief.yaml")


@pytest.fixture(scope="session")
def pairs_run(pairs_brief, tmp_path_factory):
    """A run directory with the two families. Tests must not write to it."""
    from blinded import vault

    return vault.init(pairs_brief, tmp_path_factory.mktemp("runs") / "pairs")


@pytest.fixture(scope="session")
def archetypes_brief(tmp_path_factory) -> Path:
    """Fourteen synthetic hours on a 1 s grid, with an ``archetypes`` block.

    Train 02:00-09:00, validation 09:05-12:30, test 12:35-16:00. Validation is
    205 minutes, so ``pairs_trading`` (20 x 10 min = 200 min) runs, by 5
    minutes, as on the Tardis day; the two-hour file's 25 minutes would refuse
    it. Quieter books than the default keep it to 277,189 rows. No fee and
    1-minute blocks, as in F1's families run. The file keeps every book row
    and trade, so market making could run on it too; ``only`` leaves it out,
    and the run studies the four archetypes of speed and relative value.
    """
    root = tmp_path_factory.mktemp("archetypes")
    data = make_synthetic(root / "events.parquet", minutes=840, seed=5, rates=(2, 1.5),
                          trade_rate=1)  # fmt: skip
    raw = brief_dict(data)
    raw["name"] = "synthetic-14h"
    raw["grid"] |= {"step": "1s", "start": "2024-11-28T02:00:00Z"}
    raw["target"]["horizons"] = ["1s", "10s", "60s"]
    raw["execution"]["taker_fee_bps"] = 0.0
    raw["stats"] |= {"block": "1min", "n_boot": 200}
    raw["archetypes"] = {"only": ["lead_lag", "order_book_imbalance", "trade_flow_persistence",
                                  "pairs_trading"]}  # fmt: skip
    return write_brief(raw, root / "brief.yaml")


@pytest.fixture(scope="session")
def archetypes_run(archetypes_brief, tmp_path_factory):
    """A run directory on the fourteen-hour file. Tests must not write to it."""
    from blinded import vault

    return vault.init(archetypes_brief, tmp_path_factory.mktemp("runs") / "archetypes")


@pytest.fixture(scope="session")
def synthetic_panel(synthetic_brief) -> Panel:
    return Panel.from_brief(briefs.load(synthetic_brief))




@pytest.fixture(params=["synthetic"])
def dataset(request):
    """Brief path and full panel for each dataset."""
    brief = request.getfixturevalue(f"{request.param}_brief")
    return request.param, brief, request.getfixturevalue(f"{request.param}_panel")


@pytest.fixture(scope="module")
def rules2():
    """Judge under rules 2: the hand examples worked out before rules 3, whose numbers stay
    those rules' own (rules 3's are the tests named ``under_rules_3``). Every judging function
    whose ``rules`` defaults to the current rules defaults to 2, and a brief parsed or a run
    initialised meanwhile is a rules-2 one."""
    import inspect

    from blinded import brief, fees, ladder, sim, sim_maker, vault

    with pytest.MonkeyPatch.context() as monkeypatch:
        _pin_rules_2(monkeypatch, inspect, (brief, vault), (sim, sim_maker, ladder, fees))
        yield


def _pin_rules_2(monkeypatch, inspect, parsers, modules) -> None:
    for module in parsers:
        monkeypatch.setattr(module, "RULES", 2)
    for module in modules:
        for obj in list(vars(module).values()):
            for fn in (obj, *vars(obj).values()) if inspect.isclass(obj) else (obj,):
                if not inspect.isfunction(fn) or fn.__module__ != module.__name__:
                    continue
                if fn.__defaults__ and "rules" in inspect.signature(fn).parameters:
                    names = [p.name for p in inspect.signature(fn).parameters.values()
                             if p.default is not inspect.Parameter.empty
                             and p.kind is not inspect.Parameter.KEYWORD_ONLY]  # fmt: skip
                    values = [2 if n == "rules" else v
                              for n, v in zip(names, fn.__defaults__, strict=True)]  # fmt: skip
                    monkeypatch.setattr(fn, "__defaults__", tuple(values))
                if fn.__kwdefaults__ and "rules" in fn.__kwdefaults__:
                    monkeypatch.setattr(fn, "__kwdefaults__", fn.__kwdefaults__ | {"rules": 2})
