"""Breakout Dashboard: scans stocks with the swing-trade rules from the Qullamaggie video.

Run: pip install -r requirements.txt  then  streamlit run breakout_dashboard.py
Daily data from Yahoo (free). Read-only: no orders. Educational, not financial advice.
"""
import math
import datetime as dt
import numpy as np
import pandas as pd

DEFAULT = ("NVDA AMD TSLA PLTR SMCI AVGO MU ARM CRWD NET SNOW DDOG SHOP COIN HOOD SOFI AFRM UPST RBLX APP "
           "RKLB IONQ HIMS CELH ELF ONON DKNG ROKU TTD MELI SE NU MSTR MARA RIOT AXON CAVA ANF DECK UBER "
           "ABNB DASH ENPH FSLR RIVN CVNA OPEN SOUN RGTI U W LULU")


def analyze(df, p):
    """Apply the video's setup rules to one daily OHLCV frame. Returns a dict or None."""
    if len(df) < 120:
        return None
    c, h, l, v = (df[k].values.astype(float) for k in ("Close", "High", "Low", "Volume"))
    n = len(df)
    sma = lambda k: pd.Series(c).rolling(k).mean().values
    s10, s20, s50 = sma(10), sma(20), sma(50)
    lb = p["lookback"]
    w = c[-lb - 1:-1]
    run = w / np.minimum.accumulate(w) - 1
    k = int(run.argmax())
    pk = n - 1 - lb + k
    since, move = n - 1 - pk, run[k] * 100
    if since < 3:
        return None
    ch = h[pk:-1].max()
    cl = l[pk + 1:-1].min()
    depth = (ch - cl) / ch * 100
    adr = float(np.mean(h[-20:] / l[-20:] - 1) * 100)
    dvol = float(np.mean(c[-20:] * v[-20:]))
    rng = h / l - 1
    close, trig = c[-1], round(ch + 0.01, 2)
    brk = close > ch
    near = (not brk) and close >= ch * (1 - p["near"] / 100)
    status = "Breakout" if brk else "Near trigger" if near else "Forming"
    entry = close if brk else trig
    stop = l[-1] if brk else float(l[-3:].min())
    risk = (entry - stop) / entry * 100
    vr = v[-1] / np.mean(v[-51:-1])
    checks = [
        (f"Prior run >= {p['min_run']}%", move >= p["min_run"], f"{move:.0f}%", True),
        (f"Consolidation {p['cmin']}-{p['cmax']} weeks", p["cmin"] * 5 <= since <= p["cmax"] * 5, f"{since / 5:.1f} wks", True),
        (f"Orderly pullback <= {p['max_depth']}%", depth <= p["max_depth"], f"{depth:.0f}%", True),
        ("10 > 20 > 50 day MAs rising, price holds 20-day",
         s10[-1] > s20[-1] > s50[-1] and s10[-1] > s10[-6] and s20[-1] > s20[-6] and close >= s20[-1] * 0.98, "", True),
        (f"ADR >= {p['min_adr']}%", adr >= p["min_adr"], f"{adr:.1f}%", True),
        (f"Liquidity >= {p['min_dvol']}M USD/day", dvol >= p["min_dvol"] * 1e6, f"{dvol / 1e6:.0f}M USD", True),
        ("Stop distance within ADR", 0 < risk <= adr, f"{risk:.1f}% vs {adr:.1f}%", True),
        ("Range tightening (5d < 20d)", rng[-6:-1].mean() < rng[-21:-1].mean(), "", False),
        ("Volume dries up in base", np.mean(v[-11:-1]) < np.mean(v[-51:-1]), "", False),
    ]
    if brk:
        checks.append(("Breakout volume >= 1.5x average", vr >= 1.5, f"{vr:.1f}x", False))
    fails = [x[0] for x in checks if x[3] and not x[1]]
    if not fails and brk:
        verdict = "BUY" if vr >= 1.2 else "BUY (volume weak: half size)"
        if p.get("gate") is False:
            verdict = "WATCH (market below 50-day)"
    elif not fails:
        verdict = "WATCH: alert at trigger" if near else "WAIT: setup forming"
    else:
        verdict = "WATCH (1 rule fails)" if len(fails) == 1 else "AVOID"
    ps = entry - stop
    sh = int(min(p["acct"] * p["risk_pct"] / 100 / ps, p["acct"] * p["max_pos"] / 100 / entry)) if ps > 0 else 0
    return dict(verdict=verdict, status=status, passed=f"{sum(x[1] for x in checks)}/{len(checks)}", price=round(close, 2),
                trigger=trig, stop=round(stop, 2), risk_pct=round(risk, 1), shares=sh, cost=round(sh * entry),
                risk_usd=round(sh * ps), t3R=round(entry + 3 * ps, 2), t5R=round(entry + 5 * ps, 2),
                adr=round(adr, 1), move=round(move), checks=checks, fails=fails)



NASDAQ = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
ORDER = {"BUY": 0, "WATCH": 1, "WAIT": 2, "AVOID": 3}


def parse_symbols(text, col):
    import io
    d = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    d = d[(d["Test Issue"] == "N") & (d["ETF"] == "N")]
    return [x for x in d[col].dropna().astype(str) if x.isalpha() and len(x) <= 5 and not (len(x) == 5 and x[-1] in "WRU")]


def rsi(c, n=14):
    d = np.diff(c)
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n).mean().iloc[-1]
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n).mean().iloc[-1]
    return 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)


def finish(checks, status, price, entry, stop, p, brk=True, near=False, shares=None, **extra):
    fails = [x[0] for x in checks if x[3] and not x[1]]
    if not fails:
        verdict = "BUY" if brk else "WATCH: alert at trigger" if near else "WAIT: not at trigger"
    else:
        verdict = "WATCH (1 rule fails)" if len(fails) == 1 else "AVOID"
    if verdict == "BUY" and not p["gate"]:
        verdict = "WATCH (market below 50-day)"
    ps = entry - stop
    if shares is None:
        shares = int(min(p["acct"] * p["risk_pct"] / 100 / ps, p["acct"] * p["max_pos"] / 100 / entry)) if ps > 0 else 0
    return dict(verdict=verdict, status=status, passed=f"{sum(x[1] for x in checks)}/{len(checks)}", price=round(price, 2),
                entry=round(entry, 2), stop=round(stop, 2), risk_pct=round(ps / entry * 100, 1), shares=shares,
                cost=round(shares * entry), risk_usd=round(shares * max(ps, 0)), checks=checks, fails=fails, **extra)


def analyze_trend(df, p):
    """Generic systematic trend/breakout rules (Nirvana-style approximation, NOT Nirvana's proprietary rules)."""
    if len(df) < 205:
        return None
    c, h, l, v = (df[k].values.astype(float) for k in ("Close", "High", "Low", "Volume"))
    s50, s200 = pd.Series(c).rolling(50).mean().values, pd.Series(c).rolling(200).mean().values
    lb = p["don"]
    hh, close = h[-lb - 1:-1].max(), c[-1]
    brk, near = close > hh, close >= hh * (1 - p["near"] / 100)
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    atr = tr[-14:].mean()
    entry = close if brk else round(hh + 0.01, 2)
    stop = entry - p["atr_x"] * atr
    risk, r, vr = (entry - stop) / entry * 100, rsi(c), v[-1] / np.mean(v[-51:-1])
    adr = float(np.mean(h[-20:] / l[-20:] - 1) * 100)
    checks = [("Close above 200-day, 50-day above 200-day", close > s200[-1] and s50[-1] > s200[-1], "", True),
              (f"RSI between {p['rsi_lo']} and {p['rsi_hi']}", p["rsi_lo"] <= r <= p["rsi_hi"], f"{r:.0f}", True),
              (f"Stop (ATR-based) within {p['max_risk']}%", risk <= p["max_risk"], f"{risk:.1f}%", True),
              ("Not extended: <5% above 50-day", close < s50[-1] * 1.25, "", False),
              ("Volume >= 1.2x average", vr >= 1.2, f"{vr:.1f}x", False)]
    return finish(checks, "Breakout" if brk else "Near trigger" if near else "Forming", close, entry, stop, p, brk, near,
                  trigger=round(hh + 0.01, 2), atr=round(atr, 2), adr=round(adr, 1),
                  exit10=round(l[-10:].min(), 2))


def analyze_compounder(df, p):
    """Long-term trend/relative-strength holding rules (Birbia-style PLACEHOLDER: the Birbia strategy itself wasn't visible)."""
    if len(df) < 205:
        return None
    c, h = df.Close.values.astype(float), df.High.values.astype(float)
    s50, s200 = pd.Series(c).rolling(50).mean().values, pd.Series(c).rolling(200).mean().values
    close, hi52 = c[-1], h.max()
    ret6 = (close / c[-127] - 1) * 100
    dd = (1 - c / np.maximum.accumulate(c)).max() * 100
    vol = np.std(np.diff(c) / c[:-1]) * np.sqrt(252) * 100
    checks = [("Close above rising 200-day", close > s200[-1] and s200[-1] > s200[-21], "", True),
              ("50-day above 200-day", s50[-1] > s200[-1], "", True),
              (f"Within {p['max_off']}% of 52-week high", close >= hi52 * (1 - p["max_off"] / 100), f"{(1 - close / hi52) * 100:.0f}% off", True),
              ("6-month return beats SPY", ret6 > p["spy6"], f"{ret6:.0f}% vs SPY {p['spy6']:.0f}%", True),
              (f"Worst drawdown in last year <= {p['max_dd']}%", dd <= p["max_dd"], f"{dd:.0f}%", True),
              (f"Volatility <= {p['max_vol']}% a year", vol <= p["max_vol"], f"{vol:.0f}%", False)]
    sh = int(p["acct"] / p["npos"] / close)
    return finish(checks, "Trend intact" if close > s200[-1] else "Broken", close, close, s200[-1], p, True, False, sh,
                  ret6=round(ret6), off_high=round((1 - close / hi52) * 100, 1), vol=round(vol))


def render(st, res, data, show, key, plan):
    if not res:
        st.warning("No stocks analysed.")
        return
    df = pd.DataFrame({t: {k: v for k, v in r.items() if k not in ("checks", "fails")} for t, r in res.items()}).T
    df["g"] = df.verdict.str.split().str[0].str.strip(":")
    df = df[df.g.isin(show)].assign(o=lambda d: d.g.map(ORDER)).sort_values(["o", "passed"], ascending=[True, False]).drop(columns=["o", "g"])
    st.caption(f"{len(res)} stocks analysed, {len(df)} shown")
    st.dataframe(df, use_container_width=True)
    if len(df):
        t = st.selectbox("Details", list(df.index), key=key)
        r, (a, b) = res[t], st.columns(2)
        a.subheader(f"{t}: {r['verdict']}")
        for name, ok, det, core in r["checks"]:
            a.write(f"{'PASS' if ok else 'FAIL'} - {name} {det}" + ("" if core else " (nice to have)"))
        b.subheader("Trade plan")
        b.write(plan(r))
        d = data[t][["Close"]].copy()
        d["10d"], d["20d"], d["50d"] = (d.Close.rolling(k).mean() for k in (10, 20, 50))
        st.line_chart(d.tail(120))


def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def put_pop(S, BE, T, iv, r=0.04):
    """Model probability the stock finishes above breakeven (profit), using implied volatility."""
    if T <= 0 or iv <= 0 or BE <= 0:
        return float("nan")
    return _N((math.log(S / BE) + (r - iv * iv / 2) * T) / (iv * math.sqrt(T)))


def put_delta_abs(S, K, T, iv, r=0.04):
    if T <= 0 or iv <= 0:
        return float("nan")
    return _N(-(math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T)))


def fetch_opts(sym, price, dmin, dmax, kind, diag):
    import yfinance as yf
    t, today = yf.Ticker(sym), dt.date.today()
    try:
        exps = list(t.options)
    except Exception as e:
        diag[sym] = f"expiry list failed: {str(e)[:80]}"
        return []
    if not exps:
        diag[sym] = "Yahoo returned no option expirations (blocked from this server, or not optionable)"
        return []
    use = [e for e in exps if dmin <= (dt.date.fromisoformat(e) - today).days <= dmax]
    if not use:
        diag[sym] = f"no expiry between {dmin} and {dmax} days (first available: {exps[:3]})"
        return []
    earn = None
    try:
        cal = t.calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        e = (ed[0] if isinstance(ed, (list, tuple)) else ed) if ed else None
        earn = e.date() if isinstance(e, dt.datetime) else e
    except Exception:
        pass
    rows = []
    f = lambda x: 0.0 if x is None or x != x else float(x)
    for exp in use:
        try:
            ch = t.option_chain(exp)
        except Exception as e:
            diag[sym] = f"chain failed: {str(e)[:80]}"
            continue
        d = dt.date.fromisoformat(exp)
        for r in (ch.puts if kind == "put" else ch.calls).itertuples():
            rows.append(dict(symbol=sym, price=price, expiry=d, dte=(d - today).days, strike=f(r.strike), bid=f(r.bid), ask=f(r.ask),
                             iv=f(r.impliedVolatility), oi=int(f(r.openInterest)), earnings=earn))
    diag.setdefault(sym, f"{len(rows)} contracts")
    return rows


def rank_opts(rows, p, kind):
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d = d[(d.bid > 0) & (d.dte > 0) & (d.iv > 0)]
    d = (d[d.strike < d.price] if kind == "put" else d[d.strike > d.price]).copy()
    if d.empty:
        return d
    d["mid"] = (d.bid + d.ask) / 2
    d["spread%"] = (d.ask - d.bid) / d.mid * 100
    T = (d.dte / 365).values
    if kind == "put":
        d["cash"] = d.strike * 100
        d["return%"] = d.bid / d.strike * 100
        d["cushion%"] = (d.price - d.strike) / d.price * 100
        d["breakeven"] = d.strike - d.bid
        d["delta"] = [put_delta_abs(r.price, r.strike, t, r.iv) for r, t in zip(d.itertuples(), T)]
        d["pop%"] = [put_pop(r.price, r.breakeven, t, r.iv) * 100 for r, t in zip(d.itertuples(), T)]
        d["loss if -20% (USD)"] = (d.strike - d.price * 0.8 - d.bid).clip(lower=0) * 100
    else:
        d1 = lambda r, t: (math.log(r.price / r.strike) + (0.04 + r.iv ** 2 / 2) * t) / (r.iv * math.sqrt(t))
        d["cash"] = d.price * 100
        d["return%"] = d.bid / d.price * 100
        d["cushion%"] = (d.strike - d.price) / d.price * 100
        d["breakeven"] = d.price - d.bid
        d["delta"] = [_N(d1(r, t)) for r, t in zip(d.itertuples(), T)]
        d["pop%"] = [(1 - _N(d1(r, t) - r.iv * math.sqrt(t))) * 100 for r, t in zip(d.itertuples(), T)]
        d["if called (%)"] = (d.strike - d.price + d.bid) / d.price * 100
        d["loss if -20% (USD)"] = (d.price * 0.2 - d.bid).clip(lower=0) * 100
    d["per30%"] = d["return%"] * 30 / d.dte
    today = dt.date.today()
    d["earn"] = d.apply(lambda r: isinstance(r.earnings, dt.date) and today <= r.earnings <= r.expiry, axis=1)
    ok = ((d["pop%"] >= p["min_pop"]) & (d["per30%"] >= p["min_ret"]) & (d.delta <= p["max_delta"]) & (d.oi >= p["min_oi"])
          & (d["spread%"] <= p["max_spread"]) & (d.cash <= p["acct"] * p["max_pos"] / 100) & ~d.earn)
    cols = ["symbol", "price", "expiry", "dte", "strike", "bid", "pop%", "delta", "per30%", "return%", "cushion%", "breakeven",
            "if called (%)", "cash", "loss if -20% (USD)", "oi", "spread%"]
    return d[ok].sort_values(["pop%", "per30%"], ascending=False)[[c for c in cols if c in d.columns]].round(2)


def main():
    import requests
    import streamlit as st
    import yfinance as yf
    st.set_page_config(page_title="Stock Dashboard", layout="wide")
    st.title("Stock Dashboard: three rule sets")
    sb = sb_ = st.sidebar
    mode = sb.radio("Universe", ["Starter list (55)", "Full US market (slow)", "My own list"])
    own = sb.text_area("Tickers (used by 'My own list')", DEFAULT, height=90)
    acct = sb.number_input("Account USD", 1000, 10_000_000, 10_000, 500)
    risk_pct = sb.slider("Risk per trade % of account", 0.25, 1.5, 1.0, 0.25)
    max_pos = sb.slider("Max position % of account", 10, 40, 30)
    min_price = sb.slider("Min share price USD", 1, 50, 5)
    min_dvol = sb.slider("Min dollar volume (M USD/day)", 1, 100, 10)
    gate = sb.checkbox("Market gate: BUY only if SPY above 50-day (my addition)", True)
    show = sb.multiselect("Show", ["BUY", "WATCH", "WAIT", "AVOID"], ["BUY", "WATCH"])

    @st.cache_data(ttl=86400)
    def universe():
        out = []
        for url, col in ((NASDAQ, "Symbol"), (OTHER, "ACT Symbol")):
            out += parse_symbols(requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).text, col)
        return sorted(set(out))

    @st.cache_data(ttl=3600, show_spinner="Downloading and pre-filtering prices (the full market takes several minutes)...")
    def load(tk, mp, md):
        keep = {}
        for i in range(0, len(tk), 150):
            ch = list(tk[i:i + 150])
            try:
                d = yf.download(ch, period="1y", group_by="ticker", auto_adjust=True, progress=False, threads=True)
            except Exception:
                continue
            for t in ch:
                try:
                    x = d[t].dropna()
                except Exception:
                    continue
                if len(x) >= 120 and x.Close.iloc[-1] >= mp and (x.Close.tail(20) * x.Volume.tail(20)).mean() >= md * 1e6:
                    keep[t] = x
        return keep

    if mode.startswith("Full"):
        try:
            tk = tuple(universe())
        except Exception as e:
            st.error(f"Could not load the US ticker list: {e}")
            return
    else:
        tk = tuple(dict.fromkeys((DEFAULT if mode.startswith("Starter") else own).upper().replace(",", " ").split()))[:300]
    data = load(tk + ("SPY",), min_price, min_dvol)
    spy = data.get("SPY")
    if spy is None:
        st.error("Could not download prices (Yahoo may be busy). Reload in a minute.")
        return
    mkt = bool(spy.Close.iloc[-1] > spy.Close.rolling(50).mean().iloc[-1])
    st.info(f"{len(data) - 1} stocks passed the price/liquidity pre-filter out of {len(tk)}. Market (SPY vs 50-day): "
            f"{'ABOVE: tailwind' if mkt else 'BELOW: be selective'}")
    base = dict(acct=acct, risk_pct=risk_pct, max_pos=max_pos, gate=(mkt or not gate))
    stocks = {t: d for t, d in data.items() if t != "SPY"}
    t1, t2, t3, t4 = st.tabs(["1. Qullamaggie breakout", "2. Nirvana-style trend", "3. Birbia-style compounder", "4. Options income (puts)"])

    with t1:
        st.caption("Rules from the Financial Wisdom video on Kristjan Qullamaggie's breakout setup.")
        with st.expander("Rules"):
            c = st.columns(4)
            p = dict(base, min_run=c[0].slider("Prior run >= %", 20, 100, 30), lookback=5 * c[1].slider("Run within (weeks)", 6, 16, 12),
                     max_depth=c[2].slider("Max pullback %", 10, 40, 25), min_adr=c[3].slider("Min ADR %", 2.0, 10.0, 4.0, 0.5),
                     near=c[0].slider("Near trigger %", 1.0, 8.0, 3.0, 0.5), min_dvol=min_dvol)
            p["cmin"], p["cmax"] = c[1].slider("Consolidation weeks", 1, 12, (2, 8))
        res = {}
        for t, d in stocks.items():
            try:
                r = analyze(d, p)
                if r:
                    res[t] = r
            except Exception:
                pass
        render(st, res, stocks, show, "k1", lambda r: (
            f"Entry near {r['trigger'] if r['status'] != 'Breakout' else r['price']}, stop {r['stop']} (risk {r['risk_pct']}%). "
            f"{r['shares']} shares, about {r['cost']:,} USD, risking {r['risk_usd']} USD. After 3-5 days sell one third to one half and "
            f"move the stop to breakeven; trail the rest, exiting on a close below the 10-day (fast) or 20-day average. 3R {r['t3R']}, 5R {r['t5R']}."))

    with t2:
        st.caption("A generic systematic trend/breakout rule set in the style of Nirvana Systems' OmniTrader. These are NOT Nirvana's proprietary rules, which are not public.")
        with st.expander("Rules"):
            c = st.columns(4)
            p = dict(base, don=c[0].slider("Breakout of N-day high", 10, 55, 20), atr_x=c[1].slider("Stop = ATR multiple", 1.0, 4.0, 2.0, 0.5),
                     max_risk=c[2].slider("Max stop distance %", 3, 20, 10), near=c[3].slider("Near trigger %", 0.5, 5.0, 2.0, 0.5),
                     rsi_lo=c[0].slider("RSI min", 30, 60, 50), rsi_hi=c[1].slider("RSI max", 65, 90, 80))
        res = {t: r for t, d in stocks.items() for r in [analyze_trend(d, p)] if r}
        render(st, res, stocks, show, "k2", lambda r: (
            f"Buy on a close above the {p['don']}-day high ({r['trigger']}); stop {r['stop']} ({p['atr_x']} x ATR, risk {r['risk_pct']}%). "
            f"{r['shares']} shares, about {r['cost']:,} USD, risking {r['risk_usd']} USD. Exit on a close below the 10-day low ({r['exit10']}) "
            f"or the stop, whichever comes first."))

    with t3:
        st.caption("PLACEHOLDER: birbia.com's pricing page only showed a 'compounding with AI / financial independence' community, with no visible strategy. "
                   "This tab is a plain long-term trend and relative-strength holding screen. Paste Birbia's actual rules and I will replace it.")
        with st.expander("Rules"):
            c = st.columns(4)
            p = dict(base, max_off=c[0].slider("Max % below 52-week high", 5, 40, 15), max_dd=c[1].slider("Max 1-year drawdown %", 15, 60, 35),
                     max_vol=c[2].slider("Max volatility %", 30, 120, 60), npos=c[3].slider("Number of positions", 3, 20, 10),
                     spy6=float((spy.Close.iloc[-1] / spy.Close.iloc[-127] - 1) * 100))
        res = {t: r for t, d in stocks.items() for r in [analyze_compounder(d, p)] if r}
        render(st, res, stocks, show, "k3", lambda r: (
            f"Hold about {r['shares']} shares (about {r['cost']:,} USD, one of {p['npos']} equal positions). Review monthly. "
            f"Exit on a close below the 200-day average ({r['stop']}). This is slow compounding, not income or short-term trading."))

    with t4:
        st.caption("Options income, ranked by the model's probability of profit (from implied volatility). Yahoo data, about 15 minutes delayed: confirm prices in thinkorswim before trading.")
        kind_l = st.radio("Strategy", ["Cash-secured puts (you hold cash, get paid to wait)", "Covered calls (you already own 100 shares)"], horizontal=True)
        kind = "put" if kind_l.startswith("Cash") else "call"
        src = st.radio("Which stocks to scan", ["Type my own", "Most liquid stocks I can afford, from the universe scan above"], horizontal=True)
        afford = acct * max_pos / 100 / 100
        if src.startswith("Type"):
            wl = st.text_area("Tickers (about 25 works best)", "F SOFI T PFE INTC SNAP AAL RIVN KEY HBAN WBD GRAB", height=60)
            tk4 = wl.upper().replace(",", " ").split()[:60]
        else:
            nmax = st.slider("How many stocks", 10, 60, 25)
            liq = {t: float((d.Close.tail(20) * d.Volume.tail(20)).mean()) for t, d in stocks.items() if d.Close.iloc[-1] <= afford}
            tk4 = sorted(liq, key=liq.get, reverse=True)[:nmax]
            st.caption(f"With your account and position limit, one contract can use at most {acct * max_pos / 100:,.0f} USD, so only stocks priced up to {afford:,.0f} USD qualify. "
                       f"{len(liq)} stocks in your universe do. Scanning the {len(tk4)} most liquid: {' '.join(tk4)}")
        c = st.columns(4)
        op = dict(acct=acct, max_pos=max_pos, min_pop=c[0].slider("Min probability %", 50, 95, 75),
                  min_ret=c[1].slider("Min return per 30 days %", 0.3, 5.0, 0.8, 0.1), max_delta=c[2].slider("Max delta", 0.05, 0.5, 0.30, 0.05))
        dmin, dmax = c[3].slider("Days to expiry", 7, 60, (25, 50))
        c = st.columns(2)
        op["min_oi"], op["max_spread"] = c[0].number_input("Min open interest", 0, 100000, 100), c[1].slider("Max bid/ask spread %", 1, 30, 10)
        if st.button("Scan options now") and tk4:
            rows, diag, bar = [], {}, st.progress(0.0)
            for i, sym in enumerate(tk4):
                try:
                    px = float(stocks[sym].Close.iloc[-1]) if sym in stocks else float(yf.Ticker(sym).history(period="5d").Close.iloc[-1])
                    rows += fetch_opts(sym, px, dmin, dmax, kind, diag)
                except Exception as e:
                    diag[sym] = f"error: {str(e)[:80]}"
                bar.progress((i + 1) / len(tk4))
            st.session_state["opt"] = (rank_opts(rows, op, kind), diag, len(rows), kind)
        if "opt" in st.session_state:
            res4, diag, n4, k4 = st.session_state["opt"]
            st.caption(f"{n4} contracts checked, {len(res4)} pass your filters (no earnings in the window, size within your limit). Sorted by probability, then return.")
            if n4 == 0:
                st.error("No option data came back. Open 'What happened per stock' below: if it says Yahoo returned no expirations, Yahoo is blocking options requests from this website's server.")
            if len(res4):
                st.markdown("**Best trades right now** (highest probability that also meets your minimum return):")
                for _, r in res4.head(3).iterrows():
                    what = f"sell the {r['expiry']} {r['strike']} put" if k4 == "put" else f"sell the {r['expiry']} {r['strike']} call"
                    st.success(f"{r['symbol']}: {what} for {r['bid']} a share. About {r['pop%']:.0f}% chance of profit, {r['per30%']:.2f}% per 30 days, "
                               f"{r['cash']:,.0f} USD tied up, about {r['loss if -20% (USD)']:,.0f} USD loss if the stock falls 20%.")
                st.dataframe(res4, use_container_width=True, hide_index=True)
                st.download_button("Download this list (CSV)", res4.to_csv(index=False), "options_scan.csv")
            with st.expander("What happened per stock"):
                st.write(diag)
        st.info("Higher probability means a smaller premium, and the model ignores sudden gaps. A 90% win rate can still lose money if the few losses are large: "
                "check the 'loss if -20%' column. For covered calls, 'probability' means the chance you keep your shares and the premium. "
                "A premium above about 4% per 30 days usually signals a stock the market expects to move sharply.")


if __name__ == "__main__":
    main()
