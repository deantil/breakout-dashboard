"""Breakout Dashboard: scans stocks with the swing-trade rules from the Qullamaggie video.

Run: pip install -r requirements.txt  then  streamlit run breakout_dashboard.py
Daily data from Yahoo (free). Read-only: no orders. Educational, not financial advice.
"""
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
        (f"Liquidity >= ${p['min_dvol']}M/day", dvol >= p["min_dvol"] * 1e6, f"${dvol / 1e6:.0f}M", True),
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


def main():
    import streamlit as st
    import yfinance as yf
    st.set_page_config(page_title="Breakout Dashboard", layout="wide")
    st.title("Breakout Dashboard")
    sb = st.sidebar
    uni = sb.text_area("Stocks to scan", DEFAULT, height=120)
    acct = sb.number_input("Account $", 1000, 10_000_000, 10_000, 500)
    risk_pct = sb.slider("Risk per trade % of account", 0.25, 1.5, 1.0, 0.25)
    max_pos = sb.slider("Max position % of account", 10, 40, 30)
    min_run = sb.slider("Prior run at least %", 20, 100, 30)
    lbw = sb.slider("Look for the run within (weeks)", 6, 16, 12)
    cmin, cmax = sb.slider("Consolidation length (weeks)", 1, 12, (2, 8))
    depth = sb.slider("Max pullback depth %", 10, 40, 25)
    adr = sb.slider("Min ADR %", 2.0, 10.0, 4.0, 0.5)
    dvol = sb.slider("Min dollar volume ($M/day)", 1, 100, 10)
    near = sb.slider("'Near trigger' within %", 1.0, 8.0, 3.0, 0.5)
    gate = sb.checkbox("Market gate: only BUY if SPY is above its 50-day (my addition)", True)
    show = sb.multiselect("Show", ["BUY", "WATCH", "WAIT", "AVOID"], ["BUY", "WATCH"])
    syms = list(dict.fromkeys(uni.upper().replace(",", " ").split()))[:120]

    @st.cache_data(ttl=300)
    def load(tk):
        return yf.download(list(tk) + ["SPY"], period="1y", group_by="ticker", auto_adjust=True, progress=False, threads=True)

    data = load(tuple(syms))
    spy = data["SPY"].dropna()
    mkt = bool(spy.Close.iloc[-1] > spy.Close.rolling(50).mean().iloc[-1]) if len(spy) > 60 else True
    st.info(f"Market (SPY vs 50-day): {'ABOVE: breakouts have tailwind' if mkt else 'BELOW: video setups work best in uptrends, be selective'}")
    p = dict(acct=acct, risk_pct=risk_pct, max_pos=max_pos, min_run=min_run, lookback=lbw * 5, cmin=cmin, cmax=cmax,
             max_depth=depth, min_adr=adr, min_dvol=dvol, near=near, gate=(mkt or not gate))
    res = {}
    for t in syms:
        try:
            r = analyze(data[t].dropna(), p)
            if r:
                res[t] = r
        except Exception:
            pass
    df = pd.DataFrame({t: {k: v for k, v in r.items() if k not in ("checks", "fails")} for t, r in res.items()}).T
    if df.empty:
        st.warning("No stocks analysed. Check the tickers and your internet connection.")
        return
    df = df[df.verdict.str.split().str[0].str.strip(":").isin(show)]
    order = {"BUY": 0, "WATCH": 1, "WAIT": 2, "AVOID": 3}
    df = df.assign(o=df.verdict.str.split().str[0].str.strip(":").map(order)).sort_values(["o", "passed"], ascending=[True, False]).drop(columns="o")
    st.dataframe(df, use_container_width=True)
    st.caption("BUY = every core rule passes and price broke the consolidation high today. Shares are sized so a stop-out costs your risk % and the position never exceeds your max %.")
    if len(df):
        t = st.selectbox("Details", list(df.index))
        r = res[t]
        a, b = st.columns(2)
        a.subheader(f"{t}: {r['verdict']}")
        for name, ok, det, core in r["checks"]:
            a.write(f"{'PASS' if ok else 'FAIL'} - {name} {det}" + ("" if core else " (nice to have)"))
        b.subheader("Trade plan from the video")
        b.write(f"Entry near {r['trigger'] if r['status'] != 'Breakout' else r['price']}, stop {r['stop']} (risk {r['risk_pct']}%). "
                f"{r['shares']} shares, about ${r['cost']:,}, risking ${r['risk_usd']}.\n\n"
                f"After 3-5 days sell one third to one half and move the stop to breakeven. Trail the rest: exit on a close below the 10-day "
                f"(fast mover) or 20-day average. Reference prices: 3R {r['t3R']}, 5R {r['t5R']}.")
        d = data[t].dropna()[["Close"]].copy()
        d["10d"], d["20d"], d["50d"] = (d.Close.rolling(k).mean() for k in (10, 20, 50))
        st.line_chart(d.tail(120))


if __name__ == "__main__":
    main()
