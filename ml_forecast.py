"""Multi-method forecast of India's per-capita GENERATION to 2047, with backtest-weighted ensemble.
Generation-only: target is TWh generated. Capacity and consumption are never used.

Data: data/processed/national_panel.csv with columns
  year (fiscal year end, e.g. 2025), generation_twh, population_bn, urban_share (%), [gsdp_per_capita optional]
Run:  python src/ml_forecast.py --data data/processed/national_panel.csv
Test: python src/ml_forecast.py --demo      (SYNTHETIC data, checks the code only, not a result)
"""
import argparse, numpy as np, pandas as pd
from scipy.optimize import minimize, curve_fit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

COVID, TARGET, K_MAX = [2021], 2047, 8000.0   # COVID year is interpolated; K_MAX = ASSUMED per-capita ceiling (kWh)
SCEN = {"low": dict(urban=45, pop=1.57, gsdp_g=.04), "base": dict(urban=51, pop=1.6375, gsdp_g=.055),
        "high": dict(urban=57, pop=1.70, gsdp_g=.07)}          # ASSUMPTIONS: review before presenting
ZONE_SHARE = {"West": 1306, "South": 1081, "Central": 753, "North": 1087, "East": 771, "North-East": 89}

def load(d):
    d = d.sort_values("year").reset_index(drop=True)
    d["pc"] = d.generation_twh / d.population_bn                 # kWh per person (TWh / bn)
    d.loc[d.year.isin(COVID), "pc"] = np.nan
    d["pc"] = d["pc"].interpolate(); d["y"] = np.log(d.pc)
    if "gsdp_per_capita" in d: d["gsdp_pc_log"] = np.log(d.gsdp_per_capita)
    return d

# ---- Methods: each fit(train) returns predict(years, drivers_df) -> log per-capita generation ----
def fit_trend(tr):                                              # 1. log-linear trend (constant growth)
    b = np.polyfit(tr.year, tr.y, 1); return lambda yrs, drv: np.polyval(b, np.asarray(yrs))

def fit_holt(tr):                                               # 2. damped-trend exponential smoothing
    y = tr.y.values
    def run(p):
        a, bt, ph = p; l, b, sse = y[0], y[1] - y[0], 0.0
        for v in y[1:]:
            f = l + ph * b; sse += (v - f) ** 2; ln = a * v + (1 - a) * f
            b = bt * (ln - l) + (1 - bt) * ph * b; l = ln
        return sse, l, b
    r = minimize(lambda p: run(p)[0], [.5, .3, .9], bounds=[(.01, 1), (.01, 1), (.8, .98)])
    _, l, b = run(r.x); ph = r.x[2]
    return lambda yrs, drv: np.array([l + b * sum(ph ** i for i in range(1, int(k) + 1)) for k in np.asarray(yrs) - tr.year.iloc[-1]])

def fit_ridge(tr):                                              # 3. driver regression (time, urbanisation, GSDP/capita)
    cols = [c for c in ("year", "urban_share", "gsdp_pc_log") if c in tr]
    sc = StandardScaler().fit(tr[cols]); m = RidgeCV(alphas=np.logspace(-3, 2, 20)).fit(sc.transform(tr[cols]), tr.y)
    return lambda yrs, drv: m.predict(sc.transform(drv[cols]))

def fit_logistic(tr):                                           # 4. saturating S-curve with ceiling K_MAX
    t, pc = tr.year.values - 2000.0, tr.pc.values; f = lambda t, K, r, t0: K / (1 + np.exp(-r * (t - t0)))
    try:
        p, _ = curve_fit(f, t, pc, p0=[min(pc[-1] * 2.5, K_MAX * .9), .1, t[-1] + 10],
                         bounds=([pc[-1] * 1.2, .01, t[0]], [K_MAX, .5, t[-1] + 80]), maxfev=20000)
        return lambda yrs, drv: np.log(f(np.asarray(yrs) - 2000.0, *p))
    except Exception: return fit_trend(tr)
METHODS = {"trend": fit_trend, "damped_holt": fit_holt, "ridge_drivers": fit_ridge, "s_curve": fit_logistic}
# Tree models (random forest / boosting) are deliberately excluded: they cannot extrapolate beyond the training range.

def drivers(yrs, last, s):
    yrs = np.asarray(yrs); fr = (yrs - last.year) / (TARGET - last.year)
    d = pd.DataFrame({"year": yrs, "urban_share": last.urban_share + fr * (s["urban"] - last.urban_share)})
    if "gsdp_pc_log" in last.index: d["gsdp_pc_log"] = last.gsdp_pc_log + np.log(1 + s["gsdp_g"]) * (yrs - last.year)
    return d

def backtest(df, H=3, min_train=6):                             # rolling-origin: train to year t, predict t+1..t+H
    rows = []
    for o in range(min_train, len(df)):
        tr, te = df.iloc[:o], df.iloc[o:o + H]
        for n, fit in METHODS.items():
            for h, (yt, yp) in enumerate(zip(te.y, fit(tr)(te.year.values, te)), 1): rows.append((n, h, o, yp, yt))
    b = pd.DataFrame(rows, columns=["method", "h", "origin", "yp", "yt"]); b["ape"] = np.abs(np.exp(b.yp - b.yt) - 1); return b

def main(df):
    df = load(df); bt = backtest(df); mape = bt.groupby("method").ape.mean()
    w = (1 / mape ** 2) / (1 / mape ** 2).sum()                 # inverse-squared-error weights
    e = bt.assign(w=bt.method.map(w)).assign(wy=lambda x: x.yp * x.w).groupby(["origin", "h"]).agg(yp=("wy", "sum"), yt=("yt", "first"))
    ens_err = e.yp - e.yt; H = int(bt.h.max()); sd = ens_err.std() * np.sqrt((TARGET - df.year.iloc[-1]) / H)   # random-walk error growth
    print("Backtest MAPE of per-capita generation (rolling origin, 1-3 yrs ahead):"); print((mape * 100).round(2).to_string())
    print(f"Ensemble MAPE: {np.abs(np.exp(ens_err) - 1).mean() * 100:.2f}%   weights: {w.round(2).to_dict()}\n")
    last, yrs, out = df.iloc[-1], np.arange(df.year.iloc[-1] + 1, TARGET + 1), []
    for sc, s in SCEN.items():
        drv = drivers(yrs, last, s); p = {n: fit(df)(yrs, drv)[-1] for n, fit in METHODS.items()}
        ens = sum(w[n] * p[n] for n in p)
        tot = lambda lp: np.exp(lp) * s["pop"]
        out.append(dict(scenario=sc, **{n: round(tot(v)) for n, v in p.items()}, ensemble_TWh=round(tot(ens)),
                        p10=round(tot(ens - 1.28 * sd)), p90=round(tot(ens + 1.28 * sd)),
                        method_min=round(tot(min(p.values()))), method_max=round(tot(max(p.values())))))
    r = pd.DataFrame(out); print("2047 generation (TWh):"); print(r.to_string(index=False))
    base = r.set_index("scenario").loc["base", "ensemble_TWh"]; t = sum(ZONE_SHARE.values())
    print("\nBase-case zones (TWh):", {z: round(base * v / t) for z, v in ZONE_SHARE.items()})
    r.to_csv("outputs/ml_forecast_2047.csv", index=False); bt.to_csv("outputs/backtest.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--demo", action="store_true"); a = ap.parse_args()
    if a.demo:
        print("*** DEMO: synthetic data, tests the code only ***\n"); rng = np.random.default_rng(0); yr = np.arange(2011, 2026)
        pc = 1304 / 1.04 ** (2025 - yr) * np.exp(rng.normal(0, .012, len(yr))); pc[yr == 2021] *= .96
        pop = 1.40 / 1.01 ** (2025 - yr); d = pd.DataFrame(dict(year=yr, generation_twh=pc * pop, population_bn=pop,
            urban_share=31 + (yr - 2011) * .43, gsdp_per_capita=1e5 * 1.055 ** (yr - 2011)))
    else: d = pd.read_csv(a.data)
    main(d)
