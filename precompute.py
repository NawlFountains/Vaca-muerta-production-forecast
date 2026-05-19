"""
precompute.py
Run once locally to generate CSVs used by the Streamlit dashboard.
Mirrors the notebook logic exactly.
 
Usage:
    python precompute.py
"""
 
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
 
os.makedirs("data", exist_ok=True)
 
# ── Config ─────────────────────────────────────────────────────────────────────
DATA_URL = (
    "http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c"
    "/resource/b5b58cdc-9e07-41f9-b392-fb9ec68b0725"
    "/download/produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv"
)
 
TRAIN_CUTOFF      = "2024-12-31"
FORECAST_DEPOSITS = ["LOMA CAMPANA", "CRUZ DE LORENA", "ESTACION FERNANDEZ ORO", "SIERRAS BLANCAS"]
PROD_THRESHOLD    = 0.80
CONSEC_MONTHS     = 3
WATER_CUT_WINDOW  = 3
 
 
# ── Load & filter ──────────────────────────────────────────────────────────────
print("Loading dataset from URL (this may take a minute)...")
df = pd.read_csv(DATA_URL)
print(f"Loaded {len(df):,} total rows.")
 
# Mirror notebook exactly
df_neuquina = df[df["cuenca"] == "NEUQUINA"]
print(f"Filtered to {len(df_neuquina):,} Neuquina rows.")
 
 
 # df_deposits_locations
df_deposit_locations = df_neuquina[['coordenaday','coordenadax','idpozo','areayacimiento']]
df_deposit_locations = df_deposit_locations.groupby('areayacimiento').agg(
    lat=('coordenaday', 'mean'),
    lon=('coordenadax', 'mean')
).reset_index()
df_deposit_locations.rename(columns={'areayacimiento':'deposit'}, inplace=True)

df_deposit_locations.to_csv("data/deposit_locations.csv", index = False)

# ── df_monthly_prod — mirror notebook cells 42 & 44 ───────────────────────────
df_monthly_prod = df_neuquina[["anio", "mes", "areayacimiento", "prod_pet", "prod_gas", "prod_agua"]]
df_monthly_prod = df_monthly_prod.groupby(["anio", "mes", "areayacimiento"]).agg(
    prod_pet=("prod_pet", "sum"),
    prod_gas=("prod_gas", "sum"),
    prod_agua=("prod_agua", "sum"),
).reset_index()
 
df_monthly_prod["date"] = pd.to_datetime(
    df_monthly_prod["anio"].astype(str) + "-" + df_monthly_prod["mes"].astype(str).str.zfill(2) + "-01"
)
 
df_monthly_prod["water_cut"] = (
    df_monthly_prod["prod_agua"]
    / (df_monthly_prod["prod_pet"] + df_monthly_prod["prod_agua"]).replace(0, np.nan)
)
 
 
# ── 1. basin_monthly.csv ───────────────────────────────────────────────────────
print("\nComputing basin_monthly.csv...")
df_monthly_prod.to_csv("data/basin_monthly.csv", index=False)
print(f"  Saved {len(df_monthly_prod):,} rows.")
 
 
# ── 2. company_production.csv ──────────────────────────────────────────────────
print("Computing company_production.csv...")
company_production = (
    df_neuquina.groupby("empresa")[["prod_pet", "prod_gas"]]
    .sum()
    .reset_index()
    .sort_values("prod_pet", ascending=False)
)
company_production.to_csv("data/company_production.csv", index=False)
print(f"  Saved {len(company_production):,} rows.")
 
 
# ── Decline curve functions — mirror notebook cells 48 & 53 ───────────────────
def exponential_decline(t, qi, Di):
    return qi * np.exp(-Di * t)
 
def hyperbolic_decline(t, qi, Di, b):
    return qi / (1 + b * Di * t) ** (1 / b)
 
def fit_decline_curve(deposit_df, resource="prod_pet"):
    d = deposit_df[deposit_df[resource] > 0].copy()
 
    peak_pos = d[resource].argmax()
    d = d.iloc[peak_pos:].reset_index(drop=True)
 
    if len(d) < 10:
        print(f"  Not enough data after peak ({len(d)} rows), skipping")
        return None
 
    d["t"] = range(len(d))
    t = d["t"].values
    q = d[resource].values
 
    try:
        popt_hyp, _ = curve_fit(
            hyperbolic_decline, t, q,
            p0=[q[0], 0.1, 1.0],
            bounds=([0, 0, 0], [np.inf, 10, 2]),
            maxfev=10000,
        )
        q_pred_hyp = hyperbolic_decline(t, *popt_hyp)
 
        popt_exp, _ = curve_fit(
            exponential_decline, t, q,
            p0=[q[0], 0.1],
            bounds=([0, 0], [np.inf, 10]),
            maxfev=10000,
        )
        q_pred_exp = exponential_decline(t, *popt_exp)
 
        return d, t, q, popt_hyp, q_pred_hyp, popt_exp, q_pred_exp
 
    except RuntimeError:
        print("  Could not fit curve")
        return None
 
 
# ── 3. forecast_arps.csv — mirror notebook cell 59 ────────────────────────────
print("\nFitting Arps decline curves...")
 
arps_mapes  = []
arps_frames = []
 
for deposit in FORECAST_DEPOSITS:
    d_dep = df_monthly_prod[df_monthly_prod["areayacimiento"] == deposit].sort_values("date").copy()
 
    train = d_dep[d_dep["date"] <= TRAIN_CUTOFF]
    test  = d_dep[d_dep["date"] >  TRAIN_CUTOFF]
 
    result = fit_decline_curve(train, resource="prod_pet")
    if result is None:
        print(f"  {deposit} | Could not fit")
        continue
 
    d, t, q, popt_hyp, q_pred_hyp, popt_exp, q_pred_exp = result
 
    t_test     = np.arange(len(d), len(d) + len(test))
    q_test_hyp = hyperbolic_decline(t_test, *popt_hyp)
 
    q_actual_test = test["prod_pet"].values
    mae_hyp  = mean_absolute_error(q_actual_test, q_test_hyp)
    mape_hyp = mae_hyp / train["prod_pet"].mean() * 100
 
    print(f"  {deposit} | Arps MAPE = {mape_hyp:.1f}%")
    arps_mapes.append({"deposit": deposit, "mape": round(mape_hyp, 1)})
 
    # d["date"] contains the post-peak train dates directly
    arps_out = pd.DataFrame({
        "ds":      pd.concat([d["date"], test["date"]]).values,
        "deposit": deposit,
        "yhat":    np.concatenate([q_pred_hyp, q_test_hyp]),
        "actual":  np.concatenate([d["prod_pet"].values, q_actual_test]),
    })
    arps_frames.append(arps_out)
 
if arps_frames:
    pd.concat(arps_frames, ignore_index=True).to_csv("data/forecast_arps.csv", index=False)
    pd.DataFrame(arps_mapes).to_csv("data/arps_mapes.csv", index=False)
    print("  Saved Arps forecasts.")
 
 
# ── 4. forecast_prophet.csv — mirror notebook cell 65 ─────────────────────────
print("\nRunning Prophet forecasts...")
 
prophet_frames = []
prophet_mapes  = []
 
for deposit in FORECAST_DEPOSITS:
    d_dep = df_monthly_prod[df_monthly_prod["areayacimiento"] == deposit].sort_values("date").copy()
 
    df_prophet = d_dep[["date", "prod_pet"]].rename(columns={"date": "ds", "prod_pet": "y"})
 
    train = df_prophet[df_prophet["ds"] <= TRAIN_CUTOFF]
    test  = df_prophet[df_prophet["ds"] >  TRAIN_CUTOFF]
 
    model = Prophet(seasonality_mode="multiplicative")
    model.fit(train)
 
    future   = model.make_future_dataframe(periods=len(test) + 12, freq="MS")
    forecast = model.predict(future)
 
    test_forecast = forecast[forecast["ds"].isin(test["ds"])]
    mae      = mean_absolute_error(test["y"].values, test_forecast["yhat"].values)
    avg_prod = train["y"].mean()
    mape     = mae / avg_prod * 100
 
    print(f"  {deposit}: Prophet MAPE = {mape:.1f}%")
    prophet_mapes.append({"deposit": deposit, "mape": round(mape, 1)})
 
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out["deposit"] = deposit
    out = out.merge(df_prophet.rename(columns={"y": "actual"}), on="ds", how="left")
    prophet_frames.append(out)
 
pd.concat(prophet_frames, ignore_index=True).to_csv("data/forecast_prophet.csv", index=False)
pd.DataFrame(prophet_mapes).to_csv("data/prophet_mapes.csv", index=False)
print(f"  Saved Prophet forecasts for {len(prophet_frames)} deposits.")
 
 
# ── 5. underperformance.csv — mirror notebook cell 72 ─────────────────────────
print("\nComputing underperformance flags...")
 
underperf_frames = []
 
for deposit in FORECAST_DEPOSITS:
    d_dep = df_monthly_prod[df_monthly_prod["areayacimiento"] == deposit].sort_values("date").copy()
 
    # Filter to established production (>= 10% of peak) — mirrors notebook
    peak_prod = d_dep["prod_pet"].max()
    d_dep = d_dep[d_dep["prod_pet"] >= peak_prod * 0.10]
 
    df_prophet = d_dep[["date", "prod_pet"]].rename(columns={"date": "ds", "prod_pet": "y"})
 
    # Fit on full history — mirrors notebook comment "fit on full history this time"
    model = Prophet(seasonality_mode="multiplicative")
    model.fit(df_prophet)
 
    future   = model.make_future_dataframe(periods=0, freq="MS")
    forecast = model.predict(future)
 
    df_eval = d_dep[["date", "prod_pet", "water_cut"]].copy()
    df_eval = df_eval.merge(
        forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "prod_forecast"}),
        on="date",
    )
 
    # Signal 1: consecutive months below threshold
    df_eval["below_threshold"] = df_eval["prod_pet"] < df_eval["prod_forecast"] * PROD_THRESHOLD
    df_eval["consec_below"] = (
        df_eval["below_threshold"]
        .rolling(CONSEC_MONTHS)
        .sum()
        .fillna(0) == CONSEC_MONTHS
    )
 
    # Signal 2: rising water cut over rolling window
    df_eval["water_cut_rising"] = (
        df_eval["water_cut"]
        .rolling(WATER_CUT_WINDOW)
        .apply(lambda x: x.iloc[-1] > x.iloc[0])
        .fillna(False)
        .astype(bool)
    )
 
    # Final flag: both signals active
    df_eval["underperforming"] = df_eval["consec_below"] & df_eval["water_cut_rising"]
    df_eval["deposit"] = deposit
 
    flagged = df_eval["underperforming"].sum()
    print(f"  {deposit}: {flagged} flagged months")
 
    underperf_frames.append(
        df_eval[["date", "deposit", "prod_pet", "prod_forecast", "water_cut", "below_threshold", "underperforming"]]
    )
 
pd.concat(underperf_frames, ignore_index=True).to_csv("data/underperformance.csv", index=False)
print("  Saved underperformance flags.")
 
 
# ── Done ───────────────────────────────────────────────────────────────────────
print("\nAll done! Files saved to /data:")
for f in sorted(os.listdir("data")):
    size_kb = os.path.getsize(f"data/{f}") / 1024
    print(f"  {f:<30} ({size_kb:.1f} KB)")