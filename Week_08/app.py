import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Moneyball | IPL Player Valuation", page_icon="🏏", layout="wide")
st.title("🏏 MONEYBALL: IPL Player Valuation")
st.caption("A data story: does the IPL auction pay for performance? | Team 05 | DSV 24ADI204")

PERF = ["runs", "balls_faced", "innings", "fours", "sixes", "dismissals",
        "balls_bowled", "runs_conceded", "wickets", "matches_bowled"]


@st.cache_data
def load_data(src):
    return pd.read_csv(src)


# ---------------- load data
csvs = glob.glob("*.csv")
st.sidebar.header("📁 Data")
if csvs:
    main_path = csvs[0] if len(csvs) == 1 else st.sidebar.selectbox("Main CSV", csvs)
    raw = load_data(main_path)
else:
    up = st.sidebar.file_uploader("Upload main CSV", type="csv")
    if up is None:
        st.info("Upload the dataset CSV in the sidebar to begin.")
        st.stop()
    raw = load_data(up)

# ---------------- ROLE CLEANING
# 1) Batter and Batsman are the same role
raw["role"] = raw["role"].replace({"Batter": "Batsman"})

# 2) Fill "Unknown" using the player's known role in other seasons (mode)
known = (raw[raw["role"] != "Unknown"].groupby("player")["role"]
         .agg(lambda x: x.mode().iloc[0]))
raw["role"] = raw["role"].where(raw["role"] != "Unknown",
                                raw["player"].map(known)).fillna("Unknown")

# 3) Still Unknown -> derive from career balls faced vs balls bowled
tot = raw.groupby("player")[["balls_faced", "balls_bowled"]].sum()
share = tot["balls_faced"] / (tot["balls_faced"] + tot["balls_bowled"]).replace(0, np.nan)
derived = pd.Series("All-Rounder", index=tot.index)
derived[share > 0.8] = "Batsman"
derived[share < 0.35] = "Bowler"
derived[share.isna()] = "Unknown"
mask = raw["role"] == "Unknown"
raw.loc[mask, "role"] = raw.loc[mask, "player"].map(derived).fillna("Unknown")

st.sidebar.caption(f"Rows with role still Unknown: {int((raw['role'] == 'Unknown').sum()):,}")

raw_up = st.sidebar.file_uploader("Optional: RAW CSV (for Data Quality chapter)", type="csv")
quality_df = load_data(raw_up) if raw_up is not None else raw

# ---------------- previous-season performance (what a team knows at auction time)
LAG_FEATS = [c for c in ["runs", "sixes", "fours", "innings", "wickets", "matches_bowled"] if c in raw.columns]


@st.cache_data
def build_lag(data):
    perf = (data.groupby(["player", "season"], as_index=False)[LAG_FEATS].sum()
            .sort_values(["player", "season"]).reset_index(drop=True))
    lagged = perf.groupby("player")[LAG_FEATS].shift(1).add_prefix("prev_")
    return pd.concat([perf[["player", "season"]], lagged], axis=1)


lag_df = build_lag(raw)

# ---------------- sidebar filters
st.sidebar.header("🎛️ Story controls")
chapter = st.sidebar.radio(
    "Chapters",
    ["1. Context", "2. Data Quality", "3. Fixing the Data", "4. The Conflict",
     "5. Hidden Structure (PCA)", "6. Resolution"],
)
seasons = sorted(raw["season"].dropna().unique())
sel_seasons = st.sidebar.multiselect("Seasons", seasons, default=seasons)
roles = sorted(raw["role"].dropna().unique())
sel_roles = st.sidebar.multiselect("Roles", roles, default=roles)

df = raw[raw["season"].isin(sel_seasons) & raw["role"].isin(sel_roles)].copy()
if df.empty:
    st.warning("No data for this filter. Pick at least one season and role.")
    st.stop()
auction = df.dropna(subset=["final_price_lakh"]) if "final_price_lakh" in df.columns else df.iloc[0:0]

# ================= 1 CONTEXT
if chapter.startswith("1"):
    st.header("Chapter 1: Context")
    st.markdown("Every year IPL franchises spend crores at the auction. "
                "**The question:** is that money going to the players who actually perform?")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Player-season records", f"{len(df):,}")
    c2.metric("Unique players", f"{df['player'].nunique():,}")
    c3.metric("Total runs", f"{int(df['runs'].sum()):,}")
    c4.metric("Total wickets", f"{int(df['wickets'].sum()):,}")
    if "team" in df.columns:
        st.caption(f"{len(df):,} records in total, of which {int(df['team'].notna().sum()):,} "
                   "have a team assigned (squad players). The rest were not picked that season.")

    left, right = st.columns(2)
    trend = df.groupby("season", as_index=False).agg(total_runs=("runs", "sum"), records=("player", "count"))
    trend["runs_per_record"] = (trend["total_runs"] / trend["records"]).round(1)
    left.plotly_chart(px.line(trend, x="season", y="runs_per_record", markers=True,
                              title="Average runs per player-season record"))
    role_ct = df["role"].value_counts().reset_index()
    role_ct.columns = ["role", "count"]
    right.plotly_chart(px.bar(role_ct, x="role", y="count", color="role", title="Players by role"))
    st.caption("We use runs per record (not total runs) because the latest season may be incomplete. "
               "Missing roles were filled from the player's other seasons or derived from balls faced vs bowled.")

# ================= 2 QUALITY
elif chapter.startswith("2"):
    st.header("Chapter 2: Data Quality Report")
    q = quality_df
    miss = q.isnull().sum()
    report = pd.DataFrame({"missing": miss, "missing_%": (miss / len(q) * 100).round(2),
                           "dtype": q.dtypes.astype(str)})
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows x Columns", f"{q.shape[0]:,} x {q.shape[1]}")
    c2.metric("Total missing cells", f"{int(miss.sum()):,}")
    c3.metric("Duplicate rows", f"{int(q.duplicated().sum())}")

    worst = report[report["missing"] > 0].sort_values("missing_%", ascending=False).reset_index()
    if len(worst):
        worst.columns = ["column", "missing", "missing_%", "dtype"]
        left, right = st.columns([3, 2])
        left.plotly_chart(px.bar(worst, x="missing_%", y="column", orientation="h",
                                 title="Columns with missing values (%)"))
        right.dataframe(report)
    else:
        st.success("No missing values in this file. Upload the RAW CSV in the sidebar to see the original mess.")
        st.dataframe(report)

    st.subheader("Outliers (IQR rule)")
    cols = [c for c in PERF if c in q.columns]
    num = st.selectbox("Pick a numeric column", cols)
    s = q[num].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    st.write(f"**{n_out}** outliers in `{num}` ({n_out / len(s) * 100:.1f}%).")
    st.plotly_chart(px.box(q, y=num, points="outliers", title=f"Box plot: {num}"))
    st.info("We kept outliers: in cricket they are the superstars. Missing auction fields "
            "(team, price, status) mean the player was not bought that season, so they are structural.")

# ================= 3 FIX
elif chapter.startswith("3"):
    st.header("Chapter 3: Fixing the Data (Before vs After)")
    cols = [c for c in ["runs", "balls_faced", "wickets", "runs_conceded", "sixes"] if c in df.columns]
    num = st.selectbox("Column", cols)
    skip_zero = st.checkbox("Ignore zeros (players who did not bat/bowl)", value=True)
    s = df[num].astype(float)
    if skip_zero:
        s = s[s > 0]
    if len(s) < 10:
        st.warning("Not enough non-zero values in this filter.")
        st.stop()
    logged = np.log1p(s)
    scaled = pd.Series(StandardScaler().fit_transform(logged.to_frame())[:, 0], index=s.index)
    stats = pd.DataFrame({
        "Raw": [s.mean(), s.median(), s.std(), s.skew(), s.kurt()],
        "log1p": [logged.mean(), logged.median(), logged.std(), logged.skew(), logged.kurt()],
        "log1p + StandardScaler": [scaled.mean(), scaled.median(), scaled.std(), scaled.skew(), scaled.kurt()],
    }, index=["mean", "median", "std", "skewness", "kurtosis"]).round(3)
    st.dataframe(stats)
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(px.histogram(s, nbins=40, title="Before: raw", marginal="box"))
    c2.plotly_chart(px.histogram(logged, nbins=40, title="After: log1p", marginal="box"))
    c3.plotly_chart(px.histogram(scaled, nbins=40, title="After: log1p + scaled", marginal="box"))
    st.success(f"Skewness moved from {s.skew():.2f} to {logged.skew():.2f}. Cricket stats are right-skewed "
               "(few stars, many role players), so log + scaling makes them comparable.")

# ================= 4 CONFLICT
elif chapter.startswith("4"):
    st.header("Chapter 4: The Conflict")
    st.markdown("If the auction rewarded performance, price would rise cleanly with output. Does it?")
    if auction.empty:
        st.warning("No priced players in this filter.")
        st.stop()
    metric_cols = [c for c in ["runs", "wickets", "sixes", "strike_rate"] if c in auction.columns]
    x_axis = st.selectbox("Performance metric", metric_cols)
    st.plotly_chart(px.scatter(auction, x=x_axis, y="final_price_lakh", color="role",
                               hover_name="player_full_name", opacity=0.7,
                               title=f"Final price vs {x_axis}"))
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.violin(auction, x="role", y="final_price_lakh", color="role",
                              box=True, title="Price spread by role"))
    if "team" in auction.columns:
        c2.plotly_chart(px.box(auction, x="team", y="final_price_lakh", color="team",
                               title="Price spread by team").update_layout(showlegend=False))
    ccols = ["final_price_lakh"] + [c for c in ["runs", "wickets", "sixes", "fours", "strike_rate", "economy"]
                                    if c in auction.columns]
    corr = auction[ccols].corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Correlation: price vs performance"))
    st.warning("Correlation with price is only moderate. Many high-paid players are not top performers.")

# ================= 5 PCA
elif chapter.startswith("5"):
    st.header("Chapter 5: Hidden Structure (PCA)")
    feats = [c for c in PERF if c in df.columns]
    X = StandardScaler().fit_transform(df[feats].fillna(0))
    pca = PCA().fit(X)
    evr = pca.explained_variance_ratio_
    c1, c2 = st.columns(2)
    fig = go.Figure()
    pcs = [f"PC{i+1}" for i in range(len(evr))]
    fig.add_bar(x=pcs, y=evr, name="Explained")
    fig.add_scatter(x=pcs, y=np.cumsum(evr), name="Cumulative", mode="lines+markers")
    fig.update_layout(title="Explained variance ratio")
    c1.plotly_chart(fig)
    comp = PCA(n_components=2).fit_transform(X)
    pdf = pd.DataFrame({"PC1": comp[:, 0], "PC2": comp[:, 1],
                        "role": df["role"].values, "player": df["player_full_name"].values})
    c2.plotly_chart(px.scatter(pdf, x="PC1", y="PC2", color="role", hover_name="player",
                               opacity=0.6, title="Players in PC space"))
    k = int(np.argmax(np.cumsum(evr) >= 0.90)) + 1
    st.info(f"{k} components keep about 90% of the variance out of {len(feats)} features.")
    st.dataframe(pd.DataFrame(pca.components_[:2].T, index=feats, columns=["PC1", "PC2"]).round(3))

# ================= 6 RESOLUTION (lag model)
else:
    st.header("Chapter 6: Resolution (Moneyball Value Model)")
    st.markdown(
        "A team bids **before** the season is played, so it can only use **last season's** performance. "
        "We predict the expected price from the player's previous-season stats (plus season, to adjust for "
        "auction inflation) and compare it with what was actually paid.  \n"
        "**Negative gap = priced below expectation (bargain). Positive gap = priced above expectation.**"
    )
    pf = ["prev_" + c for c in LAG_FEATS]
    m = auction.merge(lag_df, on=["player", "season"], how="left").dropna(subset=pf[:1])
    if len(m) < 30:
        st.warning("Not enough players with a previous season in this filter for the model.")
        st.stop()
    Xa = m[pf + ["season"]].fillna(0)
    ya = np.log1p(m["final_price_lakh"])
    model = LinearRegression().fit(Xa, ya)
    keep = [c for c in ["player_full_name", "season", "team", "role", "final_price_lakh"] if c in m.columns]
    out = m[keep + pf].copy()
    out["expected_price_lakh"] = np.expm1(model.predict(Xa)).round(0)
    out["gap_lakh"] = (out["final_price_lakh"] - out["expected_price_lakh"]).round(0)
    st.caption(f"Model R² = {model.score(Xa, ya):.2f} on {len(m):,} priced players who had a previous season. "
               "Rookies with no history are excluded. A modest R² means price depends on more than stats "
               "(brand value, age, nationality, bidding wars).")
    n = st.slider("Show top N", 5, 25, 10)
    c1, c2 = st.columns(2)
    c1.subheader("💎 Priced below expectation")
    c1.plotly_chart(px.bar(out.sort_values("gap_lakh").head(n), x="gap_lakh", y="player_full_name",
                           orientation="h", color="role"))
    c2.subheader("💸 Priced above expectation")
    c2.plotly_chart(px.bar(out.sort_values("gap_lakh", ascending=False).head(n), x="gap_lakh",
                           y="player_full_name", orientation="h", color="role"))
    if "team" in m.columns:
        st.plotly_chart(px.sunburst(m.dropna(subset=["team"]), path=["role", "team"],
                                    values="final_price_lakh", title="Where the auction money went"))
    st.dataframe(out.sort_values("gap_lakh"))
    st.success("**Insight:** last season's output explains only part of the price. A franchise that "
               "targets high-output players priced below expectation can beat the market.")
