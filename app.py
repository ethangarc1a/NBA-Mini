import time
import numpy as np
import pandas as pd
import streamlit as st

# --- tiny helper to import on first use (avoids long import delay before UI renders)
@st.cache_resource
def _nba_endpoint():
    from nba_api.stats.endpoints import playbyplayv2
    return playbyplayv2

st.set_page_config(page_title="RefLens Mini (copy-paste)", layout="wide")
st.title("RefLens Mini (copy-paste) — Win Probability & Foul Impact")

with st.expander("About this mini build", expanded=False):
    st.markdown(
        "- Single file Streamlit app\n"
        "- Fetches one game's play-by-play, computes a simple logistic baseline WP (home),\n"
        "- Detects fouls, shows observed ΔWP (next − current),\n"
        "- Offers a *what-if (no foul)* quick counterfactual = hold WP flat at the foul step."
    )

# --- Inputs
colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    game_id = st.text_input(
        "Enter GAME_ID (e.g., 0022300001)",
        value="0022300001",
        help="Regular-season IDs look like 00223xxxxx for the 2023-24 season."
    )
with colB:
    btn_fetch = st.button("Fetch & Compute", type="primary")
with colC:
    st.caption("Tip: change GAME_ID then click Fetch.")

# --- Core helpers
TOTAL_REG_SECS = 48 * 60  # assumes 4x12: this mini keeps it simple

def parse_period_clock_to_secs_left(period: int, pctimestr: str) -> int:
    # PCTIMESTRING is 'MM:SS' within the period
    try:
        mm, ss = pctimestr.split(":")
        p_secs_left = int(mm) * 60 + int(ss)
    except Exception:
        p_secs_left = 0
    # map period to game seconds remaining; we clamp at 0 for simplicity
    # Mini-app assumes regulation and treats OT as 0 left (still yields a chart)
    base_before_period = TOTAL_REG_SECS - (period - 1) * 12 * 60
    if base_before_period < 0:
        base_before_period = 0
    return max(0, min(base_before_period, p_secs_left + (base_before_period - 12 * 60)))

def split_score_to_home_away(score_str: str):
    # SCORE like "10 - 9"
    if not isinstance(score_str, str) or "-" not in score_str:
        return np.nan, np.nan
    try:
        a, b = score_str.replace(" ", "").split("-")
        return int(a), int(b)
    except Exception:
        return np.nan, np.nan

def quick_wp_home(score_margin: float, sec_remaining: float) -> float:
    """
    Tiny baseline: logistic on (margin, time).
    Intuition: margin helps; less time increases certainty (steeper effect).
    Coeffs are heuristic for demo only.
    """
    z = 0.14 * score_margin - 0.0022 * sec_remaining
    return 1.0 / (1.0 + np.exp(-z))

def detect_foul_flags(row: pd.Series) -> tuple[bool, str]:
    text = " ".join(
        str(x) for x in [
            row.get("HOMEDESCRIPTION",""),
            row.get("VISITORDESCRIPTION",""),
            row.get("NEUTRALDESCRIPTION","")
        ] if pd.notna(x)
    ).upper()
    is_foul = "FOUL" in text
    ftype = "FOUL" if is_foul else ""
    if is_foul:
        if "S.FOUL" in text or "SHOOTING" in text:
            ftype = "SHOOTING FOUL"
        elif "OFFENSIVE" in text:
            ftype = "OFFENSIVE FOUL"
        elif "FLAGRANT" in text:
            ftype = "FLAGRANT FOUL"
        elif "TECH" in text or "TECHNICAL" in text:
            ftype = "TECHNICAL FOUL"
    return is_foul, ftype

@st.cache_data(show_spinner=False)
def load_game(game_id: str) -> pd.DataFrame:
    pbp_endpoint = _nba_endpoint()
    # be nice to the API
    time.sleep(0.6)
    df = pbp_endpoint.PlayByPlayV2(game_id=game_id).get_data_frames()[0].copy()
    if df.empty:
        raise ValueError("Empty play-by-play for this GAME_ID.")
    # Keep the essentials
    keep_cols = [
        "GAME_ID","EVENTNUM","PERIOD","PCTIMESTRING","SCORE",
        "HOMEDESCRIPTION","VISITORDESCRIPTION","NEUTRALDESCRIPTION"
    ]
    df = df[keep_cols].copy()

    # Time
    df["sec_remaining"] = df.apply(
        lambda r: parse_period_clock_to_secs_left(int(r["PERIOD"]), str(r["PCTIMESTRING"])), axis=1
    )

    # Scores forward-fill then split
    df["SCORE"] = df["SCORE"].ffill()
    ha = df["SCORE"].apply(split_score_to_home_away)
    df["home_score"] = ha.apply(lambda t: t[0])
    df["away_score"] = ha.apply(lambda t: t[1])
    # Valid rows only for WP curve
    df["score_margin"] = df["home_score"] - df["away_score"]

    # Foul flags
    flags = df.apply(detect_foul_flags, axis=1, result_type="expand")
    df["is_foul"] = flags[0]
    df["foul_type"] = flags[1].fillna("")

    # Win prob (home) — only where we have scores
    mask = df["home_score"].notna() & df["away_score"].notna()
    df["wp_home"] = np.nan
    df.loc[mask, "wp_home"] = quick_wp_home(df.loc[mask,"score_margin"], df.loc[mask,"sec_remaining"])

    # Observed ΔWP relative to next event
    df = df.sort_values(["PERIOD","EVENTNUM"]).reset_index(drop=True)
    df["wp_next"] = df["wp_home"].shift(-1).fillna(method="ffill")
    df["dwp_obs"] = df["wp_next"] - df["wp_home"]

    return df

def foul_table(df: pd.DataFrame) -> pd.DataFrame:
    f = df[df["is_foul"] & df["wp_home"].notna()].copy()
    # Present a concise table
    return f[["EVENTNUM","PERIOD","PCTIMESTRING","foul_type","wp_home","wp_next","dwp_obs"]].rename(
        columns={
            "EVENTNUM":"event_num",
            "PCTIMESTRING":"clock",
            "wp_home":"WP (home)",
            "wp_next":"WP next",
            "dwp_obs":"ΔWP obs"
        }
    )

def simulate_no_foul(df: pd.DataFrame, event_num: int) -> float:
    """
    Minimal counterfactual: assume the foul *does not occur*,
    so the immediate next step WP equals current WP (i.e., we 'hold flat').
    ΔWP_cf = (WP_no_foul_next - WP_observed_next).
    """
    g = df.set_index("EVENTNUM")
    if event_num not in g.index:
        raise ValueError("Event not found.")
    wp_now = g.at[event_num, "wp_home"]
    wp_next_obs = g.at[event_num, "wp_next"]
    if pd.isna(wp_now) or pd.isna(wp_next_obs):
        return float("nan")
    return float(wp_now - wp_next_obs)

# --- Action
if btn_fetch:
    try:
        df = load_game(game_id)
    except Exception as e:
        st.error(f"Failed to load play-by-play: {e}")
        st.stop()

    # --- Layout
    left, right = st.columns([1.4, 1])

    # Chart
    with left:
        st.subheader("Win Probability (home) over events")
        chart_df = df.loc[df["wp_home"].notna(), ["EVENTNUM","wp_home"]].copy()
        chart_df = chart_df.rename(columns={"EVENTNUM":"event"})
        chart_df = chart_df.set_index("event")
        st.line_chart(chart_df, height=360)
        st.caption("Baseline logistic WP ~ margin & time (demo heuristic).")

    # Fouls
    with right:
        st.subheader("Fouls & observed ΔWP")
        ft = foul_table(df)
        st.dataframe(ft, use_container_width=True, hide_index=True)

        if not ft.empty:
            event_choice = st.selectbox(
                "Pick a foul to simulate 'no-foul' (hold WP flat at this step):",
                options=ft["event_num"].tolist(),
            )
            if st.button("Run what-if (no foul)"):
                dwp_cf = simulate_no_foul(df, int(event_choice))
                if pd.isna(dwp_cf):
                    st.warning("Insufficient WP data around that event.")
                else:
                    st.success(
                        f"Counterfactual ΔWP (no-foul vs observed next): {dwp_cf:+.3f}"
                    )

    # Quick totals
    with st.expander("Quick metrics"):
        foul_rows = df[df["is_foul"] & df["wp_home"].notna()]
        total_obs = foul_rows["dwp_obs"].sum() if not foul_rows.empty else 0.0
        st.write(f"Sum of observed ΔWP over fouls (this game): {total_obs:+.3f}")
        st.caption("Think of this as a super simple RII-mini proxy (observed only).")
else:
    st.info("Enter a GAME_ID and click **Fetch & Compute** to view charts and foul impact.")
