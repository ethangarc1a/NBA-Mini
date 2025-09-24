import time
import numpy as np
import pandas as pd
import streamlit as st
import requests, re, json
from datetime import datetime, timezone

# ---- Streamlit setup
st.set_page_config(page_title="RefLens Mini+ (copy-paste)", layout="wide")
st.title("RefLens Mini+ — Win Probability & Foul Impact (copy-paste)")

with st.expander("About this mini build", expanded=False):
    st.markdown(
        "- Single-file Streamlit app\n"
        "- Loads one game's play-by-play from the NBA CDN (no stats.nba.com).\n"
        "- Computes a simple logistic baseline Home WP over events.\n"
        "- Detects fouls, shows observed ΔWP (next − current).\n"
        "- Quick what-if: 'no foul' (hold WP flat at the foul step).\n"
        "- Adds Game Date, Team Names and a '% of foul calls that went toward home' metric."
    )

# ---- Inputs
colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    game_id = st.text_input(
        "Enter GAME_ID (e.g., 0022300001)",
        value="0022300001",
        help="Regular season IDs look like 00223xxxxx for the 2023–24 season."
    )
with colB:
    btn_fetch = st.button("Fetch & Compute", type="primary")
with colC:
    st.caption("Tip: change GAME_ID then click Fetch.")

# ---- Constants & helpers
TOTAL_REG_SECS = 48 * 60  # 4x12 minutes. OTs are clamped to 0 remaining for simplicity.

def parse_period_clock_to_secs_left(period: int, pctimestr: str) -> int:
    """Convert period + 'MM:SS' into game seconds remaining (reg only)."""
    try:
        mm, ss = pctimestr.split(":")
        p_secs_left = int(mm) * 60 + int(ss)
    except Exception:
        p_secs_left = 0
    base_before_period = TOTAL_REG_SECS - (period - 1) * 12 * 60
    if base_before_period < 0:
        base_before_period = 0
    return max(0, min(base_before_period, p_secs_left + (base_before_period - 12 * 60)))

def quick_wp_home(score_margin: float, sec_remaining: float) -> float:
    """
    Tiny heuristic baseline: logistic in (margin, time).
    Positive margin helps; less time increases certainty.
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

def _clock_to_mmss(clock_val) -> str:
    if not clock_val:
        return "00:00"
    s = str(clock_val)
    # ISO-like "PT11M25.00S"
    m = re.match(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", s)
    if m:
        mm = int(m.group(1) or 0)
        ss = float(m.group(2) or 0.0)
        return f"{mm:02d}:{int(round(ss)):02d}"
    if re.match(r"^\d{1,2}:\d{2}$", s):
        return s
    if re.match(r"^\d{1,2}:\d{2}\.\d+$", s):
        return s.split(".")[0]
    return "00:00"

@st.cache_data(show_spinner=False)
def load_game(game_id: str):
    """
    Load play-by-play from NBA CDN, plus light metadata.
    URL: https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_<GAME_ID>.json
    Returns (df, meta_dict).
    """
    url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"

    last_err = None
    data = None
    for i in range(4):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                break
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(0.8 * (i + 1))
    if data is None:
        raise ValueError(f"Failed to fetch CDN play-by-play: {last_err}")

    game = (data or {}).get("game", {}) or {}
    actions = game.get("actions", []) or []
    if not actions:
        raise ValueError("CDN returned no play-by-play actions for this GAME_ID.")

    # Meta (best-effort; keys vary slightly across seasons)
    home = (game.get("homeTeam") or {}) if isinstance(game.get("homeTeam"), dict) else {}
    away = (game.get("awayTeam") or {}) if isinstance(game.get("awayTeam"), dict) else {}
    meta = {
        "home_tricode": home.get("teamTricode") or home.get("triCode") or "",
        "home_city": home.get("teamCity") or home.get("city") or "",
        "home_name": home.get("teamName") or home.get("name") or "",
        "away_tricode": away.get("teamTricode") or away.get("triCode") or "",
        "away_city": away.get("teamCity") or away.get("city") or "",
        "away_name": away.get("teamName") or away.get("name") or "",
        "game_time_utc": game.get("gameTimeUTC") or game.get("gameTime") or "",
    }
    # Human date
    game_date = ""
    try:
        if meta["game_time_utc"]:
            dt = datetime.fromisoformat(meta["game_time_utc"].replace("Z","+00:00")).astimezone(timezone.utc)
            game_date = dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    meta["game_date"] = game_date

    # Rows
    rows = []
    for a in actions:
        period = int(a.get("period", 0) or 0)
        pctimestr = _clock_to_mmss(a.get("clock"))
        sh = a.get("scoreHome")
        sa = a.get("scoreAway")
        try:
            sh = int(sh) if sh is not None else None
        except Exception:
            sh = None
        try:
            sa = int(sa) if sa is not None else None
        except Exception:
            sa = None

        desc = a.get("description") or ""
        rows.append({
            "GAME_ID": str(game_id),
            "EVENTNUM": int(a.get("actionNumber", 0) or 0),
            "PERIOD": period,
            "PCTIMESTRING": pctimestr,
            "SCORE": (f"{sh} - {sa}" if sh is not None and sa is not None else None),
            "HOMEDESCRIPTION": None,
            "VISITORDESCRIPTION": None,
            "NEUTRALDESCRIPTION": desc,
            "home_score": sh,
            "away_score": sa,
        })

    df = pd.DataFrame(rows).sort_values(["PERIOD","EVENTNUM"]).reset_index(drop=True)

    # Derived
    df["sec_remaining"] = df.apply(
        lambda r: parse_period_clock_to_secs_left(int(r["PERIOD"]), str(r["PCTIMESTRING"])), axis=1
    )

    df["SCORE"] = df["SCORE"].ffill()
    if df["home_score"].isna().any() or df["away_score"].isna().any():
        df["home_score"] = df["home_score"].ffill()
        df["away_score"] = df["away_score"].ffill()

    df["score_margin"] = df["home_score"] - df["away_score"]

    # Fouls & WP
    flags = df.apply(detect_foul_flags, axis=1, result_type="expand")
    df["is_foul"] = flags[0]
    df["foul_type"] = flags[1].fillna("")

    mask = df["home_score"].notna() & df["away_score"].notna()
    df["wp_home"] = np.nan
    df.loc[mask, "wp_home"] = quick_wp_home(df.loc[mask, "score_margin"], df.loc[mask, "sec_remaining"])

    df["wp_next"] = df["wp_home"].shift(-1).fillna(method="ffill")
    df["dwp_obs"] = df["wp_next"] - df["wp_home"]

    return df, meta

def foul_table(df: pd.DataFrame) -> pd.DataFrame:
    f = df[df["is_foul"] & df["wp_home"].notna()].copy()
    f["toward_home"] = np.sign(f["dwp_obs"]).fillna(0.0).astype(float) > 0.0
    f["call_towards"] = np.where(f["toward_home"], "HOME", np.where(f["dwp_obs"] < 0, "AWAY", "NEUTRAL"))
    return f[["EVENTNUM","PERIOD","PCTIMESTRING","foul_type","call_towards","wp_home","wp_next","dwp_obs"]].rename(
        columns={
            "EVENTNUM":"event_num",
            "PCTIMESTRING":"clock",
            "wp_home":"WP (home)",
            "wp_next":"WP next",
            "dwp_obs":"ΔWP obs"
        }
    )

def simulate_no_foul(df: pd.DataFrame, event_num: int) -> float:
    """Counterfactual: if the foul did not happen, 'hold WP flat' at that step."""
    g = df.set_index("EVENTNUM")
    if event_num not in g.index:
        raise ValueError("Event not found.")
    wp_now = g.at[event_num, "wp_home"]
    wp_next_obs = g.at[event_num, "wp_next"]
    if pd.isna(wp_now) or pd.isna(wp_next_obs):
        return float("nan")
    return float(wp_now - wp_next_obs)

# ---- Action
if btn_fetch:
    try:
        df, meta = load_game(game_id)
    except Exception as e:
        st.error(f"Failed to load play-by-play: {e}")
        st.stop()

    # Header with Teams + Date
    home_label = " ".join([x for x in [meta.get("home_city"), meta.get("home_name"), f"({meta.get('home_tricode')})"] if x])
    away_label = " ".join([x for x in [meta.get("away_city"), meta.get("away_name"), f"({meta.get('away_tricode')})"] if x])
    date_label = meta.get("game_date") or "—"

    st.subheader("Game")
    st.markdown(f"**{away_label} @ {home_label}** — **Date:** {date_label}")

    # Layout
    left, right = st.columns([1.4, 1])

    # Chart
    with left:
        st.subheader("Home Win Probability over events")
        chart_df = df.loc[df["wp_home"].notna(), ["EVENTNUM","wp_home"]].copy()
        chart_df = chart_df.rename(columns={"EVENTNUM":"event"})
        chart_df = chart_df.set_index("event")
        st.line_chart(chart_df, height=360)
        st.caption("Baseline logistic WP ~ (margin, time). Heuristic for demo.")

    # Fouls panel
    with right:
        st.subheader("Fouls & observed ΔWP")
        ft = foul_table(df)
        st.dataframe(ft, use_container_width=True, hide_index=True)

        # Summary metrics
        foul_rows = df[df["is_foul"] & df["wp_home"].notna()].copy()
        total_fouls = int(foul_rows.shape[0])
        pct_toward_home = float((foul_rows["dwp_obs"] > 0).mean()*100) if total_fouls > 0 else 0.0
        sum_dwp = float(foul_rows["dwp_obs"].sum()) if total_fouls > 0 else 0.0
        avg_dwp = float(foul_rows["dwp_obs"].mean()) if total_fouls > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("% calls toward HOME", f"{pct_toward_home:.0f}%")
        m2.metric("Σ ΔWP over fouls", f"{sum_dwp:+.3f}")
        m3.metric("Avg ΔWP per foul", f"{avg_dwp:+.3f}")

        # What-if
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
                    st.success(f"Counterfactual ΔWP (no-foul vs observed next): {dwp_cf:+.3f}")

    with st.expander("Details & notes"):
        st.markdown(
            "- **Team/Date** parsed from the CDN’s `game` object (best effort).\n"
            "- **% toward HOME** = share of foul events where observed ΔWP > 0 for home.\n"
            "- This is a transparent **impact proxy**, not the officiating intent.\n"
            "- OT handling: seconds remaining clamp to 0; the chart still renders."
        )

else:
    st.info("Enter a GAME_ID and click **Fetch & Compute** to view WP, fouls, team/date, and '% toward home'.")
