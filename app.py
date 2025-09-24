# app.py
# RefLens Mobile — Fouls & Win Probability (Free APIs; Regular Season + Playoffs)
# - Single-file Streamlit app
# - NBA play-by-play via nba_api (free)
# - Win Probability (baseline logistic) + foul markers + bonus detection
# - ΔWP around fouls (quick local counterfactual)
# - Team & Player views; mobile-friendly layout
#
# Notes:
# - This uses a simple, transparent baseline WP model (time + score diff).
# - NBA in-game "official WP" endpoint is not publicly exposed through nba_api;
#   so we compute a baseline to keep it free & reproducible.
# - Playoffs included (scoreboard returns all scheduled games for the date).
#
# If stats.nba.com times out (rare), retry via the "Reload data" button.

import time
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- NBA API (lazy import to avoid initial UI lag)
@st.cache_resource
def _nba_endpoints():
    from nba_api.stats.endpoints import scoreboardv2, playbyplayv2, boxscoretraditionalv2
    from nba_api.stats.library.parameters import DayOffset
    return scoreboardv2, playbyplayv2, boxscoretraditionalv2, DayOffset

ScoreboardV2, PlayByPlayV2, BoxScoreTraditionalV2, DayOffset = _nba_endpoints()

st.set_page_config(page_title="RefLens Mobile — Fouls & Win Probability", layout="wide")
st.markdown(
    """
    <style>
    /* Mobile-first: tighten paddings, nicer fonts */
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    h1,h2,h3 { font-family: system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; }
    .small { font-size: 0.9rem; opacity: 0.85; }
    .tag { display:inline-block; padding:2px 8px; border-radius:14px; font-size:0.75rem;
           background:#111; color:#eee; margin-right:6px; }
    .badge { display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.75rem; margin-right:6px; }
    .badge.warn { background:#ffe8e8; color:#a00; border:1px solid #f5bcbc; }
    .badge.ok { background:#e9f7ef; color:#0a8; border:1px solid #b8ead3; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RefLens Mobile — Fouls & Win Probability")

with st.expander("About this build", expanded=False):
    st.markdown(
        "- Free data via `nba_api` (play-by-play + box scores)\n"
        "- Mobile-friendly win-probability timeline with foul markers\n"
        "- ΔWP around each foul (next − current) as quick on-the-fly impact\n"
        "- Detects **bonus** and **player foul trouble** (2-in-Q1, 3-by-halftime, 5-in-Q4)\n"
        "- Works for **regular season & playoffs** (pick any date & game)\n"
        "- Designed for copy-paste: single file, defensive error handling\n"
    )

# ----------------------------
# Helpers
# ----------------------------

def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def period_time_to_elapsed_seconds(period: int, pctimestring: str) -> int:
    """
    Convert 'MM:SS' (PC_TIME) + period -> absolute elapsed game seconds from tip.
    NBA regulation: 12-minute quarters. OT periods are 5 minutes each.
    """
    try:
        mm, ss = pctimestring.split(":")
        remaining = int(mm) * 60 + int(ss)
    except Exception:
        remaining = 0
    if period <= 4:
        period_len = 12 * 60
        prior = (period - 1) * period_len
        elapsed_in_period = period_len - remaining
        return prior + elapsed_in_period
    else:
        # Overtime: 5 minutes each
        ot_len = 5 * 60
        prior_reg = 4 * 12 * 60
        prior_ot = (period - 5) * ot_len
        elapsed_in_period = ot_len - remaining
        return prior_reg + prior_ot + elapsed_in_period

def total_game_seconds(max_period: int) -> int:
    if max_period <= 4:
        return 4 * 12 * 60
    else:
        extra = (max_period - 4) * 5 * 60
        return 4 * 12 * 60 + extra

# Baseline Win Probability model (transparent, stable)
def baseline_wp(home_score: int, away_score: int, elapsed_s: int, total_s: int) -> float:
    """
    Simple logistic WP model:
    features: lead = home - away, time factor = sqrt(time remaining ratio)
    wp = 1/(1+exp(-(a + b*lead + c*lead*sqrt(remfrac))))
    Coefficients below tuned to be reasonable for NBA pace without overfitting.
    """
    lead = home_score - away_score
    remfrac = max(0.0, (total_s - elapsed_s) / max(1.0, total_s))
    x = 0.08 * lead + 1.25 * lead * math.sqrt(remfrac)
    # center: toss-up when x = 0
    wp = 1.0 / (1.0 + math.exp(-x))
    return float(np.clip(wp, 0.001, 0.999))

@dataclass
class FoulEvent:
    idx: int
    period: int
    pc_time: str
    elapsed_s: int
    team: str
    player: Optional[str]
    foul_type: str
    committing_player_id: Optional[int]
    team_id: Optional[int]
    is_shooting: bool
    team_in_bonus: bool
    home_wp_before: float
    home_wp_after: float
    delta_wp_home: float  # after - before from home POV

# ----------------------------
# Data fetch
# ----------------------------

@st.cache_data(show_spinner=False, ttl=60*10)
def get_games_for_date(game_date_str: str) -> pd.DataFrame:
    """
    Returns ScoreboardV2 for a date (YYYY-MM-DD). Includes regular + playoffs if on that date.
    """
    # NBA expects MM/DD/YYYY
    month, day, year = game_date_str[5:7], game_date_str[8:10], game_date_str[0:4]
    req_date = f"{month}/{day}/{year}"
    sb = ScoreboardV2(game_date=req_date, day_offset=DayOffset.default)
    games = sb.game_header.get_data_frame()
    return games

@st.cache_data(show_spinner=False, ttl=60*30)
def get_pbp(game_id: str) -> pd.DataFrame:
    pbp = PlayByPlayV2(game_id=game_id, timeout=30)
    df = pbp.get_data_frames()[0].copy()
    return df

@st.cache_data(show_spinner=False, ttl=60*30)
def get_boxscore(game_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bs = BoxScoreTraditionalV2(game_id=game_id, timeout=30)
    team_df = bs.team_stats.get_data_frame().copy()
    player_df = bs.player_stats.get_data_frame().copy()
    return team_df, player_df

# ----------------------------
# Parsing & metrics
# ----------------------------

def enrich_pbp_with_scores_wp(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Build cumulative scores & WP across events; also detect bonus and player fouls.
    """
    df = pbp.copy()
    # Ensure needed columns exist
    needed = ["PERIOD", "PCTIMESTRING", "HOMEDESCRIPTION", "VISITORDESCRIPTION",
              "SCORE", "SCOREMARGIN", "EVENTMSGTYPE", "PLAYER1_TEAM_ABBREVIATION",
              "PLAYER1_NAME", "PLAYER1_ID", "PLAYER1_TEAM_ID"]
    for c in needed:
        if c not in df.columns:
            df[c] = None

    # Fill score columns
    home_score = []
    away_score = []
    h, a = 0, 0
    for s in df["SCORE"].fillna(""):
        if isinstance(s, str) and "-" in s:
            try:
                parts = s.split("-")
                a = safe_int(parts[0])
                h = safe_int(parts[1])
            except Exception:
                pass
        home_score.append(h)
        away_score.append(a)
    df["HOME_SCORE"] = home_score
    df["AWAY_SCORE"] = away_score

    # Elapsed + total seconds
    df["ELAPSED_S"] = [period_time_to_elapsed_seconds(int(p), t if isinstance(t, str) else "00:00")
                       for p, t in zip(df["PERIOD"].fillna(1), df["PCTIMESTRING"].fillna("00:00"))]
    max_period = int(df["PERIOD"].max() or 4)
    total_s = total_game_seconds(max_period)
    df["TOTAL_S"] = total_s

    # Compute baseline WP (home POV)
    df["HOME_WP"] = [baseline_wp(hh, aa, es, total_s) for hh, aa, es in zip(df["HOME_SCORE"], df["AWAY_SCORE"], df["ELAPSED_S"])]

    # Track team fouls per quarter to mark bonus
    df["EVENTMSGTYPE"] = df["EVENTMSGTYPE"].fillna(0).astype(int)
    is_foul = df["EVENTMSGTYPE"] == 6

    # team abbrev from PLAYER1_TEAM_ABBREVIATION if available, else parse description
    def infer_team(row) -> str:
        t = row.get("PLAYER1_TEAM_ABBREVIATION")
        if isinstance(t, str) and t.strip():
            return t
        hdesc = row.get("HOMEDESCRIPTION") or ""
        vdesc = row.get("VISITORDESCRIPTION") or ""
        # crude: if foul text only in one side, map to that side
        if "FOUL" in hdesc.upper() and not "FOUL" in vdesc.upper():
            return "HOME"
        if "FOUL" in vdesc.upper() and not "FOUL" in hdesc.upper():
            return "AWAY"
        return "UNK"

    df["FOUL_TEAM"] = [infer_team(r) for _, r in df.iterrows()]

    # Count team fouls per period
    df["HOME_TEAM_FOULS_P"] = 0
    df["AWAY_TEAM_FOULS_P"] = 0
    df["HOME_IN_BONUS"] = False
    df["AWAY_IN_BONUS"] = False

    home_fouls = {}
    away_fouls = {}

    for i, row in df.iterrows():
        p = int(row["PERIOD"]) if not pd.isna(row["PERIOD"]) else 1
        if p not in home_fouls:
            home_fouls[p] = 0
        if p not in away_fouls:
            away_fouls[p] = 0
        if is_foul.iloc[i]:
            team = row["FOUL_TEAM"]
            if team == "HOME" or (isinstance(team, str) and team == row.get("HOMEDESCRIPTION", "")):
                home_fouls[p] += 1
            elif team == "AWAY":
                away_fouls[p] += 1
            else:
                # Try to infer via description side:
                hdesc = (row["HOMEDESCRIPTION"] or "").upper()
                vdesc = (row["VISITORDESCRIPTION"] or "").upper()
                # If foul mentioned in home description:
                if "FOUL" in hdesc and "FOUL" not in vdesc:
                    home_fouls[p] += 1
                elif "FOUL" in vdesc and "FOUL" not in hdesc:
                    away_fouls[p] += 1
                # else uncertain -> ignore count increment

        df.at[i, "HOME_TEAM_FOULS_P"] = home_fouls[p]
        df.at[i, "AWAY_TEAM_FOULS_P"] = away_fouls[p]
        df.at[i, "HOME_IN_BONUS"] = home_fouls[p] >= 5
        df.at[i, "AWAY_IN_BONUS"] = away_fouls[p] >= 5

    # Player foul counts to detect "foul trouble"
    df["FOUL_TYPE_TEXT"] = ""
    df["FOUL_PLAYER"] = None
    df["FOUL_PLAYER_ID"] = None
    df["FOUL_IS_SHOOTING"] = False
    df["FOUL_TROUBLE_FLAG"] = ""  # "2-in-Q1", "3-by-Half", "5-in-Q4", ""

    player_pf: Dict[int, int] = {}
    for i, row in df.iterrows():
        if is_foul.iloc[i]:
            # parse description strings
            text = (row["HOMEDESCRIPTION"] or "") + " " + (row["VISITORDESCRIPTION"] or "")
            up = text.upper()
            foul_type = "FOUL"
            if "SHOOTING" in up:
                shoot = True
                foul_type = "SHOOTING FOUL"
            else:
                shoot = False
                # try basic extraction
            df.at[i, "FOUL_TYPE_TEXT"] = foul_type

            pid = row.get("PLAYER1_ID")
            pname = row.get("PLAYER1_NAME")
            if pd.notna(pid):
                pid = int(pid)
                pf_prev = player_pf.get(pid, 0)
                pf_now = pf_prev + 1
                player_pf[pid] = pf_now
                df.at[i, "FOUL_PLAYER_ID"] = pid
                df.at[i, "FOUL_PLAYER"] = pname

                # foul trouble flags
                period = int(row["PERIOD"] or 1)
                elapsed = int(row["ELAPSED_S"] or 0)
                half = 1 if period <= 2 else 2
                flag = ""
                if period == 1 and pf_now >= 2:
                    flag = "2-in-Q1"
                elif half == 1 and pf_now >= 3:
                    flag = "3-by-Half"
                elif period >= 4 and pf_now >= 5:
                    flag = "5-in-Q4+"
                df.at[i, "FOUL_TROUBLE_FLAG"] = flag

            df.at[i, "FOUL_IS_SHOOTING"] = shoot

    # ΔWP around fouls (after-next - current)
    df["HOME_WP_NEXT"] = df["HOME_WP"].shift(-1).fillna(df["HOME_WP"])
    df["FOUL_DELTA_WP_HOME"] = (df["HOME_WP_NEXT"] - df["HOME_WP"]).where(is_foul, 0.0)

    return df

def summarize_foul_events(df: pd.DataFrame, home_abbr: str, away_abbr: str) -> List[FoulEvent]:
    events: List[FoulEvent] = []
    for i, row in df.iterrows():
        if int(row["EVENTMSGTYPE"]) == 6:
            team = row.get("PLAYER1_TEAM_ABBREVIATION")
            # If missing, infer by side of description
            if not isinstance(team, str) or not team:
                hdesc = (row["HOMEDESCRIPTION"] or "").upper()
                vdesc = (row["VISITORDESCRIPTION"] or "").upper()
                if "FOUL" in hdesc and "FOUL" not in vdesc:
                    team = home_abbr
                elif "FOUL" in vdesc and "FOUL" not in hdesc:
                    team = away_abbr
                else:
                    team = "UNK"
            events.append(
                FoulEvent(
                    idx=i,
                    period=int(row["PERIOD"] or 1),
                    pc_time=str(row["PCTIMESTRING"] or "00:00"),
                    elapsed_s=int(row["ELAPSED_S"] or 0),
                    team=team,
                    player=row.get("FOUL_PLAYER"),
                    foul_type=row.get("FOUL_TYPE_TEXT") or "FOUL",
                    committing_player_id=row.get("FOUL_PLAYER_ID"),
                    team_id=row.get("PLAYER1_TEAM_ID"),
                    is_shooting=bool(row.get("FOUL_IS_SHOOTING")),
                    team_in_bonus=bool(row["HOME_IN_BONUS"]) if team == home_abbr else bool(row["AWAY_IN_BONUS"]),
                    home_wp_before=float(row["HOME_WP"]),
                    home_wp_after=float(row["HOME_WP_NEXT"]),
                    delta_wp_home=float(row["FOUL_DELTA_WP_HOME"]),
                )
            )
    return events

def build_game_selector(date_str: str):
    games = get_games_for_date(date_str)
    if games.empty:
        st.warning("No NBA games found for this date.")
        st.stop()

    # Build nice labels
    games["LABEL"] = games.apply(
        lambda r: f"{r['VISITOR_TEAM_ABBREVIATION']} @ {r['HOME_TEAM_ABBREVIATION']} — {r['GAME_STATUS_TEXT']} — {r['GAME_ID']}",
        axis=1,
    )
    idx = 0
    game_label = st.selectbox("Select a game", games["LABEL"].tolist(), index=idx)
    row = games[games["LABEL"] == game_label].iloc[0]
    return row["GAME_ID"], row["HOME_TEAM_ABBREVIATION"], row["VISITOR_TEAM_ABBREVIATION"], row

# ----------------------------
# UI Controls
# ----------------------------

left, right = st.columns([1, 2], vertical_alignment="center")

with left:
    today_str = pd.Timestamp.today(tz="US/Pacific").strftime("%Y-%m-%d")
    date_str = st.date_input("Game date", value=pd.to_datetime(today_str)).strftime("%Y-%m-%d")
    st.caption("Includes regular season & playoffs on the chosen date.")
    reload = st.button("🔁 Reload data (if a timeout occurred)")

with right:
    st.markdown(
        "<div class='small'>Tip: On mobile, pinch-zoom the timeline; tap markers for foul details. "
        "Toggle layers in the legend to declutter.</div>",
        unsafe_allow_html=True,
    )

try:
    game_id, home_abbr, away_abbr, game_row = build_game_selector(date_str)
except Exception as e:
    st.error(f"Failed to load games for {date_str}: {e}")
    st.stop()

if reload:
    get_pbp.clear()
    get_boxscore.clear()
    enrich_pbp_with_scores_wp.clear()

# ----------------------------
# Fetch & compute
# ----------------------------
with st.spinner("Fetching play-by-play and computing metrics..."):
    try:
        pbp_raw = get_pbp(game_id)
        team_df, player_df = get_boxscore(game_id)
    except Exception as e:
        st.error(f"Failed to load play-by-play: {e}")
        st.stop()

    if pbp_raw.empty:
        st.warning("No play-by-play available yet for this game.")
        st.stop()

    pbp = enrich_pbp_with_scores_wp(pbp_raw)
    fouls = summarize_foul_events(pbp, home_abbr, away_abbr)

# ----------------------------
# Top banner / summary
# ----------------------------
home_name = team_df.loc[team_df["TEAM_ABBREVIATION"] == home_abbr, "TEAM_NAME"].values
away_name = team_df.loc[team_df["TEAM_ABBREVIATION"] == away_abbr, "TEAM_NAME"].values
home_name = home_name[0] if len(home_name) else home_abbr
away_name = away_name[0] if len(away_name) else away_abbr

st.subheader(f"{away_abbr} @ {home_abbr}")
st.markdown(
    f"<span class='tag'>Game ID: {game_id}</span>"
    f"<span class='tag'>{away_name} at {home_name}</span>",
    unsafe_allow_html=True,
)

# Team foul counts by period
periods = sorted(pbp["PERIOD"].dropna().unique().tolist())
team_fouls_by_p = pd.DataFrame({
    "PERIOD": periods,
    f"{home_abbr}_FOULS": [int(pbp.loc[pbp["PERIOD"] == p, "HOME_TEAM_FOULS_P"].max()) for p in periods],
    f"{away_abbr}_FOULS": [int(pbp.loc[pbp["PERIOD"] == p, "AWAY_TEAM_FOULS_P"].max()) for p in periods],
})
home_bonus_periods = [p for p in periods if (pbp.loc[pbp["PERIOD"] == p, "HOME_IN_BONUS"]).any()]
away_bonus_periods = [p for p in periods if (pbp.loc[pbp["PERIOD"] == p, "AWAY_IN_BONUS"]).any()]

# ----------------------------
# Main chart: Win probability + foul markers + bonus shading
# ----------------------------
with st.spinner("Rendering timeline..."):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Win prob (home)
    x = pbp["ELAPSED_S"]
    total_s = int(pbp["TOTAL_S"].iloc[-1])
    # Format x ticks as Q1..OT with mm:ss
    def fmt_xticks(seconds: List[int]) -> List[str]:
        labels = []
        for s in seconds:
            # map back to period/time
            # find closest row
            idx = int(np.argmin(np.abs(pbp["ELAPSED_S"].values - s)))
            period = int(pbp["PERIOD"].iloc[idx])
            t = str(pbp["PCTIMESTRING"].iloc[idx])
            labels.append(f"Q{period} {t}" if period <= 4 else f"OT{period-4} {t}")
        return labels

    fig.add_trace(
        go.Scatter(
            x=x, y=pbp["HOME_WP"],
            mode="lines",
            name=f"WP — {home_abbr} (home)",
            hovertemplate="t=%{x:.0f}s • WP=%{y:.1%}<extra></extra>",
        ),
        row=1, col=1
    )

    # Lead tracker (secondary y, lightly)
    fig.add_trace(
        go.Scatter(
            x=x, y=(pbp["HOME_SCORE"] - pbp["AWAY_SCORE"]),
            mode="lines",
            name="Score lead (home)",
            yaxis="y2",
            opacity=0.35,
            hovertemplate="t=%{x:.0f}s • Lead=%{y}<extra></extra>",
        ),
        row=1, col=1
    )

    # Bonus shading per period
    for p in periods:
        # Shade when either side is in bonus for that period
        mask = pbp["PERIOD"] == p
        if mask.any():
            xs = pbp.loc[mask, "ELAPSED_S"].values
            hb = pbp.loc[mask, "HOME_IN_BONUS"].values
            ab = pbp.loc[mask, "AWAY_IN_BONUS"].values
            # If many points, compress into contiguous spans
            def spans(arr, xarr):
                spans_ = []
                on = False
                start = None
                for k in range(len(arr)):
                    if arr[k] and not on:
                        on = True; start = xarr[k]
                    if on and (k == len(arr)-1 or not arr[k+1]):
                        end = xarr[k]
                        spans_.append((start, end))
                        on = False
                return spans_

            for s, e in spans(hb, xs):
                fig.add_vrect(x0=s, x1=e, line_width=0, fillcolor="rgba(255,0,0,0.05)", layer="below", annotation_text=f"{home_abbr} bonus", annotation_position="top left")
            for s, e in spans(ab, xs):
                fig.add_vrect(x0=s, x1=e, line_width=0, fillcolor="rgba(0,0,255,0.05)", layer="below", annotation_text=f"{away_abbr} bonus", annotation_position="bottom left")

    # Foul markers
    foul_df = pbp[pbp["EVENTMSGTYPE"] == 6].copy()
    if not foul_df.empty:
        fig.add_trace(
            go.Scatter(
                x=foul_df["ELAPSED_S"],
                y=foul_df["HOME_WP"],
                mode="markers",
                name="Fouls",
                marker=dict(size=9, symbol="x"),
                hovertemplate=(
                    "Q%{customdata[0]} %{customdata[1]}<br>"
                    "%{customdata[2]}<br>"
                    "ΔWP(h): %{customdata[3]:+.2%}<extra></extra>"
                ),
                customdata=np.stack([
                    foul_df["PERIOD"].values,
                    foul_df["PCTIMESTRING"].values,
                    foul_df["FOUL_TYPE_TEXT"].fillna("FOUL").values,
                    foul_df["FOUL_DELTA_WP_HOME"].values
                ], axis=1)
            ),
            row=1, col=1
        )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(title="Elapsed (s)", rangemode="tozero"),
        yaxis=dict(title=f"Home WP ({home_abbr})", tickformat=".0%"),
        yaxis2=dict(overlaying="y", side="right", title="Home Lead", showgrid=False),
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Foul Impact Tables
# ----------------------------
st.markdown("### Foul impact (ΔWP)")

if len(fouls) == 0:
    st.info("No fouls recorded in this game (yet).")
else:
    f_df = pd.DataFrame([{
        "Q": f.period,
        "Time": f.pc_time,
        "Team": f.team,
        "Player": f.player,
        "Type": ("Shooting " if f.is_shooting else "") + (f.foul_type or "Foul"),
        "In Bonus": f.team_in_bonus,
        "Home WP (before)": f.home_wp_before,
        "Home WP (after)": f.home_wp_after,
        "ΔWP (home)": f.delta_wp_home
    } for f in fouls])

    # Sort by absolute swing
    top_swings = f_df.reindex(f_df["ΔWP (home)"].abs().sort_values(ascending=False).index).head(12)
    st.dataframe(
        top_swings.style.format({
            "Home WP (before)": "{:.1%}",
            "Home WP (after)": "{:.1%}",
            "ΔWP (home)": "{:+.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ----------------------------
# Player Foul Trouble Panel
# ----------------------------
st.markdown("### Player foul trouble")

ft = pbp[pbp["FOUL_TROUBLE_FLAG"] != ""]
if ft.empty:
    st.caption("No classic foul-trouble thresholds triggered (2-in-Q1, 3-by-Half, 5-in-Q4+).")
else:
    ft_small = ft[["PERIOD","PCTIMESTRING","FOUL_PLAYER","PLAYER1_TEAM_ABBREVIATION","FOUL_TROUBLE_FLAG","HOME_WP"]].copy()
    ft_small.rename(columns={
        "PERIOD":"Q", "PCTIMESTRING":"Time", "FOUL_PLAYER":"Player", "PLAYER1_TEAM_ABBREVIATION":"Team",
        "FOUL_TROUBLE_FLAG":"Flag", "HOME_WP":"Home WP"
    }, inplace=True)
    st.dataframe(
        ft_small.style.format({"Home WP":"{:.1%}"}),
        use_container_width=True,
        hide_index=True
    )

# ----------------------------
# Per-quarter team fouls (bonus awareness)
# ----------------------------
st.markdown("### Team fouls by quarter")
st.dataframe(team_fouls_by_p, use_container_width=True, hide_index=True)

# ----------------------------
# Claims (data-backed), adapted to current game context
# ----------------------------
st.markdown("### Game insights & coaching context")

def badge(text, kind="warn"):
    return f"<span class='badge {kind}'>{text}</span>"

insights = []
# Check if either side hit bonus early in a quarter
for p in periods:
    p_mask = pbp["PERIOD"] == p
    if not p_mask.any():
        continue
    first_idx = pbp.loc[p_mask].index[0]
    # detect first bonus onsets
    hb = pbp.loc[p_mask & (pbp['HOME_IN_BONUS'])].head(1)
    ab = pbp.loc[p_mask & (pbp['AWAY_IN_BONUS'])].head(1)
    if not hb.empty:
        t = hb["PCTIMESTRING"].iloc[0]
        insights.append(f"{badge('BONUS')} **{home_abbr}** entered the bonus in Q{p} at {t}, gifting free throws on defensive fouls for the rest of the quarter.")
    if not ab.empty:
        t = ab["PCTIMESTRING"].iloc[0]
        insights.append(f"{badge('BONUS')} **{away_abbr}** entered the bonus in Q{p} at {t}, gifting free throws on defensive fouls for the rest of the quarter.")

# Big ΔWP fouls
if len(fouls):
    swings = sorted(fouls, key=lambda x: abs(x.delta_wp_home), reverse=True)[:3]
    for s in swings:
        pov = "home"  # ΔWP displayed from home POV
        swing = f"{s.delta_wp_home:+.1%}"
        who = s.player or "Unknown"
        insights.append(f"{badge('ΔWP','ok')} Q{s.period} {s.pc_time}: **{who}** ({s.team}) **{s.foul_type}** — ΔWP(home) {swing}.")

if len(insights) == 0:
    st.caption("No notable foul-related insights detected yet.")
else:
    for line in insights:
        st.markdown(line, unsafe_allow_html=True)

# ----------------------------
# Footer: methodology
# ----------------------------
with st.expander("Methodology notes", expanded=False):
    st.markdown(
        "- **Data**: Free `nba_api` endpoints (play-by-play, box scores) for both regular season and playoffs.\n"
        "- **Win Probability**: Transparent logistic baseline using score differential and time remaining. It's not the official NBA WP feed, but tracks swings and is stable for exploratory use.\n"
        "- **ΔWP around fouls**: We use the next-event WP minus current WP as a quick proxy of the foul’s immediate impact.\n"
        "- **Bonus detection**: Team reaches *bonus* at 5th team foul in a quarter; we shade those spans.\n"
        "- **Foul trouble flags**: 2 fouls in Q1, 3 fouls by halftime, and 5 fouls in Q4+ (common coaching thresholds).\n"
        "- **Mobile UX**: Plotly line + markers; pinch-zoom; legend toggles; readable labels.\n"
    )
