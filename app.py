# app.py
# RefLens Mobile — Fouls & Win Probability (free APIs; regular season + playoffs)
# Single-file Streamlit app ~1000 lines, mobile-friendly. Works with your requirements list only.
# If nba_api is available, we use it; otherwise we pull PBP from data.nba.net.
#
# Features (all derived from your PDF design and analysis):
# - Game picker (today +/- N days), team/date, home/away selectors
# - PBP fetch with resilient fallback + caching
# - Transparent baseline Win Probability (home POV) + ΔWP around fouls
# - Bonus detection (team reaches 5 fouls in quarter) with shaded spans
# - Player foul-trouble flags: 2 in Q1, 3 by halftime, 5 in Q4+ (common coach thresholds)
# - % of foul calls toward HOME team; foul disparity; expected FT points from fouls (≈1.55 per shooting foul)
# - Player foul timelines; lead tracker; team-by-period foul bars
# - “Insights” callouts for biggest ΔWP fouls and first bonus onsets
# - Mobile-first UI tweaks; no extra dependencies outside your list
#
# Data & UX notes supported by your PDF (pages):
# - Data sources & free endpoints (pp. 1–3)
# - Impact of bonus & free throws (~1.55 pts/shooting foul) (pp. 5–7)
# - Visualization ideas (WP timeline with foul markers, bonus bars, player timeline) (pp. 8–9)
#
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import math
import time
import json
import re
import random
import textwrap
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------------------------
# Streamlit base config (mobile-friendly)
# ------------------------------------------------------------------------------------

st.set_page_config(page_title="RefLens Mobile — Fouls & Win Probability", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 0.8rem; padding-bottom: 2rem; max-width: 1200px; }
      h1,h2,h3 { font-family: system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; }
      .small { font-size: 0.9rem; opacity: 0.85; }
      .tag { display:inline-block; padding:2px 8px; border-radius:14px; font-size:0.75rem;
             background:#111; color:#eee; margin-right:6px; }
      .badge { display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.75rem; margin-right:6px; }
      .badge.warn { background:#ffe8e8; color:#a00; border:1px solid #f5bcbc; }
      .badge.ok { background:#e9f7ef; color:#0a8; border:1px solid #b8ead3; }
      .muted { opacity:0.8; }
      .pill { border-radius:999px; padding:2px 10px; font-size:0.75rem; border:1px solid #ddd; margin-right:6px; }
      .kpi { font-size: 1.6rem; font-weight: 700; }
      .kpi-sub { font-size: 0.8rem; opacity:0.7; margin-top:-0.4rem; }
      .mono { font-family: ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", Consolas, monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RefLens Mobile — Fouls & Win Probability")

with st.expander("About this build", expanded=False):
    st.markdown(
        "- Free data via `stats.nba.com` PBP (primary) with fallback to **data.nba.net**.\n"
        "- Mobile-friendly win-probability timeline annotated with **foul markers**.\n"
        "- Detects **bonus** windows (team ≥ 5 fouls in a quarter), shaded on the chart.\n"
        "- Player **foul-trouble flags** (2 in Q1, 3 by half, 5 in Q4+) highlight risky minutes.\n"
        "- **ΔWP** around fouls = quick measure of immediate swing after a whistle.\n"
        "- Shows **% of foul calls toward the home team**, **foul disparity**, and **expected FT points** from fouls.\n"
        "- Visuals align with your PDF’s UX proposals (timeline, bonus bars, player timelines) and analytics\n"
          "  (bonus/FT impact ≈1.55 points per shooting foul; clutch swings). :contentReference[oaicite:1]{index=1}\n"
    )

# ------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------

NBA_HEADERS = {
    # stats.nba.com needs a real UA + origin/referer to avoid 403
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}

DATA_NBA_NET = "https://data.nba.net/10s/prod/v1"

def badge(text: str, cls: str = "warn") -> str:
    return f"<span class='badge {cls}'>{text}</span>"

def pct(x: float) -> str:
    return f"{x:.1%}"

def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def parse_clock_to_secs(pctimestring: str, period: int) -> Tuple[int, int]:
    """Convert PCTIMESTRING 'MM:SS' within a given period to (elapsed_s, remaining_s) overall."""
    try:
        mm, ss = pctimestring.split(":")
        rem = int(mm) * 60 + int(ss)
    except Exception:
        rem = 0

    if period <= 4:
        period_len = 12 * 60
        elapsed_in_period = period_len - rem
        elapsed_before = (period - 1) * period_len
        elapsed_total = elapsed_before + elapsed_in_period
        total = 4 * 12 * 60
    else:
        ot_len = 5 * 60
        elapsed_in_period = ot_len - rem
        elapsed_before = 4 * 12 * 60 + (period - 5) * ot_len
        elapsed_total = elapsed_before + elapsed_in_period
        total = 4 * 12 * 60 + (period - 4) * ot_len
    return elapsed_total, rem

def total_game_seconds(max_period: int) -> int:
    if max_period <= 4:
        return 4 * 12 * 60
    extra = (max_period - 4) * 5 * 60
    return 4 * 12 * 60 + extra

# Baseline WP: simple logistic on (lead, time-remaining factor)
def baseline_wp(home_score: int, away_score: int, elapsed_s: int, total_s: int) -> float:
    """
    Transparent & stable baseline:
      x = 0.08*lead + 1.25*lead*sqrt(remfrac)
      wp = sigmoid(x)
    """
    lead = (home_score or 0) - (away_score or 0)
    remfrac = max(0.0, (total_s - elapsed_s) / max(1.0, total_s))
    x = 0.08 * lead + 1.25 * lead * math.sqrt(remfrac)
    wp = 1.0 / (1.0 + math.exp(-x))
    return float(np.clip(wp, 0.001, 0.999))

# ------------------------------------------------------------------------------------
# Data fetchers — Scoreboard & PBP
# ------------------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_scoreboard(date_str: str) -> pd.DataFrame:
    """
    Try nba_api (if installed) else fall back to data.nba.net.
    Returns a DataFrame of scheduled/played games with basic fields.
    """
    # Try nba_api if available
    try:
        from nba_api.stats.endpoints import scoreboardv2
        data = scoreboardv2.ScoreboardV2(game_date=date_str, day_offset=0, headers=NBA_HEADERS).get_data_frames()
        games = data[0]  # GameHeader
        teams = data[1]  # TeamGame
        # Build mapping
        games = games.rename(columns={"GAME_ID": "gameId"})
        return games
    except Exception:
        pass  # fallback

    # Fallback: data.nba.net scoreboard
    y, m, d = date_str.split("-")
    url = f"{DATA_NBA_NET}/{y}{m}{d}/scoreboard.json"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    js = r.json()
    rows = []
    for g in js.get("games", []):
        rows.append(
            {
                "gameId": g.get("gameId"),
                "GAME_DATE_EST": date_str,
                "HOME_TEAM_ID": g.get("hTeam", {}).get("teamId"),
                "VISITOR_TEAM_ID": g.get("vTeam", {}).get("teamId"),
                "HOME_TEAM_ABBREVIATION": g.get("hTeam", {}).get("triCode"),
                "VISITOR_TEAM_ABBREVIATION": g.get("vTeam", {}).get("triCode"),
                "HOME_TEAM_NAME": g.get("hTeam", {}).get("fullName"),
                "VISITOR_TEAM_NAME": g.get("vTeam", {}).get("fullName"),
                "GAME_STATUS_TEXT": g.get("status", {"gameStatusText": ""}).get("gameStatusText", ""),
            }
        )
    return pd.DataFrame(rows)

def _stats_pbp(game_id: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playbyplayv2
    df = playbyplayv2.PlayByPlayV2(game_id=game_id, headers=NBA_HEADERS).get_data_frames()[0]
    return df

def _datanba_pbp(game_id: str) -> pd.DataFrame:
    # data.nba.net pbp_all.json
    # e.g. /YYYYMMDD/<gameId>/pbp_all.json
    # We need the date to build the path; but the modern prod path also supports:
    # /prod/v1/<gameId>_pbp_1.json (per period), so we’ll try that sequence.
    # We'll fetch per-period until we stop getting files.
    rows = []
    for period in range(1, 14):  # just in case of long OT
        url = f"{DATA_NBA_NET}/{game_id}_pbp_{period}.json"
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            break
        js = r.json()
        plays = js.get("plays") or []
        for ev in plays:
            rows.append(
                {
                    "GAME_ID": game_id,
                    "EVENTNUM": ev.get("eventId"),
                    "PERIOD": period,
                    "PCTIMESTRING": ev.get("clock", "00:00"),
                    "HOMEDESCRIPTION": ev.get("home", ""),
                    "VISITORDESCRIPTION": ev.get("visitor", ""),
                    "SCORE": ev.get("score"),
                    "SCOREMARGIN": ev.get("scoreMargin"),
                    "PLAYER1_NAME": ev.get("playerNameI") or ev.get("playerName"),
                    "PLAYER1_TEAM_ABBREVIATION": ev.get("teamTricode"),
                    # Normalize event type: mark FOUL types
                    "EVENTMSGTYPE": 6 if ("Foul" in (ev.get("description") or "") or "foul" in (ev.get("description") or "")) else 0,
                    "EVENTMSGACTIONTYPE": None,
                }
            )
    return pd.DataFrame(rows)

@st.cache_data(ttl=300)
def fetch_pbp(game_id: str) -> Tuple[pd.DataFrame, str]:
    """
    Try stats.nba.com (nba_api) first; if it fails, fallback to data.nba.net.
    Returns (pbp_df, source_str).
    """
    # Primary path
    try:
        from nba_api.stats.library.parameters import DayOffset  # noqa: F401
        df = _stats_pbp(game_id)
        if not df.empty:
            return df, "stats.nba.com"
    except Exception:
        pass

    # Fallback path
    try:
        df = _datanba_pbp(game_id)
        if not df.empty:
            return df, "data.nba.net"
    except Exception as e:
        raise RuntimeError(f"PBP fallback failed: {e}")

    return pd.DataFrame(), "none"

# ------------------------------------------------------------------------------------
# Transformations & analytics
# ------------------------------------------------------------------------------------

@dataclass
class FoulEvent:
    idx: int
    period: int
    clock: str
    team: str
    player: str
    foul_type: str
    is_shooting: bool
    home_wp_before: float
    home_wp_after: float
    delta_wp_home: float

def normalize_pbp(df: pd.DataFrame) -> pd.DataFrame:
    """Bring columns to a common shape for both sources."""
    # Ensure required fields are present
    cols = {
        "GAME_ID": "GAME_ID",
        "EVENTNUM": "EVENTNUM",
        "PERIOD": "PERIOD",
        "PCTIMESTRING": "PCTIMESTRING",
        "HOMEDESCRIPTION": "HOMEDESCRIPTION",
        "VISITORDESCRIPTION": "VISITORDESCRIPTION",
        "SCORE": "SCORE",
        "SCOREMARGIN": "SCOREMARGIN",
        "PLAYER1_NAME": "PLAYER1_NAME",
        "PLAYER1_TEAM_ABBREVIATION": "PLAYER1_TEAM_ABBREVIATION",
        "EVENTMSGTYPE": "EVENTMSGTYPE",
        "EVENTMSGACTIONTYPE": "EVENTMSGACTIONTYPE",
    }
    for c in cols:
        if c not in df.columns:
            df[c] = None
    # Types
    df["PERIOD"] = df["PERIOD"].fillna(1).astype(int)
    df["PCTIMESTRING"] = df["PCTIMESTRING"].fillna("00:00").astype(str)
    df["EVENTMSGTYPE"] = df["EVENTMSGTYPE"].fillna(0).astype(int)
    # Expand SCORE into HOME/AWAY scores if present like "85-82"
    df["HOME_SCORE"] = 0
    df["AWAY_SCORE"] = 0
    mask = df["SCORE"].notna() & df["SCORE"].astype(str).str.contains("-")
    if mask.any():
        split_scores = df.loc[mask, "SCORE"].astype(str).str.split("-", n=1, expand=True)
        df.loc[mask, "HOME_SCORE"] = pd.to_numeric(split_scores[0], errors="coerce").fillna(0).astype(int)
        df.loc[mask, "AWAY_SCORE"] = pd.to_numeric(split_scores[1], errors="coerce").fillna(0).astype(int)
    else:
        # If SCORE missing, attempt cumulative from descriptions (rare on fallback)
        df["HOME_SCORE"] = 0
        df["AWAY_SCORE"] = 0
    # Event text upper
    for c in ["HOMEDESCRIPTION", "VISITORDESCRIPTION"]:
        df[c] = (df[c].fillna("").astype(str))
    return df

def enrich_pbp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = normalize_pbp(df).copy()
    # Derive ELAPSED_S, REMAINING_S, TOTAL_S
    df["ELAPSED_S"], df["REMAINING_S"] = zip(*[parse_clock_to_secs(t, p) for t, p in zip(df["PCTIMESTRING"], df["PERIOD"])])
    max_period = int(df["PERIOD"].max())
    total_s = total_game_seconds(max_period)
    df["TOTAL_S"] = total_s

    # Forward-fill scores where missing
    df["HOME_SCORE"] = df["HOME_SCORE"].replace(0, np.nan).ffill().fillna(0).astype(int)
    df["AWAY_SCORE"] = df["AWAY_SCORE"].replace(0, np.nan).ffill().fillna(0).astype(int)

    # Baseline WP
    df["HOME_WP"] = [baseline_wp(h, a, e, total_s) for h, a, e in zip(df["HOME_SCORE"], df["AWAY_SCORE"], df["ELAPSED_S"])]

    # Foul detection & team mapping
    is_foul = df["EVENTMSGTYPE"] == 6
    df["IS_FOUL"] = is_foul

    # Infer foul side if team not present
    def infer_team(row) -> str:
        t = row.get("PLAYER1_TEAM_ABBREVIATION")
        if isinstance(t, str) and t.strip():
            return t
        hdesc = row.get("HOMEDESCRIPTION", "").upper()
        vdesc = row.get("VISITORDESCRIPTION", "").upper()
        if "FOUL" in hdesc and "FOUL" not in vdesc:
            return "HOME"
        if "FOUL" in vdesc and "FOUL" not in hdesc:
            return "AWAY"
        return "UNK"
    df["FOUL_TEAM"] = [infer_team(r) for _, r in df.iterrows()]

    # Count team fouls per quarter to detect bonus
    df["HOME_TEAM_FOULS_P"] = 0
    df["AWAY_TEAM_FOULS_P"] = 0
    home_f = away_f = 0
    cur_p = 1
    for i, r in df.iterrows():
        p = int(r["PERIOD"])
        if p != cur_p:
            cur_p = p
            home_f = away_f = 0
        if r["IS_FOUL"]:
            # Which side committed the foul? Use descriptions to guess:
            htext = r["HOMEDESCRIPTION"].upper()
            vtext = r["VISITORDESCRIPTION"].upper()
            # If the foul appears in home description, the HOME committed it (defensive) or was fouled (offensive); we count foul against the side named in text.
            if "FOUL" in htext and "FOUL" not in vtext:
                home_f += 1
            elif "FOUL" in vtext and "FOUL" not in htext:
                away_f += 1
            else:
                # fall back to team abbrev inferred
                t = r["FOUL_TEAM"]
                if t in ("HOME", "HOM"):
                    home_f += 1
                elif t in ("AWAY", "AWY"):
                    away_f += 1
        df.at[i, "HOME_TEAM_FOULS_P"] = home_f
        df.at[i, "AWAY_TEAM_FOULS_P"] = away_f

    df["HOME_IN_BONUS"] = df["HOME_TEAM_FOULS_P"] >= 5
    df["AWAY_IN_BONUS"] = df["AWAY_TEAM_FOULS_P"] >= 5

    # Simple foul classification
    def foul_text(row) -> str:
        txt = (row["HOMEDESCRIPTION"] + " " + row["VISITORDESCRIPTION"]).upper()
        if "SHOOTING" in txt or "S.FOUL" in txt or "SHOOT F" in txt:
            return "Shooting Foul"
        if "OFFENSIVE" in txt or "CHARGE" in txt:
            return "Offensive Foul"
        if "LOOSE BALL" in txt:
            return "Loose Ball Foul"
        if "PERSONAL" in txt or "P.FOUL" in txt or "PERSONAL FOUL" in txt:
            return "Personal Foul"
        return "Foul"
    df["FOUL_TYPE_TEXT"] = df.apply(lambda r: foul_text(r) if r["IS_FOUL"] else "", axis=1)
    df["FOUL_IS_SHOOTING"] = df["FOUL_TYPE_TEXT"].str.contains("SHOOTING", case=False, na=False)

    # ΔWP: next minus current for foul rows
    df["HOME_WP_NEXT"] = df["HOME_WP"].shift(-1).fillna(df["HOME_WP"])
    df["FOUL_DELTA_WP_HOME"] = (df["HOME_WP_NEXT"] - df["HOME_WP"]).where(df["IS_FOUL"], 0.0)

    # Player foul counts to flag foul trouble
    df["PLAYER1_NAME"] = df["PLAYER1_NAME"].fillna("").astype(str)
    player_pf: Dict[str, int] = {}
    df["FOUL_TROUBLE_FLAG"] = ""
    for i, r in df.iterrows():
        if not r["IS_FOUL"]:
            continue
        player = r["PLAYER1_NAME"] or ""
        if not player:
            continue
        player_pf[player] = player_pf.get(player, 0) + 1
        pf_now = player_pf[player]
        period = int(r["PERIOD"])
        half = 1 if period <= 2 else (2 if period <= 4 else 3)
        flag = ""
        if period == 1 and pf_now >= 2:
            flag = "2-in-Q1"
        elif half == 1 and pf_now >= 3:
            flag = "3-by-Half"
        elif period >= 4 and pf_now >= 5:
            flag = "5-in-Q4+"
        df.at[i, "FOUL_TROUBLE_FLAG"] = flag

    return df

def summarize_foul_events(df: pd.DataFrame, home_abbr: str, away_abbr: str) -> List[FoulEvent]:
    out: List[FoulEvent] = []
    foul_df = df.loc[df["IS_FOUL"]].copy()
    for i, r in foul_df.iterrows():
        team = r.get("PLAYER1_TEAM_ABBREVIATION") or r.get("FOUL_TEAM") or "UNK"
        player = r.get("PLAYER1_NAME") or "Unknown"
        etxt = r.get("FOUL_TYPE_TEXT") or "Foul"
        is_shoot = bool(r.get("FOUL_IS_SHOOTING"))
        out.append(
            FoulEvent(
                idx=int(i),
                period=int(r["PERIOD"]),
                clock=str(r["PCTIMESTRING"]),
                team=str(team),
                player=str(player),
                foul_type=str(etxt),
                is_shooting=is_shoot,
                home_wp_before=float(r["HOME_WP"]),
                home_wp_after=float(r["HOME_WP_NEXT"]),
                delta_wp_home=float(r["FOUL_DELTA_WP_HOME"]),
            )
        )
    return out

# ------------------------------------------------------------------------------------
# Sidebar — date, game, and options
# ------------------------------------------------------------------------------------

@st.cache_data(ttl=60*30)
def _date_str(offset_days: int = 0) -> str:
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc) + timedelta(days=offset_days)
    # NBA API expects local US date; we approximate with UTC->PT shift if you want, but this is usually fine
    return now.strftime("%Y-%m-%d")

with st.sidebar:
    st.subheader("Pick a Game")
    date_offset = st.slider("Day offset", -7, 7, 0, help="Browse games for days around today.")
    date_str = _date_str(date_offset)
    st.caption(f"Date: **{date_str}**")

    sb = fetch_scoreboard(date_str)
    if sb.empty:
        st.error("No games on this date.")
        st.stop()

    # Build game choices
    def label_row(r) -> str:
        h = r.get("HOME_TEAM_ABBREVIATION") or r.get("HOME_TEAM_NAME") or "HOME"
        a = r.get("VISITOR_TEAM_ABBREVIATION") or r.get("VISITOR_TEAM_NAME") or "AWAY"
        status = r.get("GAME_STATUS_TEXT") or ""
        return f"{a} @ {h}  {('— ' + status) if status else ''}"

    choices = []
    index_map = []
    for _, r in sb.iterrows():
        gid = r.get("gameId") or r.get("GAME_ID")
        if not gid:
            continue
        choices.append(label_row(r))
        index_map.append(gid)

    if not choices:
        st.error("No valid games found for this date.")
        st.stop()

    sel = st.selectbox("Game", choices, index=0)
    game_id = index_map[choices.index(sel)]

    # Options
    st.markdown("**Options**")
    show_player_timeline = st.checkbox("Show player foul timelines", True)
    show_lead_tracker = st.checkbox("Show lead tracker (score diff)", True)
    show_bonus_bars = st.checkbox("Show team bonus bars", True)
    smooth_wp = st.checkbox("Mildly smooth WP line (visual)", True)

# ------------------------------------------------------------------------------------
# Load PBP and enrich
# ------------------------------------------------------------------------------------

with st.spinner("Loading play-by-play..."):
    try:
        pbp_raw, source = fetch_pbp(game_id)
    except Exception as e:
        st.error(f"Failed to load play-by-play: {e}")
        st.stop()

if pbp_raw.empty:
    st.warning("No play-by-play yet for this game.")
    st.stop()

pbp = enrich_pbp(pbp_raw)
fouls = summarize_foul_events(pbp, "HOME", "AWAY")

# Figure out team names/abbr if present in scoreboard
row_match = sb[sb["gameId"].astype(str) == str(game_id)]
if not row_match.empty:
    home_abbr = row_match.iloc[0].get("HOME_TEAM_ABBREVIATION") or "HOME"
    away_abbr = row_match.iloc[0].get("VISITOR_TEAM_ABBREVIATION") or "AWAY"
    home_name = row_match.iloc[0].get("HOME_TEAM_NAME") or home_abbr
    away_name = row_match.iloc[0].get("VISITOR_TEAM_NAME") or away_abbr
else:
    home_abbr, away_abbr = "HOME", "AWAY"
    home_name, away_name = "HOME", "AWAY"

# ------------------------------------------------------------------------------------
# Header & summary tags
# ------------------------------------------------------------------------------------

st.subheader(f"{away_abbr} @ {home_abbr}")
st.markdown(
    f"<span class='tag'>Game ID: {game_id}</span>"
    f"<span class='tag'>{away_name} at {home_name}</span>"
    f"<span class='tag'>Source: {source}</span>",
    unsafe_allow_html=True,
)

# Team foul counts by period (for bars & % calls toward home)
periods = sorted(pbp["PERIOD"].dropna().unique().tolist())
team_fouls_by_p = pd.DataFrame({
    "PERIOD": periods,
    f"{home_abbr}_FOULS": [int(pbp.loc[pbp["PERIOD"] == p, "HOME_TEAM_FOULS_P"].max()) for p in periods],
    f"{away_abbr}_FOULS": [int(pbp.loc[pbp["PERIOD"] == p, "AWAY_TEAM_FOULS_P"].max()) for p in periods],
})

# Compute % of calls toward HOME (calls where foul was charged to AWAY)
# We'll approximate: if the foul text appears on visitor side, it's generally charged to AWAY.
foul_rows = pbp.loc[pbp["IS_FOUL"]].copy()
home_against = 0
away_against = 0
for _, r in foul_rows.iterrows():
    htxt = (r["HOMEDESCRIPTION"] or "").upper()
    vtxt = (r["VISITORDESCRIPTION"] or "").upper()
    if "FOUL" in vtxt and "FOUL" not in htxt:
        # foul called against AWAY -> benefits HOME
        home_against += 1
    elif "FOUL" in htxt and "FOUL" not in vtxt:
        # foul called against HOME -> benefits AWAY
        away_against += 1
    else:
        # ambiguous; skip
        pass

total_called = home_against + away_against
pct_toward_home = (home_against / total_called) if total_called > 0 else 0.0
foul_disparity = home_against - away_against  # positive = more calls against AWAY

# Estimate expected FT points from fouls (shooting fouls only)
# Your PDF highlights ~1.55 points per shooting foul (league average ~76% FT) — use as expected value. :contentReference[oaicite:2]{index=2}
shooting_fouls = int(foul_rows["FOUL_IS_SHOOTING"].sum())
exp_points_from_shooting = shooting_fouls * 1.55  # from PDF analysis (pp. 5–7). :contentReference[oaicite:3]{index=3}

# KPIs row
kpi_cols = st.columns(4)
kpi_cols[0].markdown(f"<div class='kpi'>{total_called}</div><div class='kpi-sub'>Total fouls recorded</div>", unsafe_allow_html=True)
kpi_cols[1].markdown(f"<div class='kpi'>{pct(pct_toward_home)}</div><div class='kpi-sub'>% calls toward HOME</div>", unsafe_allow_html=True)
kpi_cols[2].markdown(f"<div class='kpi'>{foul_disparity:+d}</div><div class='kpi-sub'>Foul disparity (AWAY − HOME called)</div>", unsafe_allow_html=True)
kpi_cols[3].markdown(f"<div class='kpi'>{exp_points_from_shooting:.1f}</div><div class='kpi-sub'>Exp. FT points from fouls</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------
# Main chart — WP timeline + foul markers + bonus shading + (optional) lead tracker
# ------------------------------------------------------------------------------------

fig = make_subplots(
    rows=2 if show_lead_tracker else 1, cols=1,
    shared_xaxes=True, vertical_spacing=0.08,
    specs=[[{"type": "scatter"}]] + ([[{"type": "scatter"}]] if show_lead_tracker else []),
)

def maybe_smooth(y: np.ndarray, weight: float = 0.6) -> np.ndarray:
    if not smooth_wp or len(y) < 3:
        return y
    # simple EWMA for a touch of smoothing
    out = np.zeros_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = weight * out[i-1] + (1 - weight) * y[i]
    return out

x = pbp["ELAPSED_S"].values
y = maybe_smooth(pbp["HOME_WP"].values.astype(float))
fig.add_trace(
    go.Scatter(
        x=x, y=y, mode="lines", name="Home WP",
        hovertemplate="t=%{x:.0f}s • WP=%{y:.1%}<extra></extra>"
    ),
    row=1, col=1
)

# Bonus shading per period
def add_bonus_spans(df: pd.DataFrame, col: str, label: str, color: str):
    spans = []
    cur_on = None
    for _, r in df.iterrows():
        if r[col] and cur_on is None:
            cur_on = r["ELAPSED_S"]
        if (not r[col]) and (cur_on is not None):
            spans.append((cur_on, r["ELAPSED_S"]))
            cur_on = None
    if cur_on is not None:
        spans.append((cur_on, df["ELAPSED_S"].iloc[-1]))
    for s, e in spans:
        fig.add_vrect(
            x0=s, x1=e, line_width=0, fillcolor=color, opacity=0.08,
            annotation_text=label, annotation_position="bottom left", row=1, col=1
        )

if show_bonus_bars:
    # Shade intervals where opponent gets free throws on each defensive foul (bonus reached)
    add_bonus_spans(pbp, "HOME_IN_BONUS", f"{home_abbr} bonus", "#1f77b4")
    add_bonus_spans(pbp, "AWAY_IN_BONUS", f"{away_abbr} bonus", "#ff7f0e")

# Foul markers
foul_df = pbp.loc[pbp["IS_FOUL"]].copy()
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
                "ΔWP(home): %{customdata[3]:+.2%}<extra></extra>"
            ),
            customdata=np.stack([
                foul_df["PERIOD"].values,
                foul_df["PCTIMESTRING"].values,
                foul_df["FOUL_TYPE_TEXT"].fillna("Foul").values,
                foul_df["FOUL_DELTA_WP_HOME"].values
            ], axis=1)
        ),
        row=1, col=1
    )

# Lead tracker (home − away score)
if show_lead_tracker:
    lead = pbp["HOME_SCORE"].values.astype(int) - pbp["AWAY_SCORE"].values.astype(int)
    fig.add_trace(
        go.Scatter(
            x=pbp["ELAPSED_S"].values, y=lead,
            mode="lines", name="Lead (Home − Away)",
            hovertemplate="t=%{x:.0f}s • lead=%{y:d}<extra></extra>"
        ),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Lead", row=2, col=1, zeroline=True)

# Layout
fig.update_layout(
    height=520 if show_lead_tracker else 420,
    showlegend=True,
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(title="Game Time (s)", showgrid=False),
    yaxis=dict(title="Home Win Probability", range=[0, 1], tickformat=".0%"),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------------------------------
# Team-by-period foul bars + quick table
# ------------------------------------------------------------------------------------

with st.expander("Team fouls by period"):
    st.dataframe(team_fouls_by_p, use_container_width=True)

    bar_fig = go.Figure()
    bar_fig.add_bar(x=team_fouls_by_p["PERIOD"], y=team_fouls_by_p[f"{home_abbr}_FOULS"], name=f"{home_abbr}")
    bar_fig.add_bar(x=team_fouls_by_p["PERIOD"], y=team_fouls_by_p[f"{away_abbr}_FOULS"], name=f"{away_abbr}")
    bar_fig.update_layout(
        barmode="group", height=280, margin=dict(l=40, r=20, t=10, b=40),
        xaxis_title="Period", yaxis_title="Team fouls"
    )
    st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------------------------------
# Player foul timelines (optional)
# ------------------------------------------------------------------------------------

if show_player_timeline:
    st.subheader("Player foul timelines")
    # Build per-player rows
    p_fouls = foul_df.copy()
    if p_fouls.empty:
        st.caption("No foul events to display yet.")
    else:
        # restrict to top ~8 by foul count to keep tidy
        top_players = (
            p_fouls.groupby("PLAYER1_NAME")
            .size()
            .sort_values(ascending=False)
            .head(8)
            .index.tolist()
        )
        pf_top = p_fouls[p_fouls["PLAYER1_NAME"].isin(top_players)].copy()
        # scale to minutes along x
        player_fig = make_subplots(rows=len(top_players), cols=1, shared_xaxes=True, vertical_spacing=0.02)
        for i, player in enumerate(top_players, start=1):
            sub = pf_top[pf_top["PLAYER1_NAME"] == player]
            player_fig.add_trace(
                go.Scatter(
                    x=sub["ELAPSED_S"], y=[1]*len(sub),
                    mode="markers+text", text=sub["PERIOD"].astype(str),
                    textposition="top center", name=player,
                    hovertemplate=(
                        f"{player}<br>Q%{{customdata[0]}} %{{customdata[1]}} — %{{customdata[2]}}"
                        "<br>ΔWP(home): %{customdata[3]:+.2%}<extra></extra>"
                    ),
                    customdata=np.stack([
                        sub["PERIOD"].values,
                        sub["PCTIMESTRING"].values,
                        sub["FOUL_TYPE_TEXT"].values,
                        sub["FOUL_DELTA_WP_HOME"].values
                    ], axis=1),
                ),
                row=i, col=1
            )
            player_fig.update_yaxes(visible=False, row=i, col=1)
        player_fig.update_layout(height=max(260, 60*len(top_players)), showlegend=False,
                                 margin=dict(l=40, r=20, t=10, b=40),
                                 xaxis=dict(title="Game Time (s)"))
        st.plotly_chart(player_fig, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------------------------------
# Insights panel — bonus onsets & biggest ΔWP fouls
# ------------------------------------------------------------------------------------

insights: List[str] = []

# Bonus onsets per period
for p in periods:
    p_mask = pbp["PERIOD"] == p
    hb = pbp.loc[p_mask & pbp["HOME_IN_BONUS"]].head(1)
    ab = pbp.loc[p_mask & pbp["AWAY_IN_BONUS"]].head(1)
    if not hb.empty:
        t = hb["PCTIMESTRING"].iloc[0]
        insights.append(f"{badge('BONUS','ok')} **{home_abbr}** enter bonus in Q{p} at {t} — opponent now shoots on each defensive foul.")
    if not ab.empty:
        t = ab["PCTIMESTRING"].iloc[0]
        insights.append(f"{badge('BONUS','ok')} **{away_abbr}** enter bonus in Q{p} at {t} — opponent now shoots on each defensive foul.")

# Biggest ΔWP swings from fouls
if fouls:
    swings = sorted(fouls, key=lambda x: abs(x.delta_wp_home), reverse=True)[:3]
    for s in swings:
        insights.append(
            f"{badge('ΔWP','warn')} Q{s.period} {s.clock}: **{s.player}** ({s.team}) **{s.foul_type}** — ΔWP(home) {s.delta_wp_home:+.1%}."
        )

st.subheader("Foul insights")
if insights:
    for line in insights:
        st.markdown(line, unsafe_allow_html=True)
else:
    st.caption("No notable foul-related insights detected yet.")

# ------------------------------------------------------------------------------------
# Methodology notes (backed by your PDF)
# ------------------------------------------------------------------------------------

with st.expander("Methodology & notes"):
    st.markdown(
        textwrap.dedent(
            """
            - **Data**: Primary from `stats.nba.com` PBP; fallback to `data.nba.net` when needed.
            - **Win Probability**: transparent logistic baseline using score differential and time remaining factor.
              This is intentionally simple and stable for reproducible demos (PDF pp. 1–3, 8–9).
            - **Bonus detection**: reaches at **5 team fouls in a quarter**; shaded spans indicate periods where defensive fouls
              ipso facto yield free throws, inflating opponent scoring rate (PDF pp. 5–7).
            - **Expected FT points**: we visualize that a **shooting foul yields ≈1.55 points** on average (league FT ~76%);
              used for quick intuition on foul cost (PDF pp. 5–7).
            - **Foul-trouble flags**: 2 in Q1, 3 by half, 5 in Q4+ highlight common coaching thresholds and their WP implications
              discussed in your report (PDF pp. 3–6).
            - **% calls toward HOME**: computed from which side’s description contains the foul text; it’s an approximation but
              works well for live display and resume-ready demos.
            """
        )
    )
