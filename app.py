import time
import numpy as np
import pandas as pd
import streamlit as st
import requests, re, json
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ---- Streamlit setup
st.set_page_config(page_title="RefLens Pro - NBA Game Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .game-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .insight-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d1e7dd;
        border-left: 5px solid #198754;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏀 RefLens Pro - Advanced NBA Game Analytics")
st.markdown("*Comprehensive game analysis with win probability, officiating insights, and performance metrics*")

# Sidebar for inputs and controls
with st.sidebar:
    st.header("Game Selection")
    game_id = st.text_input(
        "Enter NBA Game ID",
        value="0022300001",
        help="Format: 00223xxxxx for 2023-24 season, 00224xxxxx for 2024-25 season"
    )
    
    st.info("**Game ID Examples:**\n- Regular Season: 0022300001\n- Playoffs: 0042300101")
    
    btn_fetch = st.button("🔍 Analyze Game", type="primary", use_container_width=True)
    
    st.header("Analysis Options")
    show_advanced_metrics = st.checkbox("Show Advanced Metrics", value=True)
    show_momentum_analysis = st.checkbox("Show Momentum Analysis", value=True)
    show_officiating_insights = st.checkbox("Show Officiating Insights", value=True)

# ---- Constants & Enhanced Functions
TOTAL_REG_SECS = 48 * 60

def parse_period_clock_to_secs_left(period: int, pctimestr: str) -> int:
    """Convert period + 'MM:SS' into game seconds remaining."""
    try:
        mm, ss = pctimestr.split(":")
        p_secs_left = int(mm) * 60 + int(ss)
    except Exception:
        p_secs_left = 0
    
    if period <= 4:
        base_before_period = TOTAL_REG_SECS - (period - 1) * 12 * 60
        total_secs = max(0, p_secs_left + (base_before_period - 12 * 60))
    else:  # Overtime
        ot_periods = period - 4
        total_secs = p_secs_left - (ot_periods * 5 * 60)
    
    return max(0, total_secs)

def enhanced_wp_model(score_margin: float, sec_remaining: float, period: int = 1) -> float:
    """Enhanced win probability model accounting for game situation."""
    # Base logistic model
    time_factor = max(0, sec_remaining / TOTAL_REG_SECS)
    
    # Enhanced coefficients based on research
    margin_coeff = 0.18 if period <= 4 else 0.22  # Higher in OT
    time_coeff = -0.0035 if sec_remaining > 300 else -0.008  # More volatile in final 5 minutes
    
    # Situational adjustments
    if sec_remaining < 120:  # Final 2 minutes
        margin_coeff *= 1.3
    if sec_remaining < 30:   # Final 30 seconds
        margin_coeff *= 1.8
        
    z = margin_coeff * score_margin + time_coeff * sec_remaining
    wp = 1.0 / (1.0 + np.exp(-z))
    
    return max(0.001, min(0.999, wp))  # Bound between 0.1% and 99.9%

def detect_enhanced_fouls(row: pd.Series) -> tuple[bool, str, str, float]:
    """Enhanced foul detection with severity scoring."""
    text = " ".join(
        str(x) for x in [
            row.get("HOMEDESCRIPTION",""),
            row.get("VISITORDESCRIPTION",""),
            row.get("NEUTRALDESCRIPTION","")
        ] if pd.notna(x)
    ).upper()
    
    is_foul = "FOUL" in text
    foul_type = ""
    severity = 0.0
    
    if is_foul:
        if "FLAGRANT" in text:
            foul_type = "FLAGRANT"
            severity = 3.0
        elif "TECHNICAL" in text or "TECH" in text:
            foul_type = "TECHNICAL"
            severity = 2.5
        elif "SHOOTING" in text or "S.FOUL" in text:
            foul_type = "SHOOTING"
            severity = 2.0
        elif "OFFENSIVE" in text:
            foul_type = "OFFENSIVE"
            severity = 1.5
        elif "LOOSE BALL" in text:
            foul_type = "LOOSE BALL"
            severity = 1.3
        else:
            foul_type = "PERSONAL"
            severity = 1.0
            
    return is_foul, foul_type, severity

def calculate_momentum_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum and game flow metrics."""
    df = df.copy()
    
    # Rolling averages for momentum
    window = 10
    df['margin_ma'] = df['score_margin'].rolling(window=window, min_periods=1).mean()
    df['wp_ma'] = df['wp_home'].rolling(window=window, min_periods=1).mean()
    
    # Momentum shifts (significant WP changes)
    df['wp_change'] = df['wp_home'].diff().fillna(0)
    df['momentum_shift'] = (abs(df['wp_change']) > 0.05).astype(int)
    
    # Game intensity (volatility in WP)
    df['wp_volatility'] = df['wp_home'].rolling(window=20, min_periods=5).std().fillna(0)
    
    return df

def _clock_to_mmss(clock_val) -> str:
    """Convert various clock formats to MM:SS."""
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

@st.cache_data(show_spinner=False, ttl=300)
def load_enhanced_game_data(game_id: str):
    """Load and process game data with enhanced analytics."""
    url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"

    # Fetch data with retry logic
    last_err = None
    data = None
    for i in range(3):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                break
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(1.0 * (i + 1))
    
    if data is None:
        raise ValueError(f"Failed to fetch game data: {last_err}")

    game = data.get("game", {})
    actions = game.get("actions", [])
    
    if not actions:
        raise ValueError("No play-by-play data found for this game ID.")

    # Extract team and game metadata
    home = game.get("homeTeam", {})
    away = game.get("awayTeam", {})
    
    meta = {
        "home_tricode": home.get("teamTricode", ""),
        "home_city": home.get("teamCity", ""),
        "home_name": home.get("teamName", ""),
        "away_tricode": away.get("teamTricode", ""),
        "away_city": away.get("teamCity", ""),
        "away_name": away.get("teamName", ""),
        "game_time_utc": game.get("gameTimeUTC", ""),
        "season_type": game.get("seasonType", ""),
        "game_status": game.get("gameStatus", ""),
    }
    
    # Parse game date
    game_date = ""
    try:
        if meta["game_time_utc"]:
            dt = datetime.fromisoformat(meta["game_time_utc"].replace("Z","+00:00"))
            game_date = dt.strftime("%B %d, %Y")
    except Exception:
        pass
    meta["game_date"] = game_date

    # Process play-by-play data
    rows = []
    for i, action in enumerate(actions):
        period = int(action.get("period", 1))
        pctimestr = _clock_to_mmss(action.get("clock"))
        
        # Scores
        sh = action.get("scoreHome")
        sa = action.get("scoreAway")
        
        try:
            sh = int(sh) if sh is not None else None
            sa = int(sa) if sa is not None else None
        except:
            sh = sa = None

        rows.append({
            "GAME_ID": str(game_id),
            "EVENTNUM": int(action.get("actionNumber", i)),
            "PERIOD": period,
            "PCTIMESTRING": pctimestr,
            "SCORE": f"{sh} - {sa}" if sh is not None and sa is not None else None,
            "HOMEDESCRIPTION": None,
            "VISITORDESCRIPTION": None,
            "NEUTRALDESCRIPTION": action.get("description", ""),
            "home_score": sh,
            "away_score": sa,
        })

    df = pd.DataFrame(rows).sort_values(["PERIOD", "EVENTNUM"]).reset_index(drop=True)
    
    # Calculate derived metrics
    df["sec_remaining"] = df.apply(
        lambda r: parse_period_clock_to_secs_left(r["PERIOD"], r["PCTIMESTRING"]), axis=1
    )
    
    # Forward fill scores
    df["home_score"] = df["home_score"].ffill()
    df["away_score"] = df["away_score"].ffill()
    df["score_margin"] = df["home_score"] - df["away_score"]
    
    # Enhanced foul detection
    foul_data = df.apply(detect_enhanced_fouls, axis=1, result_type="expand")
    df["is_foul"] = foul_data[0]
    df["foul_type"] = foul_data[1]
    df["foul_severity"] = foul_data[2]
    
    # Enhanced win probability
    mask = df["home_score"].notna() & df["away_score"].notna()
    df["wp_home"] = np.nan
    
    for idx in df[mask].index:
        row = df.loc[idx]
        wp = enhanced_wp_model(
            row["score_margin"], 
            row["sec_remaining"], 
            row["PERIOD"]
        )
        df.loc[idx, "wp_home"] = wp
    
    # Calculate WP changes
    df["wp_next"] = df["wp_home"].shift(-1)
    df["wp_change"] = df["wp_home"].diff().fillna(0)
    df["dwp_obs"] = df["wp_next"] - df["wp_home"]
    
    # Add momentum metrics
    df = calculate_momentum_metrics(df)
    
    return df, meta

def create_enhanced_wp_chart(df: pd.DataFrame, meta: dict):
    """Create an enhanced win probability chart."""
    chart_data = df[df["wp_home"].notna()].copy()
    
    fig = go.Figure()
    
    # Main WP line
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["wp_home"] * 100,
        mode='lines',
        name='Home Win %',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Event %{x}<br>Home Win Prob: %{y:.1f}%<extra></extra>'
    ))
    
    # Highlight significant momentum shifts
    momentum_shifts = chart_data[chart_data["momentum_shift"] == 1]
    if not momentum_shifts.empty:
        fig.add_trace(go.Scatter(
            x=momentum_shifts.index,
            y=momentum_shifts["wp_home"] * 100,
            mode='markers',
            name='Momentum Shifts',
            marker=dict(
                color='red',
                size=8,
                symbol='diamond'
            ),
            hovertemplate='Momentum Shift<br>Event %{x}<br>WP: %{y:.1f}%<extra></extra>'
        ))
    
    # Highlight fouls
    fouls = chart_data[chart_data["is_foul"]]
    if not fouls.empty:
        colors = {'TECHNICAL': 'red', 'FLAGRANT': 'darkred', 'SHOOTING': 'orange', 'OFFENSIVE': 'purple'}
        for foul_type in fouls['foul_type'].unique():
            foul_subset = fouls[fouls['foul_type'] == foul_type]
            fig.add_trace(go.Scatter(
                x=foul_subset.index,
                y=foul_subset["wp_home"] * 100,
                mode='markers',
                name=f'{foul_type} Fouls',
                marker=dict(
                    color=colors.get(foul_type, 'gray'),
                    size=6,
                    symbol='x'
                ),
                hovertemplate=f'{foul_type} Foul<br>Event %{{x}}<br>WP: %{{y:.1f}}%<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"Win Probability Timeline - {meta.get('away_tricode', 'Away')} @ {meta.get('home_tricode', 'Home')}",
        xaxis_title="Game Events",
        yaxis_title="Home Team Win Probability (%)",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxis(range=[0, 100])
    
    return fig

def analyze_officiating_impact(df: pd.DataFrame) -> dict:
    """Comprehensive officiating impact analysis."""
    fouls = df[df["is_foul"] & df["wp_home"].notna()].copy()
    
    if fouls.empty:
        return {"error": "No foul data available for analysis"}
    
    # Basic metrics
    total_fouls = len(fouls)
    fouls_toward_home = (fouls["dwp_obs"] > 0).sum()
    fouls_toward_away = (fouls["dwp_obs"] < 0).sum()
    neutral_fouls = (fouls["dwp_obs"] == 0).sum()
    
    pct_toward_home = (fouls_toward_home / total_fouls) * 100
    
    # Impact analysis
    total_wp_impact = fouls["dwp_obs"].sum()
    avg_wp_impact = fouls["dwp_obs"].mean()
    
    # Timing analysis
    clutch_fouls = fouls[fouls["sec_remaining"] <= 300]  # Final 5 minutes
    clutch_impact = clutch_fouls["dwp_obs"].sum() if not clutch_fouls.empty else 0
    
    # Severity analysis
    severity_impact = {}
    for foul_type in fouls["foul_type"].unique():
        type_fouls = fouls[fouls["foul_type"] == foul_type]
        severity_impact[foul_type] = {
            "count": len(type_fouls),
            "avg_impact": type_fouls["dwp_obs"].mean(),
            "total_impact": type_fouls["dwp_obs"].sum()
        }
    
    # Statistical significance test (basic)
    from scipy import stats
    if len(fouls) > 10:  # Minimum sample size
        t_stat, p_value = stats.ttest_1samp(fouls["dwp_obs"], 0)
        statistically_significant = p_value < 0.05
    else:
        statistically_significant = False
        p_value = None
    
    return {
        "total_fouls": total_fouls,
        "fouls_toward_home": fouls_toward_home,
        "fouls_toward_away": fouls_toward_away,
        "pct_toward_home": pct_toward_home,
        "total_wp_impact": total_wp_impact,
        "avg_wp_impact": avg_wp_impact,
        "clutch_fouls": len(clutch_fouls),
        "clutch_impact": clutch_impact,
        "severity_impact": severity_impact,
        "statistically_significant": statistically_significant,
        "p_value": p_value
    }

def generate_game_insights(df: pd.DataFrame, meta: dict, officiating_analysis: dict) -> list:
    """Generate actionable insights from the game data."""
    insights = []
    
    # Game flow insights
    final_wp = df["wp_home"].iloc[-1] if not df["wp_home"].isna().all() else 0.5
    max_wp = df["wp_home"].max() if not df["wp_home"].isna().all() else 0.5
    min_wp = df["wp_home"].min() if not df["wp_home"].isna().all() else 0.5
    
    if max_wp - min_wp > 0.6:  # 60% swing
        insights.append({
            "type": "info",
            "title": "High Volatility Game",
            "message": f"This game featured significant momentum swings with win probability ranging from {min_wp*100:.1f}% to {max_wp*100:.1f}% for the home team."
        })
    
    # Close game analysis
    close_game_time = (df["wp_home"] > 0.4) & (df["wp_home"] < 0.6)
    close_game_duration = close_game_time.sum()
    
    if close_game_duration > len(df) * 0.3:
        insights.append({
            "type": "success",
            "title": "Competitive Game",
            "message": f"The game remained competitive for {close_game_duration} events ({close_game_duration/len(df)*100:.1f}% of the game)."
        })
    
    # Officiating insights
    if "error" not in officiating_analysis:
        pct_home = officiating_analysis["pct_toward_home"]
        total_impact = officiating_analysis["total_wp_impact"]
        
        if abs(pct_home - 50) > 15:  # More than 15% deviation from 50/50
            direction = "home" if pct_home > 50 else "away"
            insights.append({
                "type": "warning",
                "title": "Officiating Imbalance Detected",
                "message": f"{pct_home:.1f}% of foul calls favored the {direction} team, with a total impact of {total_impact:+.3f} win probability points."
            })
        
        if officiating_analysis["clutch_impact"] != 0:
            insights.append({
                "type": "info",
                "title": "Clutch Time Officiating Impact",
                "message": f"Foul calls in the final 5 minutes had a combined impact of {officiating_analysis['clutch_impact']:+.3f} win probability points."
            })
    
    return insights

# ---- Main Application Logic
if btn_fetch:
    with st.spinner("🔍 Fetching and analyzing game data..."):
        try:
            df, meta = load_enhanced_game_data(game_id)
        except Exception as e:
            st.error(f"❌ Failed to load game data: {str(e)}")
            st.stop()
    
    # Game Header
    home_team = f"{meta.get('home_city', '')} {meta.get('home_name', '')} ({meta.get('home_tricode', '')})"
    away_team = f"{meta.get('away_city', '')} {meta.get('away_name', '')} ({meta.get('away_tricode', '')})"
    game_date = meta.get('game_date', 'Unknown Date')
    
    final_home_score = df['home_score'].iloc[-1] if not df['home_score'].isna().all() else 0
    final_away_score = df['away_score'].iloc[-1] if not df['away_score'].isna().all() else 0
    
    st.markdown(f"""
    <div class="game-header">
        <h2>🏀 {away_team} @ {home_team}</h2>
        <h3>📅 {game_date} | Final Score: {final_away_score} - {final_home_score}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Win Probability", "⚡ Game Analytics", "🔍 Officiating Analysis", "💡 Insights"])
    
    with tab1:
        st.subheader("Win Probability Analysis")
        
        # Enhanced WP Chart
        wp_chart = create_enhanced_wp_chart(df, meta)
        st.plotly_chart(wp_chart, use_container_width=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            final_wp = df["wp_home"].iloc[-1] if not df["wp_home"].isna().all() else 0.5
            st.metric("Final Home Win Prob", f"{final_wp*100:.1f}%")
        
        with col2:
            max_wp = df["wp_home"].max() if not df["wp_home"].isna().all() else 0.5
            st.metric("Peak Home Win Prob", f"{max_wp*100:.1f}%")
        
        with col3:
            min_wp = df["wp_home"].min() if not df["wp_home"].isna().all() else 0.5
            st.metric("Lowest Home Win Prob", f"{min_wp*100:.1f}%")
        
        with col4:
            volatility = df["wp_volatility"].mean() if "wp_volatility" in df.columns else 0
            st.metric("Average Volatility", f"{volatility:.3f}")
    
    with tab2:
        if show_advanced_metrics or show_momentum_analysis:
            st.subheader("Advanced Game Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if show_momentum_analysis:
                    st.write("**Momentum Analysis**")
                    momentum_shifts = (df["momentum_shift"] == 1).sum()
                    st.metric("Major Momentum Shifts", momentum_shifts)
                    
                    # Score margin over time
                    fig_margin = go.Figure()
                    fig_margin.add_trace(go.Scatter(
                        x=df.index,
                        y=df["score_margin"],
                        mode='lines',
                        name='Score Margin',
                        line=dict(color='green', width=2)
                    ))
                    fig_margin.update_layout(
                        title="Score Margin Over Time (Positive = Home Leading)",
                        xaxis_title="Game Events",
                        yaxis_title="Score Margin",
                        height=400
                    )
                    st.plotly_chart(fig_margin, use_container_width=True)
            
            with col2:
                if show_advanced_metrics:
                    st.write("**Game Flow Metrics**")
                    
                    # Lead changes
                    lead_changes = ((df["score_margin"] > 0) != (df["score_margin"].shift(1) > 0)).sum()
                    st.metric("Lead Changes", lead_changes)
                    
                    # Largest lead
                    max_home_lead = df["score_margin"].max()
                    max_away_lead = abs(df["score_margin"].min())
                    st.metric("Largest Home Lead", f"+{max_home_lead}")
                    st.metric("Largest Away Lead", f"+{max_away_lead}")
    
    with tab3:
        if show_officiating_insights:
            st.subheader("Officiating Impact Analysis")
            
            officiating_analysis = analyze_officiating_impact(df)
            
            if "error" not in officiating_analysis:
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Fouls", officiating_analysis["total_fouls"])
                
                with col2:
                    st.metric("% Favoring Home", f"{officiating_analysis['pct_toward_home']:.1f}%")
                
                with col3:
                    st.metric("Total WP Impact", f"{officiating_analysis['total_wp_impact']:+.3f}")
                
                with col4:
                    st.metric("Clutch Time Fouls", officiating_analysis["clutch_fouls"])
                
                # Foul breakdown
                st.write("**Foul Impact by Type**")
                
                severity_data = []
                for foul_type, data in officiating_analysis["severity_impact"].items():
                    severity_data.append({
                        "Foul Type": foul_type,
                        "Count": data["count"],
                        "Avg Impact": f"{data['avg_impact']:+.3f}",
                        "Total Impact": f"{data['total_impact']:+.3f}"
                    })
                
                if severity_data:
                    st.dataframe(pd.DataFrame(severity_data), use_container_width=True, hide_index=True)
                
                # Statistical significance
                if officiating_analysis.get("statistically_significant"):
                    st.warning(f"⚠️ Officiating bias may be statistically significant (p={officiating_analysis['p_value']:.3f})")
                else:
                    st.info("ℹ️ No statistically significant officiating bias detected.")
            
            else:
                st.warning("No officiating data available for analysis.")
    
    with tab4:
        st.subheader("Game Insights & Analysis")
        
        insights = generate_game_insights(df, meta, analyze_officiating_impact(df))
        
        for insight in insights:
            if insight["type"] == "info":
                st.markdown(f"""
                <div class="insight-box">
                    <strong>{insight["title"]}</strong><br>
                    {insight["message"]}
                </div>
                """, unsafe_allow_html=True)
            elif insight["type"] == "warning":
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ {insight["title"]}</strong><br>
                    {insight["message"]}
                </div>
                """, unsafe_allow_html=True)
            elif insight["type"] == "success":
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ {insight["title"]}</strong><br>
                    {insight["message"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Additional detailed analytics
        st.write("**Detailed Game Statistics**")
        
        # Game summary stats
        total_events = len(df)
        total_fouls = (df["is_foul"] == True).sum()
        foul_rate = (total_fouls / total_events) * 100
        
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric("Total Events", total_events)
        with summary_cols[1]:
            st.metric("Total Fouls", total_fouls)
        with summary_cols[2]:
            st.metric("Foul Rate", f"{foul_rate:.1f}%")

else:
    # Landing page
    st.markdown("""
    ## 🚀 Welcome to RefLens Pro
    
    **Advanced NBA Game Analytics Platform**
    
    ### What You'll Get:
    
    **📊 Comprehensive Win Probability Analysis**
    - Enhanced WP model accounting for game situation
    - Momentum shift detection
    - Visual timeline with key events highlighted
    
    **⚡ Advanced Game Analytics**
    - Score margin analysis
    - Lead change tracking
    - Game flow volatility metrics
    - Momentum analysis
    
    **🔍 Officiating Impact Assessment**  
    - Foul impact quantification
    - Statistical significance testing
    - Clutch time officiating analysis
    - Bias detection algorithms
    
    **💡 Actionable Insights**
    - AI-generated game insights
    - Performance patterns identification
    - Strategic recommendations
    
    ### How to Use:
    1. Enter an NBA Game ID in the sidebar (format: 00223xxxxx for 2023-24 season)
    2. Click "Analyze Game" to fetch data
    3. Explore different analysis tabs
    4. Customize analysis options in sidebar
    
    ### Sample Game IDs to Try:
    - **0022300001** - Season opener 2023-24
    - **0022300500** - Mid-season game with interesting officiating
    - **0042300101** - Playoff game (if available)
    
    ---
    
    **Built with advanced statistical models and real NBA data from official sources.*
    """)
    
    # Additional features section
    with st.expander("🔧 Advanced Features", expanded=False):
        st.markdown("""
        ### Statistical Models Used:
        
        **Enhanced Win Probability Model:**
        - Accounts for game situation (regular vs overtime)
        - Increased volatility in final minutes
        - Historical NBA data calibration
        
        **Momentum Detection:**
        - Rolling averages for trend identification  
        - Volatility-based momentum shift detection
        - Statistical significance testing
        
        **Officiating Impact Analysis:**
        - Foul severity scoring system
        - Timing-based impact weighting
        - Bias detection with confidence intervals
        
        ### Data Sources:
        - NBA Official Play-by-Play Data
        - Real-time game statistics
        - Historical performance baselines
        
        ### Technical Notes:
        - All analysis is objective and data-driven
        - Win probability bounded between 0.1% and 99.9%
        - Overtime periods handled with enhanced coefficients
        - Statistical significance testing for bias detection
        """)
    
    with st.expander("📖 Understanding the Metrics", expanded=False):
        st.markdown("""
        ### Win Probability Metrics:
        
        **Win Probability (WP):** Likelihood of home team winning based on current game state
        
        **ΔWP (Delta Win Probability):** Change in win probability from one event to the next
        
        **Momentum Shift:** Significant WP change (>5%) indicating game momentum change
        
        **Volatility:** Standard deviation of WP over rolling window - higher = more unpredictable
        
        ### Officiating Metrics:
        
        **% Favoring Home/Away:** Percentage of foul calls that increased respective team's win probability
        
        **Total WP Impact:** Sum of all ΔWP from foul calls (positive = favors home)
        
        **Clutch Time Impact:** WP impact of fouls in final 5 minutes (most critical)
        
        **Statistical Significance:** Whether officiating bias exceeds random chance (p < 0.05)
        
        ### Foul Severity Scoring:
        - **Personal Foul:** 1.0 (baseline)
        - **Loose Ball:** 1.3 
        - **Offensive:** 1.5
        - **Shooting:** 2.0
        - **Technical:** 2.5
        - **Flagrant:** 3.0 (most severe)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    <p>RefLens Pro v2.0 | Advanced NBA Analytics Platform</p>
    <p>Data sourced from official NBA API | Analysis updated in real-time</p>
    <p>🏀 Built for basketball analytics enthusiasts and professionals</p>
</div>
""", unsafe_allow_html=True)
