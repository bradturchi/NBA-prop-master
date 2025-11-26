import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import requests
import re
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from fake_useragent import UserAgent

# --- CONFIGURATION ---
st.set_page_config(page_title="NBA Prop Master Mobile", page_icon="📱", layout="wide")

# --- 1. ROBUST DATA FETCHERS ---

@st.cache_data(ttl=3600)
def fetch_pace_stats():
    """Scrapes 'Possessions Per Game' (Pace) from TeamRankings."""
    url = "https://www.teamrankings.com/nba/stat/possessions-per-game"
    try:
        dfs = pd.read_html(url)
        df = dfs[0]
        pace_map = {}
        for _, row in df.iterrows():
            team_name = str(row['Team']).lower()
            if "okla" in team_name: team_name = "oklahoma city thunder"
            elif "la clippers" in team_name: team_name = "los angeles clippers"
            elif "la lakers" in team_name: team_name = "los angeles lakers"
            elif "golden" in team_name: team_name = "golden state warriors"
            elif "washington" in team_name: team_name = "washington wizards"
            try:
                pace = float(row.iloc[2]) # Current season is usually col 2
                pace_map[team_name] = pace
                if "boston" in team_name: pace_map["celtics"] = pace
                if "miami" in team_name: pace_map["heat"] = pace
            except: continue
        return pace_map
    except: return {}

@st.cache_data(ttl=3600)
def fetch_team_defense_stats():
    """Scrapes 'Opponent Points Per Game' from TeamRankings."""
    url = "https://www.teamrankings.com/nba/stat/opponent-points-per-game"
    try:
        dfs = pd.read_html(url)
        df = dfs[0]
        def_map = {}
        for _, row in df.iterrows():
            team_name = str(row['Team']).lower()
            if "okla" in team_name: team_name = "oklahoma city thunder"
            elif "la clippers" in team_name: team_name = "los angeles clippers"
            elif "la lakers" in team_name: team_name = "los angeles lakers"
            try:
                ppg = float(row.iloc[2])
                def_map[team_name] = ppg
                if "boston" in team_name: def_map["celtics"] = ppg
                if "golden" in team_name: def_map["warriors"] = ppg
            except: continue
        return def_map
    except: return fetch_defense_fallback()

def fetch_defense_fallback():
    """Backup scraper (B-Ref) if TeamRankings fails."""
    url = "https://www.basketball-reference.com/leagues/NBA_2026.html"
    try:
        dfs = pd.read_html(url)
        def_map = {}
        for df in dfs:
            if 'PA/G' in df.columns:
                for _, row in df.iterrows():
                    raw = str(row[df.columns[0]])
                    if "Division" in raw: continue
                    clean = re.sub(r'\s*\(\d+\)', '', raw).replace("*", "").strip().lower()
                    try: def_map[clean] = float(row['PA/G'])
                    except: continue
        return def_map
    except: return {}

@st.cache_data(ttl=3600)
def fetch_active_player_stats():
    """Scrapes 2026 Player Stats."""
    url = "https://www.basketball-reference.com/leagues/NBA_2026_per_game.html"
    try:
        dfs = pd.read_html(url)
        df = dfs[0]
        df = df[df['Player'] != 'Player']
        player_map = {}
        for _, row in df.iterrows():
            name = row['Player'].replace("*", "").strip().lower()
            try:
                player_map[name] = {
                    'PTS': float(row['PTS']), 'REB': float(row['TRB']),
                    'AST': float(row['AST']), 'MP': float(row['MP']),
                    'Pos': row['Pos'], 'Team': row['Team']
                }
            except: continue
        return player_map
    except: return {}

# --- 2. SCHEDULE ENGINE (UPDATED FOR B2B) ---
ABBREV_MAP = {'CHO': 'CHA', 'PHO': 'PHX', 'BRK': 'BKN', 'NOP': 'NO', 'SAS': 'SA', 'UTA': 'UTAH', 'WAS': 'WSH'}

def normalize_abbrev(abbrev):
    clean = abbrev.upper().strip()
    return ABBREV_MAP.get(clean, clean)

def get_espn_schedule(my_team_abbrev):
    raw_abbrev = my_team_abbrev.upper().strip()
    espn_abbrev = normalize_abbrev(raw_abbrev)
    
    NICKNAMES = {
        'BOS': 'CELTICS', 'LAL': 'LAKERS', 'LAC': 'CLIPPERS', 'PHI': '76ERS', 'MIA': 'HEAT',
        'MIL': 'BUCKS', 'CHI': 'BULLS', 'TOR': 'RAPTORS', 'NYK': 'KNICKS', 'BKN': 'NETS',
        'CLE': 'CAVALIERS', 'IND': 'PACERS', 'DET': 'PISTONS', 'ORL': 'MAGIC', 'ATL': 'HAWKS',
        'CHA': 'HORNETS', 'WAS': 'WIZARDS', 'MIN': 'TIMBERWOLVES', 'DEN': 'NUGGETS', 'OKC': 'THUNDER',
        'POR': 'TRAIL BLAZERS', 'UTA': 'JAZZ', 'GSW': 'WARRIORS', 'SAC': 'KINGS', 'PHX': 'SUNS',
        'SAS': 'SPURS', 'HOU': 'ROCKETS', 'DAL': 'MAVERICKS', 'MEM': 'GRIZZLIES', 'NOP': 'PELICANS'
    }
    nickname = NICKNAMES.get(espn_abbrev, "UNKNOWN")

    today = datetime.now()
    start_date = today - timedelta(days=1)
    end_date = today + timedelta(days=7)
    date_str = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}&limit=100"
    
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        
        def_map = fetch_team_defense_stats()
        pace_map = fetch_pace_stats()
        
        my_games = []
        for event in data.get('events', []):
            comp = event['competitions'][0]
            for i, team_data in enumerate(comp['competitors']):
                t_abbrev = team_data['team'].get('abbreviation', '').upper()
                t_name = team_data['team'].get('displayName', '').upper()
                
                if (t_abbrev == espn_abbrev) or (nickname in t_name):
                    d_str = comp['date'].replace('Z', '')
                    try: date_obj = datetime.strptime(d_str, "%Y-%m-%dT%H:%M:%S")
                    except: date_obj = datetime.strptime(d_str, "%Y-%m-%dT%H:%M")
                    
                    is_home = (team_data['homeAway'] == 'home')
                    opp_idx = 1 - i
                    opp_team = comp['competitors'][opp_idx]['team']
                    
                    my_games.append({
                        'date': date_obj,
                        'is_home': is_home,
                        'opp_name': opp_team.get('displayName', 'Unknown'),
                    })

        my_games.sort(key=lambda x: x['date'])
        
        next_game = None
        is_b2b = False
        
        for i, g in enumerate(my_games):
            if g['date'].date() >= today.date():
                next_game = g
                if i > 0:
                    prev_game = my_games[i-1]
                    if (g['date'].date() - prev_game['date'].date()).days == 1:
                        is_b2b = True
                break
        
        if next_game:
            opp_name = next_game['opp_name']
            def find_stat(map_obj, default_val):
                val = map_obj.get(opp_name.lower())
                if not val:
                        for key in map_obj:
                            if key in opp_name.lower(): return map_obj[key]
                return val if val else default_val

            opp_ppg = find_stat(def_map, 114.5)
            opp_pace = find_stat(pace_map, 100.0)
            
            return {
                'date': next_game['date'], 
                'is_home': 1 if next_game['is_home'] else 0,
                'opp_name': opp_name, 
                'opp_ppg': opp_ppg, 
                'opp_pace': opp_pace,
                'is_b2b': is_b2b
            }
            
    except: pass
    return None

# --- 3. LOGIC CLASS ---
class NBAPredictorLogic:
    def __init__(self):
        self.player_stats_backup = fetch_active_player_stats()
    
    def train(self, player_name):
        p_key = player_name.lower().strip()
        if p_key in self.player_stats_backup:
            stats = self.player_stats_backup[p_key]
            return {
                'player_name': player_name.title(),
                'team_abbrev': stats['Team'], 
                'last_date': datetime.now(), 
                'avgs': {'PTS': stats['PTS'], 'REB': stats['REB'], 'AST': stats['AST']}
            }, "Success"
        return None, "Player not found."

# --- 4. UI LAYOUT ---
st.title("NBA Player Performance Projections")

# --- NEW: HOW IT WORKS DROPDOWN ---
with st.expander("ℹ️ About"):
    st.markdown("""
    A cloud-native, mobile-first sports analytics tool designed to predict NBA player performance for prop betting.
    
    ### 📝 User Instructions
    1.  **Enter Player Name:** Type the full name (e.g., *Jaylen Brown*) and press **Enter** or **Analyze**.
    2.  **Review Factors:** Check the top dashboard for Opponent Strength, Pace, and Fatigue.
    3.  **Check Projections:** Scroll down to see the projected stats (PTS, REB, AST).
    4.  **Adjust the Line:** If your sportsbook has a different line (e.g., 24.5), type it into the "Line" box to see the new Edge rating.
    
    ### ⚙️ The "Quad-Factor" Engine
    This app modifies a player's season average based on four live factors:
    
    * **🛡️ Opponent Defense:** If the opponent allows *more* points than average (114.5), we boost the projection.
    * **⚡ Pace Factor:** If the opponent plays *faster* than average (100.0 possessions), we boost the projection (more opportunities).
    * **🏠 Home Court:** Players typically perform **3% better** at home.
    * **💤 Fatigue (Back-to-Back):** If the team played yesterday, we apply a **-5% penalty** for fatigue.
    """)

st.caption("Scraggyly's betting buddy")

if 'data' not in st.session_state: st.session_state.data = None

player_name = st.text_input("Player Name:", placeholder="e.g. Jaylen Brown")

if st.button("Analyze", type="primary"):
    if not player_name: st.warning("Enter name."); st.stop()
    predictor = NBAPredictorLogic()
    with st.spinner("Crunching Numbers..."):
        state, msg = predictor.train(player_name)
    
    if not state: st.error(msg)
    else:
        next_game = get_espn_schedule(state['team_abbrev'])
        if not next_game: st.error(f"No games found for {state['team_abbrev']}")
        else: st.session_state.data = {'state': state, 'next_game': next_game}

if st.session_state.data:
    d = st.session_state.data
    state = d['state']
    game = d['next_game']
    
    # FACTORS
    opp_ppg = game['opp_ppg']
    opp_pace = game['opp_pace']
    is_home = game['is_home']
    is_b2b = game['is_b2b']
    
    
    # --- VISUALS SECTION (UPDATED) ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Opponent", game['opp_name'], f"{'Home' if is_home else 'Away'}")
    
    # 1. DEFENSE FIX: Calculate difference from Avg (114.5)
    # If Opp PPG is 112.6, diff is -1.9 -> Red Down Arrow (Correct for betting)
    # If Opp PPG is 120.0, diff is +5.5 -> Green Up Arrow (Correct for betting)
    def_diff = opp_ppg - 114.5
    c2.metric("Defense (PPG)", f"{opp_ppg:.1f}", delta=f"{def_diff:.1f} vs Avg")
    
    # 2. PACE FIX: Calculate difference from Avg (100.0)
    # Slower than avg (-2.0) = Red/Bad. Faster (+2.0) = Green/Good.
    pace_diff = opp_pace - 100.0
    c3.metric("Pace", f"{opp_pace:.1f}", delta=f"{pace_diff:.1f} vs Avg")
    
    # 3. B2B LOGIC (Keep existing logic)
    if is_b2b:
        b2b_val = "Yes"
        b2b_delta = "Tired (-5%)"
        b2b_color = "inverse" # Red
    else:
        b2b_val = "No"
        b2b_delta = "Fresh"
        b2b_color = "normal"  # Green

    c4.metric("Back-to-Back?", b2b_val, delta=b2b_delta, delta_color=b2b_color)

    
    # --- PROJECTION MATH ---
    def_factor = (opp_ppg - 114.5) / 114.5 
    pace_factor = (opp_pace - 100.0) / 100.0 
    home_factor = 0.03 if is_home else 0.0
    b2b_factor = -0.05 if is_b2b else 0.0
    
    total_boost = def_factor + pace_factor + home_factor + b2b_factor
    
    st.subheader("📊 Factor Breakdown")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    b_col1.info(f"Defense: {def_factor*100:+.1f}%")
    b_col2.warning(f"Pace: {pace_factor*100:+.1f}%")
    b_col3.success(f"Home: {home_factor*100:+.1f}%")
    if is_b2b: b_col4.error(f"Fatigue: {b2b_factor*100:+.1f}%")
    else: b_col4.caption("Fatigue: 0%")
    
    st.divider()
    
    for stat in ['PTS', 'REB', 'AST']:
        base = state['avgs'][stat]
        pred = base * (1 + total_boost)
        
        with st.container():
            c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
            c1.markdown(f"### {stat}")
            c1.caption(f"Avg: {base:.1f}")
            line = c2.number_input(f"Line", value=float(round(base)), step=0.5, key=f"line_{stat}")
            
            diff = pred - line
            if diff > 0: d_dir, color = "OVER", "green"
            else: d_dir, color = "UNDER", "red"
            
            abs_diff = abs(diff)
            if abs_diff > (line * 0.12): stars = "⭐⭐⭐⭐⭐"
            elif abs_diff > (line * 0.08): stars = "⭐⭐⭐"
            elif abs_diff > (line * 0.04): stars = "⭐"
            else: stars = "⚠️"

            c3.markdown(f"**Proj:** {pred:.1f}")
            c4.markdown(f":{color}[**{d_dir}**] {stars}")
        st.divider()
