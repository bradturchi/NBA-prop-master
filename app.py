import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import requests
import re  # Correctly placed import
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from fake_useragent import UserAgent

# --- CONFIGURATION ---
st.set_page_config(page_title="NBA Prop Master Mobile", page_icon="📱", layout="wide")

# --- API & IMPORTS ---
try:
    from nba_api.stats.static import players, teams
    from nba_api.stats.endpoints import playergamelog
    API_STATUS = "Online"
except ImportError:
    API_STATUS = "Offline"

# --- 1. ROBUST DATA FETCHERS ---

@st.cache_data(ttl=3600)
def fetch_team_defense_stats():
    """
    Scrapes 'Opponent Points Per Game' from TeamRankings.com.
    This is more reliable than B-Ref standings for simple PPG data.
    """
    url = "https://www.teamrankings.com/nba/stat/opponent-points-per-game"
    try:
        # TeamRankings tables are clean and simple
        dfs = pd.read_html(url)
        df = dfs[0]
        def_map = {}
        
        for _, row in df.iterrows():
            # Team names are like "Boston" or "Okla City"
            team_name = str(row['Team']).lower()
            
            # The column '2025' holds the current season stats (TeamRankings year naming convention)
            # We look for the numeric column that represents the current season
            # Usually the 3rd column (index 2) is the current season average
            try:
                ppg = float(row.iloc[2]) # Grab the current season value
                
                # Clean mappings
                if "okla" in team_name: team_name = "oklahoma city thunder"
                elif "la clippers" in team_name: team_name = "los angeles clippers"
                elif "la lakers" in team_name: team_name = "los angeles lakers"
                
                def_map[team_name] = ppg
                
                # Also map common short names
                if "boston" in team_name: def_map["celtics"] = ppg
                if "golden" in team_name: def_map["warriors"] = ppg
            except:
                continue
                
        return def_map
    except Exception as e:
        # Fallback to B-Ref if TeamRankings fails
        return fetch_defense_fallback()

def fetch_defense_fallback():
    """Backup scraper for B-Ref."""
    url = "https://www.basketball-reference.com/leagues/NBA_2026.html"
    try:
        dfs = pd.read_html(url)
        def_map = {}
        for df in dfs:
            if 'PA/G' in df.columns:
                for _, row in df.iterrows():
                    raw_team = str(row[df.columns[0]])
                    if "Division" in raw_team: continue
                    clean_name = re.sub(r'\s*\(\d+\)', '', raw_team).replace("*", "").strip().lower()
                    try: def_map[clean_name] = float(row['PA/G'])
                    except: continue
        return def_map
    except: return {}

@st.cache_data(ttl=3600)
def fetch_active_player_stats():
    """
    Scrapes Player Per Game Stats.
    Targeting 2026 Season (Current) to fix Jaylen Brown's PPG.
    """
    url = "https://www.basketball-reference.com/leagues/NBA_2026_per_game.html"
    try:
        dfs = pd.read_html(url)
        df = dfs[0]
        df = df[df['Player'] != 'Player']
        player_map = {}
        
        for _, row in df.iterrows():
            name = row['Player'].replace("*", "").strip().lower()
            try:
                stats = {
                    'PTS': float(row['PTS']),
                    'REB': float(row['TRB']),
                    'AST': float(row['AST']),
                    'MP': float(row['MP']),
                    'Pos': row['Pos'],
                    'Team': row['Team']
                }
                player_map[name] = stats
            except: continue
        return player_map
    except: return {}

# --- 2. SCHEDULE ENGINE ---

ABBREV_MAP = {
    'CHO': 'CHA', 'PHO': 'PHX', 'BRK': 'BKN', 'NOP': 'NO', 'SAS': 'SA', 'UTA': 'UTAH', 'WAS': 'WSH'
}

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
    end_date = today + timedelta(days=7)
    date_str = f"{today.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}&limit=100"
    
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        def_map = fetch_team_defense_stats()
        
        if not def_map:
            st.sidebar.error("⚠️ Defense stats not loaded.")
        
        for event in data.get('events', []):
            comp = event['competitions'][0]
            for i, team_data in enumerate(comp['competitors']):
                t_abbrev = team_data['team'].get('abbreviation', '').upper()
                t_name = team_data['team'].get('displayName', '').upper()
                
                if (t_abbrev == espn_abbrev) or (nickname in t_name):
                    d_str = comp['date'].replace('Z', '')
                    try: date_obj = datetime.strptime(d_str, "%Y-%m-%dT%H:%M:%S")
                    except: date_obj = datetime.strptime(d_str, "%Y-%m-%dT%H:%M")

                    if date_obj.date() < datetime.now().date(): continue 

                    is_home = (team_data['homeAway'] == 'home')
                    opp_idx = 1 - i
                    opp_team_data = comp['competitors'][opp_idx]['team']
                    opp_name = opp_team_data.get('displayName', 'Unknown')
                    
                    # --- SMART LOOKUP ---
                    # Try 1: Full match
                    opp_ppg = def_map.get(opp_name.lower())
                    
                    # Try 2: Partial match (e.g. "oklahoma city" in "oklahoma city thunder")
                    if not opp_ppg:
                        for key in def_map:
                            if key in opp_name.lower() or opp_name.lower() in key:
                                opp_ppg = def_map[key]
                                break
                    
                    # Try 3: Fallback to 114.5
                    if not opp_ppg: opp_ppg = 114.5
                    
                    return {
                        'date': date_obj, 'is_home': 1 if is_home else 0,
                        'opp_name': opp_name, 'opp_ppg': opp_ppg 
                    }
    except: pass
    return None

# --- 3. LOGIC CLASS ---

@st.cache_resource
def load_models():
    return {k: RandomForestRegressor(n_estimators=100, random_state=42) for k in ['PTS', 'REB', 'AST']}

class NBAPredictorLogic:
    def __init__(self):
        self.models = load_models()
        self.player_stats_backup = fetch_active_player_stats()
    
    def train(self, player_name):
        p_key = player_name.lower().strip()
        
        if p_key in self.player_stats_backup:
            stats = self.player_stats_backup[p_key]
            current_state = {
                'player_name': player_name.title(),
                'team_abbrev': stats['Team'], 
                'last_date': datetime.now(), 
                'last_min': stats['MP'],
                'avgs': {'PTS': stats['PTS'], 'REB': stats['REB'], 'AST': stats['AST']}
            }
            return current_state, "Success"
        
        return None, "Player not found in active roster."

# --- 4. UI LAYOUT ---
st.title("📱 NBA Prop Master")
st.caption("Powered by Basketball-Reference & ESPN")

if 'data' not in st.session_state: st.session_state.data = None

player_name = st.text_input("Player Name:", placeholder="e.g. Jaylen Brown")

if st.button("Analyze", type="primary"):
    if not player_name: st.warning("Enter name."); st.stop()
    
    predictor = NBAPredictorLogic()
    with st.spinner("Scouting Data..."):
        state, msg = predictor.train(player_name)
    
    if not state: st.error(msg)
    else:
        with st.spinner(f"Checking Schedule..."):
            next_game = get_espn_schedule(state['team_abbrev'])
        
        if not next_game:
            st.error(f"No games found for {state['team_abbrev']} in next 7 days.")
        else:
            st.session_state.data = {'state': state, 'next_game': next_game}

if st.session_state.data:
    d = st.session_state.data
    state = d['state']
    next_game = d['next_game']

    game_date = next_game['date'].strftime("%a %b %d")
    opp_ppg = next_game['opp_ppg']
    
    if opp_ppg > 118: def_color = "green"
    elif opp_ppg < 110: def_color = "red"
    else: def_color = "gray"

    c1, c2, c3 = st.columns(3)
    c1.metric("Opponent", next_game['opp_name'], delta=game_date, delta_color="off")
    c2.metric("Opp PPG", f"{opp_ppg:.1f}", delta="Allowed", delta_color="off") 
    c3.metric("Location", "Home" if next_game['is_home'] else "Away")
    
    if def_color == "green": st.success(f"✅ High Scoring Matchup! {next_game['opp_name']} allows {opp_ppg} PPG.")
    elif def_color == "red": st.error(f"🛡️ Grinder Matchup. {next_game['opp_name']} allows only {opp_ppg} PPG.")

    st.divider()
    st.subheader("🎯 Projections")
    
    def_impact = (opp_ppg - 114.5) / 114.5
    
    for stat in ['PTS', 'REB', 'AST']:
        base = state['avgs'][stat]
        pred = base * (1 + (def_impact * 0.8))
        
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
