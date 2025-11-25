import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import requests
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
# --- 1. ROBUST DATA FETCHERS (UPDATED FOR 2025-26 SEASON) ---

@st.cache_data(ttl=3600)
import re # Add this import at the very top of your file if not there

@st.cache_data(ttl=3600)
def fetch_team_defense_stats():
    """
    Scrapes 'Points Allowed Per Game' (PA/G) from Basketball-Reference.
    UPDATED: Removes seeding numbers like '(1)' from team names.
    """
    url = "https://www.basketball-reference.com/leagues/NBA_2026.html"
    try:
        dfs = pd.read_html(url)
        def_map = {}
        
        for df in dfs:
            # Look for the Standings Table (it has PA/G)
            if 'PA/G' in df.columns:
                team_col = df.columns[0]
                
                for _, row in df.iterrows():
                    raw_team = str(row[team_col])
                    if "Division" in raw_team: continue
                    
                    # CLEANING: Remove "*" and "(1)" seed numbers
                    # "Boston Celtics (2)" -> "Boston Celtics"
                    clean_name = re.sub(r'\s*\(\d+\)', '', raw_team)
                    clean_name = clean_name.replace("*", "").strip().lower()
                    
                    try:
                        pag = float(row['PA/G'])
                        def_map[clean_name] = pag
                    except:
                        continue
                        
        return def_map
    except Exception as e:
        return {}


@st.cache_data(ttl=3600)
def fetch_active_player_stats():
    """
    Scrapes Player Per Game Stats.
    UPDATED: Now points to 2026 Season (Current).
    """
    # CHANGE: 2025 -> 2026
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
    # 1. Setup
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

    # 2. Fetch
    today = datetime.now()
    end_date = today + timedelta(days=7)
    date_str = f"{today.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}&limit=100"
    
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        
        # USE NEW DEFENSE SCRAPER
        def_map = fetch_team_defense_stats()
        
        # DEBUG SIDEBAR (To verify data is loaded)
        if not def_map:
            st.sidebar.error("❌ Stats Map Empty! B-Ref blocked?")
        else:
            st.sidebar.success(f"✅ Loaded Defense Stats for {len(def_map)} teams")
            # st.sidebar.write(def_map) # Uncomment to see all teams
        
        for event in data.get('events', []):
            comp = event['competitions'][0]
            for i, team_data in enumerate(comp['competitors']):
                t_abbrev = team_data['team'].get('abbreviation', '').upper()
                t_name = team_data['team'].get('displayName', '').upper()
                
                if (t_abbrev == espn_abbrev) or (nickname in t_name):
                    # Found Game
                    d_str = comp['date'].replace('Z', '')
                    try: date_obj = datetime.strptime(d_str, "%Y-%m-%dT%H:%M:%S")
                    except: date_obj = datetime.strptime(d_str, "%Y-%m-%dT%H:%M")

                    if date_obj.date() < datetime.now().date(): continue 

                    is_home = (team_data['homeAway'] == 'home')
                    opp_idx = 1 - i
                    opp_team_data = comp['competitors'][opp_idx]['team']
                    opp_name = opp_team_data.get('displayName', 'Unknown')
                    
                    # ROBUST LOOKUP
                    # Try full name first: "boston celtics"
                    opp_ppg = def_map.get(opp_name.lower())
                    
                    # Try last word: "celtics"
                    if not opp_ppg:
                        opp_ppg = def_map.get(opp_name.split()[-1].lower())
                        
                    # Try short name: "celtics"
                    if not opp_ppg:
                        short = opp_team_data.get('shortDisplayName', '').lower()
                        opp_ppg = def_map.get(short)

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
        
        # Direct Backup Use (Since we know API is flaky)
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
    
    # 114.5 is rough league average
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
