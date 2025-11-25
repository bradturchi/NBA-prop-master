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
# We keep these for local runs, but we won't rely on them failing
try:
    from nba_api.stats.static import players, teams
    from nba_api.stats.endpoints import playergamelog
    API_STATUS = "Online"
except ImportError:
    API_STATUS = "Offline"

# --- 1. ROBUST DATA FETCHERS (NON-NBA SOURCES) ---

@st.cache_data(ttl=3600)
def fetch_espn_defense_map():
    """Fetches 'Points Allowed Per Game' from ESPN (No Blocks)."""
    url = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        def_map = {}
        for conference in data.get('children', []):
            for division in conference.get('children', []):
                for team_entry in division.get('teams', []):
                    team = team_entry.get('team', {})
                    name = team.get('displayName', '').lower()
                    stats = team_entry.get('stats', [])
                    pa, gp = 0, 1
                    for s in stats:
                        if s['name'] == 'pointsAgainst': pa = float(s['value'])
                        if s['name'] == 'gamesPlayed': gp = float(s['value'])
                    if gp > 0:
                        avg_pa = pa / gp
                        def_map[name] = avg_pa
                        def_map[team.get('shortDisplayName', '').lower()] = avg_pa
        return def_map
    except: return {}

@st.cache_data(ttl=3600)
def fetch_active_player_stats():
    """
    Scrapes Basketball-Reference for 2025 Season Stats.
    This is our 'Indestructible' backup data source.
    """
    try:
        url = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html"
        # read_html returns a list of dfs, we want the first one
        dfs = pd.read_html(url)
        df = dfs[0]
        
        # Cleanup headers that repeat in the middle of the table
        df = df[df['Player'] != 'Player']
        
        player_map = {}
        
        for _, row in df.iterrows():
            # Clean name (remove "Hall of Fame" markers etc)
            name = row['Player'].replace("*", "").strip().lower()
            
            # Parse Stats safely
            try:
                stats = {
                    'PTS': float(row['PTS']),
                    'REB': float(row['TRB']), # BR uses 'TRB' for total rebounds
                    'AST': float(row['AST']),
                    'MP': float(row['MP']),
                    'Pos': row['Pos'],
                    'Team': row['Team']
                }
                player_map[name] = stats
            except:
                continue
                
        return player_map
    except Exception as e:
        print(f"Bball Ref Scraping Error: {e}")
        return {}

def get_espn_schedule(my_team_abbrev):
    """Finds next game using ESPN. Matches via Abbreviation (e.g. BOS)."""
    today = datetime.now()
    end_date = today + timedelta(days=7)
    date_str = f"{today.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}&limit=100"
    
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        def_map = fetch_espn_defense_map()
        
        # Normalize search key
        search_key = my_team_abbrev.lower() # e.g. "bos"

        for event in data.get('events', []):
            comp = event['competitions'][0]
            
            # Check competitors for our abbreviation
            home_team = comp['competitors'][0]['team']
            away_team = comp['competitors'][1]['team']
            
            h_abbrev = home_team.get('abbreviation', '').lower()
            a_abbrev = away_team.get('abbreviation', '').lower()
            
            if search_key == h_abbrev or search_key == a_abbrev:
                # Found Game!
                date_obj = datetime.strptime(comp['date'], "%Y-%m-%dT%H:%M:%SZ")
                if date_obj.date() < datetime.now().date(): continue # Skip finished games
                
                is_home = (search_key == h_abbrev)
                opp_team = away_team if is_home else home_team
                opp_name = opp_team.get('displayName', 'Unknown')
                
                # Def Lookup
                opp_ppg = def_map.get(opp_name.lower(), 114.5)
                
                return {
                    'date': date_obj,
                    'is_home': 1 if is_home else 0,
                    'opp_name': opp_name,
                    'opp_ppg': opp_ppg 
                }
    except: pass
    return None

# --- 2. LOGIC CLASS ---

@st.cache_resource
def load_models():
    # Only needed if full API works, but we keep structure
    return {k: RandomForestRegressor(n_estimators=100, random_state=42) for k in ['PTS', 'REB', 'AST']}

class NBAPredictorLogic:
    def __init__(self):
        self.models = load_models()
        self.player_stats_backup = fetch_active_player_stats()
        self.safe_mode = False
    
    def train(self, player_name):
        p_key = player_name.lower().strip()
        
        # --- ATTEMPT 1: FULL NBA API (Likely to fail on Cloud) ---
        try:
            # Short timeout to fail fast
            nba_players = players.get_players()
            nba_p = next((p for p in nba_players if p['full_name'].lower() == p_key), None)
            
            if nba_p:
                p_id = nba_p['id']
                # Try to fetch logs
                gamelog = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25', timeout=3)
                df = gamelog.get_data_frames()[0]
                
                # If we get here, API is working!
                if not df.empty:
                    self.safe_mode = False
                    # ... (Standard processing would go here) ...
                    # For simplicity in this robust mobile version, we will 
                    # drop to "Smart Safe Mode" to save code space and complexity,
                    # unless you explicitly need rolling logs.
                    # Let's trust the backup for mobile stability.
                    pass 
        except:
            # API Failed - Silently proceed to backup
            pass

        # --- ATTEMPT 2: BACKUP (Basketball Reference) ---
        # This is what runs on your phone
        if p_key in self.player_stats_backup:
            self.safe_mode = True
            stats = self.player_stats_backup[p_key]
            
            current_state = {
                'player_name': player_name.title(),
                'team_abbrev': stats['Team'], # e.g. "BOS"
                'last_date': datetime.now(), 
                'last_min': stats['MP'],
                'avgs': {'PTS': stats['PTS'], 'REB': stats['REB'], 'AST': stats['AST']}
            }
            return current_state, "Success"
        
        return None, "Player not found in active roster."

# --- 3. UI LAYOUT ---
st.title("📱 NBA Prop Master")
st.caption("Scragglys betting buddy")

if 'data' not in st.session_state: st.session_state.data = None

player_name = st.text_input("Player Name:", placeholder="e.g. Jaylen Brown")

if st.button("Analyze", type="primary"):
    if not player_name: st.warning("Enter name."); st.stop()
    
    predictor = NBAPredictorLogic()
    
    with st.spinner("Scouting Player Data..."):
        state, msg = predictor.train(player_name)
    
    if not state: st.error(msg)
    else:
        with st.spinner(f"Checking {state['team_abbrev']} Schedule..."):
            next_game = get_espn_schedule(state['team_abbrev'])
        
        if not next_game:
            st.error(f"No games found for {state['team_abbrev']} in next 7 days.")
        else:
            st.session_state.data = {
                'state': state, 'next_game': next_game,
                'predictor': predictor
            }

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
    c2.metric("Opp PPG", f"{opp_ppg:.1f}", delta="Defense", delta_color="off") 
    c3.metric("Location", "Home" if next_game['is_home'] else "Away")
    
    if def_color == "green": st.success(f"✅ Soft Defense! {next_game['opp_name']} allows {opp_ppg} PPG.")
    elif def_color == "red": st.error(f"🛡️ Tough Defense. {next_game['opp_name']} allows only {opp_ppg} PPG.")

    st.divider()
    
    st.subheader("🎯 Projections")
    
    # Simple, Robust Logic for Mobile
    def_impact = (opp_ppg - 114.5) / 114.5
    
    for stat in ['PTS', 'REB', 'AST']:
        base = state['avgs'][stat]
        pred = base * (1 + (def_impact * 0.8)) # 80% weight on defense
        
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
