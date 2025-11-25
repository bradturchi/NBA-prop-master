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
    from nba_api.stats.endpoints import playergamelog, playercareerstats
    API_STATUS = "Online"
except ImportError:
    API_STATUS = "Offline"
    st.error("NBA API not found. Please run: pip install nba_api")

# --- STRONG HEADERS (To reduce blocking) ---
ua = UserAgent()
def get_headers():
    return {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'Connection': 'keep-alive',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true'
    }

# --- 1. ESPN FETCHERS (Bypasses NBA Blocks) ---

@st.cache_data(ttl=3600)
def fetch_espn_defense_map():
    """
    Fetches 'Points Allowed Per Game' (PPG) from ESPN.
    Reliable and Mobile-Friendly.
    """
    url = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        def_map = {}
        
        # Parse nested ESPN structure
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
                        # Short name map (e.g. "celtics")
                        def_map[team.get('shortDisplayName', '').lower()] = avg_pa
                        
        return def_map
    except: return {}

def get_espn_schedule(my_team_name):
    """
    Finds the next game using ESPN's public scoreboard.
    """
    # 1. Setup Dates (Today -> +7 Days)
    today = datetime.now()
    end_date = today + timedelta(days=7)
    date_str = f"{today.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}&limit=100"
    
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        
        # Get Defense Data
        def_map = fetch_espn_defense_map()
        
        my_short = my_team_name.split()[-1].lower() # "celtics"

        for event in data.get('events', []):
            short_name = event['name'].lower()
            
            # Simple match
            if my_short in short_name:
                comp = event['competitions'][0]
                date_obj = datetime.strptime(comp['date'], "%Y-%m-%dT%H:%M:%SZ")
                
                # Filter Past Games (ESPN sometimes returns yesterday if late)
                if date_obj.date() < datetime.now().date(): continue

                competitors = comp['competitors']
                home_comp = next(c for c in competitors if c['homeAway'] == 'home')
                away_comp = next(c for c in competitors if c['homeAway'] == 'away')
                
                is_home = (my_short in home_comp['team']['displayName'].lower())
                opp_comp = away_comp if is_home else home_comp
                opp_name = opp_comp['team']['displayName']
                
                # Get Opponent PPG Allowed (Default to 114.5 if missing)
                opp_ppg = def_map.get(opp_name.lower(), def_map.get(opp_name.split()[-1].lower(), 114.5))
                
                return {
                    'date': date_obj,
                    'is_home': 1 if is_home else 0,
                    'opp_name': opp_name,
                    'opp_ppg': opp_ppg 
                }
    except Exception as e:
        st.error(f"ESPN Connection Error: {e}")
        
    return None

# --- 2. LOGIC CLASS ---

@st.cache_resource
def load_models():
    return {k: RandomForestRegressor(n_estimators=100, random_state=42) for k in ['PTS', 'REB', 'AST']}

class NBAPredictorLogic:
    def __init__(self):
        self.models = load_models()
        self.safe_mode = False
    
    def train(self, player_name):
        if API_STATUS == "Offline": return None, None
        
        # 1. ID Lookup
        p_key = player_name.lower()
        nba_players = players.get_players()
        nba_p = next((p for p in nba_players if p['full_name'].lower() == p_key), None)
        
        if not nba_p: return None, "Player not found."
        p_id = nba_p['id']
        
        # 2. Get Recent Logs (Try/Except for Blocking)
        logs = []
        for _ in range(2): # 2 Retries
            try:
                for season in ['2023-24', '2024-25']:
                    time.sleep(random.uniform(0.6, 1.2)) # Slow down to avoid blocks
                    gamelog = playergamelog.PlayerGameLog(player_id=p_id, season=season, headers=get_headers(), timeout=10)
                    logs.append(gamelog.get_data_frames()[0])
                break 
            except:
                time.sleep(1)
                continue

        # 3. SUCCESS PATH (Game Logs Found)
        if logs and not any(l.empty for l in logs):
            self.safe_mode = False
            df = pd.concat(logs, ignore_index=True)
            df.columns = [c.upper() for c in df.columns]
            
            # Identify Team Name for Schedule Lookup
            try: 
                # NBA API returns "BOS", we need to map to full name if possible, or just grab from user input context later
                # We will just fetch team name from the static players/teams list
                t_id = df['TEAM_ID'].iloc[0]
                t_info = teams.find_team_name_by_id(t_id)
                team_full_name = t_info['full_name']
            except: 
                team_full_name = "Unknown"

            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE').reset_index(drop=True)
            
            # Features
            df['days_rest'] = df['GAME_DATE'].diff().dt.days - 1
            df['days_rest'] = df['days_rest'].fillna(3).clip(0, 7)
            df['is_home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
            
            if 'MIN' in df.columns:
                df['MIN'] = df['MIN'].astype(str).apply(lambda x: float(x.split(':')[0]) if ':' in x else float(x) if x else 0.0)
            else: df['MIN'] = 32.0
            
            df['prev_game_min'] = df['MIN'].shift(1).fillna(32)
            
            # Rolling Averages
            current_avgs = {}
            for stat in ['PTS', 'REB', 'AST']:
                df[f'season_avg_{stat}'] = df[stat].expanding().mean().shift(1).fillna(df[stat].mean())
                current_avgs[stat] = df[stat].tail(10).mean()

            # Train Models
            clean_df = df.dropna()
            # We use a placeholder for opp_rating in training since historical ESPN data isn't easily mapped
            clean_df['opponent_rating'] = 114.5 
            
            for stat in ['PTS', 'REB', 'AST']:
                features = ['days_rest', 'is_home', 'prev_game_min', f'season_avg_{stat}', 'opponent_rating']
                if not clean_df.empty:
                    self.models[stat].fit(clean_df[features], clean_df[stat])

            current_state = {
                'player_name': nba_p['full_name'],
                'team_full_name': team_full_name,
                'last_date': df['GAME_DATE'].iloc[-1], 
                'last_min': df['MIN'].iloc[-1],
                'avgs': current_avgs
            }
            return current_state, "Success"

        # 4. SAFE MODE (Fallback if Player Stats are blocked)
        else:
            self.safe_mode = True
            try:
                career = playercareerstats.PlayerCareerStats(player_id=p_id, headers=get_headers())
                df = career.get_data_frames()[0].iloc[-1]
                
                gp = df['GP'] if df['GP'] > 0 else 1
                avgs = {'PTS': df['PTS']/gp, 'REB': df['REB']/gp, 'AST': df['AST']/gp}
                
                # Try to guess team
                t_id = df['TEAM_ID']
                t_info = teams.find_team_name_by_id(t_id)
                team_full_name = t_info['full_name']
                
                current_state = {
                    'player_name': nba_p['full_name'],
                    'team_full_name': team_full_name,
                    'last_date': datetime.now(), 
                    'last_min': 32.0, 
                    'avgs': avgs
                }
                return current_state, "Safe Mode"
            except Exception as e:
                return None, f"Player Stats Blocked. ({str(e)})"

# --- 3. UI LAYOUT ---
st.title("📱 NBA Prop Master")
st.caption("Powered by ESPN Data Feed")

if 'data' not in st.session_state: st.session_state.data = None

player_name = st.text_input("Player Name:", placeholder="e.g. Jaylen Brown")

if st.button("Analyze", type="primary"):
    if not player_name: st.warning("Enter name."); st.stop()
    
    predictor = NBAPredictorLogic()
    
    with st.spinner("Fetching Player Stats..."):
        state, msg = predictor.train(player_name)
    
    if not state: st.error(msg)
    else:
        if msg == "Safe Mode": st.warning("⚠️ Using Season Averages (Live Logs Blocked)")
        
        with st.spinner("Checking ESPN Schedule..."):
            next_game = get_espn_schedule(state['team_full_name'])
        
        if not next_game:
            st.error(f"No games found for {state['team_full_name']} in next 7 days.")
        else:
            st.session_state.data = {
                'state': state, 'next_game': next_game,
                'predictor': predictor, 'safe_mode': predictor.safe_mode
            }

if st.session_state.data:
    d = st.session_state.data
    state = d['state']
    next_game = d['next_game']
    predictor = d['predictor']

    # Date/Rest Logic
    game_date = next_game['date'].strftime("%a %b %d")
    raw_rest = (next_game['date'] - state['last_date']).days - 1
    days_rest = 3 if raw_rest > 7 or raw_rest < 0 else raw_rest
    
    # DEFENSE LOGIC (Points Allowed)
    opp_ppg = next_game['opp_ppg']
    
    # Color Logic: High PPG Allowed = GREEN (Good for Over)
    if opp_ppg > 118: def_color = "green" # Easy matchup
    elif opp_ppg < 110: def_color = "red" # Hard matchup
    else: def_color = "gray"

    # --- TOP METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Opponent", next_game['opp_name'], delta=game_date, delta_color="off")
    c2.metric("Opp PPG", f"{opp_ppg:.1f}", delta="Allowed", delta_color="off") 
    # Visual Hack: Show if defense is soft or hard
    if def_color == "green": c2.success("Soft Def")
    elif def_color == "red": c2.error("Tough Def")
    
    c3.metric("Rest Days", f"{days_rest}", delta="Fresh" if days_rest > 1 else "Tired")
    c4.metric("Location", "Home" if next_game['is_home'] else "Away")

    st.divider()
    
    # --- PROJECTIONS ---
    st.subheader("🎯 Betting Recommendations")
    
    # Calculate Impact based on PPG
    # 114.5 is average. 
    # If Opp allows 120, that's +5.5 points better context (approx +4.8% boost)
    def_impact = (opp_ppg - 114.5) / 114.5 
    
    for stat in ['PTS', 'REB', 'AST']:
        if d['safe_mode']:
            # Simple Logic for Safe Mode
            base = state['avgs'][stat]
            # Defense Mod: If they allow 5% more points, expect 5% more stats
            pred = base * (1 + def_impact)
            if days_rest == 0: pred *= 0.95
        else:
            # Random Forest Logic
            # We treat Opp PPG as the rating
            inputs = [[days_rest, next_game['is_home'], state['last_min'], state['avgs'][stat], opp_ppg]]
            base = predictor.models[stat].predict(inputs)[0]
            pred = base # RF already learned from the rating
        
        with st.container():
            c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
            c1.markdown(f"### {stat}")
            c1.caption(f"Proj: {pred:.1f}")
            
            line = c2.number_input(f"Line", value=float(round(state['avgs'][stat])), step=0.5, key=f"line_{stat}")
            diff = pred - line
            
            if diff > 0: d_dir, color = "OVER", "green"
            else: d_dir, color = "UNDER", "red"
            
            abs_diff = abs(diff)
            # Star Rating System
            if abs_diff > (line * 0.12): stars = "⭐⭐⭐⭐⭐"
            elif abs_diff > (line * 0.08): stars = "⭐⭐⭐"
            elif abs_diff > (line * 0.04): stars = "⭐"
            else: stars = "⚠️"

            c3.markdown(f"**Edge:** :{color}[{d_dir} {abs_diff:.1f}]")
            c4.markdown(f"**Rating:** {stars}")
        st.divider()
