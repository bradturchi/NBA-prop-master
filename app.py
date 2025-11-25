import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from fake_useragent import UserAgent

# --- CONFIGURATION ---
st.set_page_config(page_title="NBA Prop Master Pro", page_icon="🚀", layout="wide")

# --- API & IMPORTS ---
try:
    from nba_api.stats.static import players, teams
    from nba_api.stats.endpoints import (
        playergamelog, leaguedashplayerstats, 
        teamdashboardbygeneralsplits, playercareerstats
    )
    API_STATUS = "Online"
except ImportError:
    API_STATUS = "Offline"
    st.error("NBA API not found. Please run: pip install nba_api")

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


# --- 1. LIVE INDEX ---
@st.cache_data(ttl=3600)
def build_player_index():
    try:
        url = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html"
        dfs = pd.read_html(url)
        df = dfs[0]
        df = df[df['Player'] != 'Player']
        player_map = {}
        
        for _, row in df.iterrows():
            name = row['Player'].replace("*", "").strip().lower()
            pos_raw = row['Pos']
            pos_list = []
            if "PG" in pos_raw or "SG" in pos_raw: pos_list.append("G")
            if "SF" in pos_raw or "PF" in pos_raw: pos_list.append("F")
            if "C" in pos_raw: pos_list.append("C")
            
            player_map[name] = {'pos': pos_list}
            
        return player_map
    except: return {}

# --- 2. FETCHERS ---
@st.cache_data(ttl=3600)
def fetch_specific_team_defense(team_id):
    try:
        time.sleep(random.uniform(0.2, 0.5))
        stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
            team_id=team_id, measure_type_detailed_defense='Advanced',
            season='2024-25', headers=get_headers(), timeout=5
        )
        return stats.get_data_frames()[0]['DEF_RATING'].iloc[0]
    except: return 114.5 

@st.cache_data(ttl=600)
def fetch_live_injury_report():
    try:
        dfs = pd.read_html("https://www.cbssports.com/nba/injuries/")
        combined = pd.concat(dfs, ignore_index=True)
        report = {}
        if 'Player' in combined.columns and 'Status' in combined.columns:
            for _, row in combined.iterrows():
                name = str(row['Player']).split(" • ")[0].strip().lower()
                report[name] = str(row['Status']).lower()
        return report
    except: return {}

def check_player_status(report, name):
    for key in report:
        if name.lower() in key: return report[key]
    return "Active"

# --- 3. SCHEDULE HELPERS (NEW) ---
@st.cache_data
def get_team_map():
    """Creates a dictionary mapping Team Names to NBA API IDs."""
    nba_teams = teams.get_teams()
    # Maps "Boston Celtics" -> 1610612738
    return {t['full_name'].lower(): t['id'] for t in nba_teams}

@st.cache_data
def load_schedule():
    """Loads the hardcoded schedule from CSV."""
    try:
        # Expects a file named 'nba_schedule_2025.csv' with columns: Date, Visitor, Home
        df = pd.read_csv("nba_schedule_2025.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- 4. MODELS & LOGIC ---
@st.cache_resource
def load_models():
    return {k: RandomForestRegressor(n_estimators=100, random_state=42) for k in ['PTS', 'REB', 'AST']}

class NBAPredictorLogic:
    def __init__(self):
        self.models = load_models()
        self.player_index = build_player_index()
        self.safe_mode = False
    
    def train(self, player_name):
        if API_STATUS == "Offline": return None, None
        
        # 1. ID Lookup
        p_key = player_name.lower()
        nba_players = players.get_players()
        nba_p = next((p for p in nba_players if p['full_name'].lower() == p_key), None)
        
        if not nba_p: return None, "Player not found."
        p_id = nba_p['id']
        
        # Get Position from index or fallback
        pos_list = self.player_index.get(p_key, {}).get('pos', ['F'])

        # 2. Get Recent Logs (Last 2 Seasons)
        logs = []
        for _ in range(3):
            try:
                for season in ['2023-24', '2024-25']:
                    time.sleep(random.uniform(0.5, 0.8)) 
                    gamelog = playergamelog.PlayerGameLog(player_id=p_id, season=season, headers=get_headers(), timeout=10)
                    logs.append(gamelog.get_data_frames()[0])
                break 
            except:
                time.sleep(1)
                continue

        # 3. SUCCESS PATH
        if logs and not any(l.empty for l in logs):
            self.safe_mode = False
            df = pd.concat(logs, ignore_index=True)
            df.columns = [c.upper() for c in df.columns]
            
            try: t_id = df['TEAM_ID'].iloc[0]
            except: t_id = 0

            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE').reset_index(drop=True)
            
            df['days_rest'] = df['GAME_DATE'].diff().dt.days - 1
            df['days_rest'] = df['days_rest'].fillna(3).clip(0, 7)
            df['is_home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
            
            if 'MIN' in df.columns:
                df['MIN'] = df['MIN'].astype(str).apply(lambda x: float(x.split(':')[0]) if ':' in x else float(x) if x else 0.0)
            else: df['MIN'] = 32.0
            
            df['prev_game_min'] = df['MIN'].shift(1).fillna(32)
            
            current_avgs = {}
            for stat in ['PTS', 'REB', 'AST']:
                df[f'season_avg_{stat}'] = df[stat].expanding().mean().shift(1).fillna(df[stat].mean())
                current_avgs[stat] = df[stat].tail(10).mean()

            clean_df = df.dropna()
            clean_df['opponent_def_rating'] = 114.5 
            
            for stat in ['PTS', 'REB', 'AST']:
                features = ['days_rest', 'is_home', 'prev_game_min', f'season_avg_{stat}', 'opponent_def_rating']
                if not clean_df.empty:
                    self.models[stat].fit(clean_df[features], clean_df[stat])

            current_state = {
                'player_id': p_id, 'team_id': t_id, 
                'player_name': nba_p['full_name'], 'pos_list': pos_list,
                'last_date': df['GAME_DATE'].iloc[-1], 
                'last_min': df['MIN'].iloc[-1],
                'avgs': current_avgs
            }
            return current_state, "Success"

        # 4. SAFE MODE
        else:
            self.safe_mode = True
            try:
                career = playercareerstats.PlayerCareerStats(player_id=p_id, headers=get_headers())
                df = career.get_data_frames()[0].iloc[-1]
                
                gp = df['GP'] if df['GP'] > 0 else 1
                avgs = {'PTS': df['PTS']/gp, 'REB': df['REB']/gp, 'AST': df['AST']/gp}
                t_id = df['TEAM_ID']
                
                current_state = {
                    'player_id': p_id, 'team_id': t_id,
                    'player_name': nba_p['full_name'], 'pos_list': pos_list,
                    'last_date': datetime.now(), 
                    'last_min': 32.0, 
                    'avgs': avgs
                }
                return current_state, "Safe Mode"
            except Exception as e:
                return None, f"All APIs Blocked. Try running locally. ({str(e)})"

    def get_next_game(self, team_id):
        """
        Scans the local CSV schedule (Robust).
        Then uses API only for the Opponent's defensive stats (Dynamic).
        """
        schedule = load_schedule()
        if schedule.empty:
            st.error("❌ Schedule file 'nba_schedule_2025.csv' not found.")
            return None

        # Get the team name for the ID we are searching for (e.g., 1610612738 -> "Boston Celtics")
        all_teams = teams.get_teams()
        my_team_info = next((t for t in all_teams if t['id'] == team_id), None)
        if not my_team_info: return None
        
        my_team_name = my_team_info['full_name'].lower()
        
        # Filter schedule for games in the next 7 days
        today = pd.Timestamp.now().normalize()
        end_date = today + pd.Timedelta(days=7)
        
        # Normalize strings for comparison
        schedule['Visitor'] = schedule['Visitor'].astype(str).str.lower()
        schedule['Home'] = schedule['Home'].astype(str).str.lower()
        
        # Find relevant games
        mask = (
            (schedule['Date'] >= today) & 
            (schedule['Date'] <= end_date) & 
            ((schedule['Visitor'] == my_team_name) | (schedule['Home'] == my_team_name))
        )
        
        upcoming_games = schedule[mask].sort_values('Date')
        
        if upcoming_games.empty: return None
        
        # Pick the first game
        game = upcoming_games.iloc[0]
        
        # Determine Opponent
        is_home = (game['Home'] == my_team_name)
        opp_name_str = game['Visitor'] if is_home else game['Home']
        
        # Map Opponent Name back to ID (Critical for Dynamic Defender logic)
        team_map = get_team_map()
        opp_id = team_map.get(opp_name_str, 0)
        
        # Fetch REAL TIME Opponent Defense (This keeps it dynamic!)
        def_rtg = fetch_specific_team_defense(opp_id)
        
        display_opp_name = opp_name_str.title()

        return {
            'date': game['Date'],
            'is_home': 1 if is_home else 0,
            'opp_id': opp_id,
            'opp_name': display_opp_name,
            'opp_def_rtg': def_rtg
        }

    def check_teammates(self, team_id, my_player_name, injury_report):
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(team_id_nullable=team_id, season='2024-25', headers=get_headers(), timeout=5)
            df = stats.get_data_frames()[0]
            top_scorers = df.sort_values('PTS', ascending=False).head(4)
            
            for _, row in top_scorers.iterrows():
                if row['PLAYER_NAME'].lower() != my_player_name.lower():
                    status = check_player_status(injury_report, row['PLAYER_NAME'])
                    if "out" in status or "injured" in status:
                        return True, row['PLAYER_NAME'], status
            return False, None, None
        except: return False, None, None

    def check_dynamic_defender(self, opp_team_id, my_positions, injury_report):
        if opp_team_id == 0: return False, None, None, 0.0
        try:
            # Get Opponent Roster Stats
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                team_id_nullable=opp_team_id, season='2024-25', 
                measure_type_detailed_defense='Advanced', headers=get_headers(), timeout=5
            )
            df = stats.get_data_frames()[0]
            
            # Filter: Regular rotation players (>24 mins)
            rotation_players = df[df['MIN'] > 20.0].sort_values('DEF_RATING', ascending=True)
            
            best_match = None
            
            for _, row in rotation_players.iterrows():
                def_name = row['PLAYER_NAME']
                def_rtg = row['DEF_RATING']
                
                status = check_player_status(injury_report, def_name)
                if "out" in status: continue
                
                def_key = def_name.lower()
                def_pos_list = self.player_index.get(def_key, {}).get('pos', ['F'])
                
                # INTERSECTION LOGIC
                common_pos = set(my_positions).intersection(def_pos_list)
                is_match = False
                match_type = ""
                
                if common_pos:
                    if len(common_pos) >= len(my_positions): 
                        match_type = "PRIMARY" 
                    else: 
                        match_type = "SWITCH" 
                    is_match = True
                
                if is_match and def_rtg < 112.0:
                    best_match = (def_name, match_type, def_rtg)
                    break
            
            if best_match:
                name, m_type, rtg = best_match
                # Calculate Penalty
                base_pen = 0.10 if rtg < 108.0 else 0.05
                if m_type == "SWITCH": base_pen *= 0.6
                return True, name, f"{m_type} ({rtg:.1f})", base_pen
                
            return False, None, None, 0.0
        except: return False, None, None, 0.0

# --- 5. UI LAYOUT ---
st.title("NBA Prop Master")
st.caption("Scragglys betting buddy")

if 'data' not in st.session_state: st.session_state.data = None

player_name = st.text_input("Player Name:", placeholder="e.g. Jaylen Brown")

if st.button("Analyze", type="primary"):
    if not player_name: st.warning("Enter name."); st.stop()
    
    predictor = NBAPredictorLogic()
    state, msg = predictor.train(player_name)
    
    if not state: st.error(msg)
    else:
        if msg == "Safe Mode": st.warning("⚠️ API Busy: Using Safe Mode (Base Stats Only)")
        
        with st.spinner("Checking Schedule..."):
            next_game = predictor.get_next_game(state['team_id'])
        
        if not next_game:
            st.error(f"No games scheduled for {state['player_name']} in the next 7 days.")
        else:
            injury_report = fetch_live_injury_report()
            
            is_tm_out, tm_name, tm_stat = predictor.check_teammates(state['team_id'], state['player_name'], injury_report)
            
            is_def_active = False
            def_name, def_type, penalty_amt = None, None, 0.0
            
            with st.spinner("Scouting Defense..."):
                is_def_active, def_name, def_type, penalty_amt = predictor.check_dynamic_defender(
                    next_game['opp_id'], state['pos_list'], injury_report
                )
            
            st.session_state.data = {
                'state': state, 'next_game': next_game,
                'is_tm_out': is_tm_out, 'tm_name': tm_name, 'tm_stat': tm_stat,
                'is_def_active': is_def_active, 'def_name': def_name, 
                'def_type': def_type, 'penalty_amt': penalty_amt,
                'predictor': predictor, 'safe_mode': predictor.safe_mode
            }

if st.session_state.data:
    d = st.session_state.data
    state = d['state']
    next_game = d['next_game']
    predictor = d['predictor']

    if not next_game: 
        st.warning("No upcoming games found."); st.stop()

    raw_rest = (next_game['date'] - state['last_date']).days - 1
    days_rest = 3 if raw_rest > 7 or raw_rest < 0 else raw_rest
    
    opp_rating = next_game['opp_def_rtg']
    if opp_rating < 110: def_color = "red"
    elif opp_rating > 118: def_color = "green"
    else: def_color = "gray"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Opponent", next_game['opp_name'])
    c2.metric("Def Rtg", f"{opp_rating:.1f}", delta=f"{114.5 - opp_rating:.1f}", delta_color="inverse")
    c3.metric("Rest", f"{days_rest}", delta="Fresh" if days_rest > 1 else "Tired")
    c4.metric("Pos", "/".join(state['pos_list']))

    st.divider()
    
    st.subheader("⚙️ Matchup Context")
    c1, c2 = st.columns(2)
    
    tm_val = c1.checkbox("Boost: Teammate Out?", value=d['is_tm_out'])
    if d['is_tm_out']: c1.caption(f"ℹ️ {d['tm_name']} is {d['tm_stat']}")
    
    def_label = "Penalty: Matchup?"
    if d['def_name']: def_label = f"Penalty: {d['def_name']}?"
    
    def_val = c2.checkbox(def_label, value=d['is_def_active'])
    if d['def_name']:
        if d['is_def_active']: c2.error(f"🔒 {d['def_name']} ({d['def_type']})")
        else: c2.success(f"🔓 {d['def_name']} is OUT")
    else: c2.caption("No elite positional defender found.")
    
    tm_mod = 1.15 if tm_val else 1.0
    def_mod = (1.0 - d['penalty_amt']) if def_val else 1.0

    st.divider()
    
    st.subheader("🎯 Betting Recommendations")
    for stat in ['PTS', 'REB', 'AST']:
        if d['safe_mode']:
            pred = state['avgs'][stat] * tm_mod * def_mod
            if days_rest == 0: pred *= 0.95
            if opp_rating < 110: pred *= 0.95
            if opp_rating > 118: pred *= 1.05
        else:
            inputs = [[days_rest, next_game['is_home'], state['last_min'], state['avgs'][stat], opp_rating]]
            base = predictor.models[stat].predict(inputs)[0]
            pred = base * tm_mod * def_mod
        
        with st.container():
            c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
            c1.markdown(f"### {stat}")
            c1.caption(f"Proj: {pred:.1f}")
            line = c2.number_input(f"{stat} Line", value=float(round(state['avgs'][stat])), step=0.5, key=f"line_{stat}")
            diff = pred - line
            
            if diff > 0: d_dir, color = "OVER", "green"
            else: d_dir, color = "UNDER", "red"
            
            abs_diff = abs(diff)
            if abs_diff > (line * 0.12): stars = "⭐⭐⭐⭐⭐"
            elif abs_diff > (line * 0.08): stars = "⭐⭐⭐"
            elif abs_diff > (line * 0.04): stars = "⭐"
            else: stars = "⚠️"

            c3.markdown(f"**Edge:** :{color}[{d_dir} {abs_diff:.1f}]")
            c4.markdown(f"**Rating:** {stars}")
        st.divider()
