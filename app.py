import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from fake_useragent import UserAgent

# --- CONFIGURATION ---
# No hardcoded defenders. We scout live.

st.set_page_config(page_title="NBA Prop Master Pro", page_icon="🧠", layout="wide")

try:
    from nba_api.stats.static import players, teams
    from nba_api.stats.endpoints import (
        playergamelog, scoreboardv2, leaguedashplayerstats, 
        commonplayerinfo, teamdashboardbygeneralsplits, leaguegamefinder, playercareerstats
    )
    API_STATUS = "Online"
except ImportError:
    API_STATUS = "Offline"
    st.error("NBA API not found. Check requirements.txt")

ua = UserAgent()
def get_headers():
    return {
        'User-Agent': ua.random,
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com/',
        'Accept-Language': 'en-US,en;q=0.9'
    }

# --- 1. FETCHERS ---
@st.cache_data(ttl=3600)
def fetch_specific_team_defense(team_id):
    """Fetches Defensive Rating for ONE specific team"""
    try:
        time.sleep(0.2)
        stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
            team_id=team_id, 
            measure_type_detailed_defense='Advanced',
            season='2024-25',
            headers=get_headers(),
            timeout=5
        )
        df = stats.get_data_frames()[0]
        return df['DEF_RATING'].iloc[0]
    except:
        return 114.5 

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

# --- 2. MODEL ---
@st.cache_resource
def load_models():
    return {k: RandomForestRegressor(n_estimators=100, random_state=42) for k in ['PTS', 'REB', 'AST']}

class NBAPredictorLogic:
    def __init__(self):
        self.models = load_models()
        self.safe_mode = False
    
    def train(self, player_name):
        if API_STATUS == "Offline": return None, None
        
        nba_players = players.get_players()
        player = next((p for p in nba_players if p['full_name'].lower() == player_name.lower()), None)
        if not player: return None, "Player not found."
        
        # --- STEP 1: GAME LOGS ---
        logs = []
        with st.spinner(f"Scouting {player_name}..."):
            for season in ['2023-24', '2024-25']:
                try:
                    time.sleep(random.uniform(0.2, 0.6))
                    gamelog = playergamelog.PlayerGameLog(player_id=player['id'], season=season, headers=get_headers(), timeout=10)
                    logs.append(gamelog.get_data_frames()[0])
                except: pass
        
        if logs:
            self.safe_mode = False
            df = pd.concat(logs, ignore_index=True)
            df.columns = [c.upper() for c in df.columns]
            
            try: t_id = df['TEAM_ID'].iloc[0]
            except: 
                try:
                    prof = commonplayerinfo.CommonPlayerInfo(player_id=player['id'], headers=get_headers())
                    t_id = prof.get_data_frames()[0]['TEAM_ID'].iloc[0]
                except: return None, "Could not identify team."
            
            # Auto-Position
            try:
                avg_ast = df['AST'].mean()
                avg_reb = df['REB'].mean()
                pos_list = []
                if avg_ast > 4.5: pos_list.append('G')
                if avg_reb > 6.0: pos_list.append('F')
                if avg_reb > 9.0: pos_list.append('C')
                if not pos_list: pos_list = ['G', 'F']
            except: pos_list = ['F']

            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE').reset_index(drop=True)
            df['days_rest'] = df['GAME_DATE'].diff().dt.days - 1
            df['days_rest'] = df['days_rest'].fillna(3).apply(lambda x: 3 if x > 7 or x < 0 else x)
            df['is_home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
            if 'MIN' in df.columns:
                df['MIN'] = df['MIN'].astype(str).apply(lambda x: float(x.split(':')[0]) if ':' in x else float(x))
            else: df['MIN'] = 32.0
            df['prev_game_min'] = df['MIN'].shift(1).fillna(32)
            
            for stat in ['PTS', 'REB', 'AST']:
                df[f'season_avg_{stat}'] = df[stat].expanding().mean().shift(1).fillna(df[stat].mean())

            current_state = {
                'player_id': player['id'], 'team_id': t_id, 
                'player_name': player['full_name'], 'pos_list': pos_list,
                'last_date': df['GAME_DATE'].iloc[-1], 'last_min': df['MIN'].iloc[-1],
                'avgs': {k: df[k].mean() for k in ['PTS', 'REB', 'AST']}
            }
            
            clean_df = df.dropna()
            for stat in ['PTS', 'REB', 'AST']:
                features = ['days_rest', 'is_home', 'prev_game_min', f'season_avg_{stat}', 'opponent_def_rating']
                clean_df['opponent_def_rating'] = 114.5
                self.models[stat].fit(clean_df[features], clean_df[stat])
            
            return current_state, "Success"

        else:
            # SAFE MODE
            self.safe_mode = True
            try:
                career = playercareerstats.PlayerCareerStats(player_id=player['id'], headers=get_headers())
                df = career.get_data_frames()[0]
                curr = df.iloc[-1]
                t_id = curr['TEAM_ID']
                gp = curr['GP'] if curr['GP'] > 0 else 1
                avgs = {'PTS': curr['PTS']/gp, 'REB': curr['REB']/gp, 'AST': curr['AST']/gp}
                
                current_state = {
                    'player_id': player['id'], 'team_id': t_id,
                    'player_name': player['full_name'], 'pos_list': ['F'],
                    'last_date': datetime.now(), 'last_min': 32.0, 'avgs': avgs
                }
                return current_state, "Safe Mode"
            except: return None, "API Fully Blocked."

    def get_next_game(self, team_id):
        today = datetime.now()
        for i in range(7):
            d_str = (today + timedelta(days=i)).strftime('%m/%d/%Y')
            try:
                board = scoreboardv2.ScoreboardV2(game_date=d_str, headers=get_headers(), timeout=5)
                games = board.get_data_frames()[0]
                games.columns = [c.upper() for c in games.columns]
                
                h_col, v_col = 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'
                if h_col in games.columns:
                    my_game = games[(games[h_col] == team_id) | (games[v_col] == team_id)]
                    if not my_game.empty:
                        g = my_game.iloc[0]
                        is_home = 1 if g[h_col] == team_id else 0
                        opp_id = g[v_col] if is_home else g[h_col]
                        try:
                            opp_info = teams.find_team_name_by_id(opp_id)
                            opp_name = opp_info['nickname']
                        except: opp_name = "Opponent"
                        
                        def_rtg = fetch_specific_team_defense(opp_id)
                        return {'date': today + timedelta(days=i), 'is_home': is_home, 'opp_id': opp_id, 'opp_name': opp_name, 'opp_def_rtg': def_rtg}
            except: continue
        return None

    def check_teammates(self, team_id, my_player_name, injury_report):
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(team_id_nullable=team_id, season='2024-25', headers=get_headers(), timeout=5)
            df = stats.get_data_frames()[0]
            top_scorers = df.sort_values('PTS', ascending=False)
            teammates = []
            for _, row in top_scorers.iterrows():
                if row['PLAYER_NAME'] != my_player_name:
                    teammates.append(row['PLAYER_NAME'])
                if len(teammates) >= 3: break
            for tm in teammates:
                status = check_player_status(injury_report, tm)
                if "out" in status or "injured" in status:
                    return True, tm, status
            return False, None, None
        except: return False, None, None

    # --- TRUE DYNAMIC SCOUTING (No Hardcoded List) ---
    def check_dynamic_defender(self, opp_team_id, injury_report):
        try:
            # 1. Fetch Opponent Roster Stats
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                team_id_nullable=opp_team_id, 
                measure_type_detailed_defense='Advanced',
                season='2024-25', 
                headers=get_headers(), 
                timeout=5
            )
            df = stats.get_data_frames()[0]
            
            # 2. Filter for Rotation Players (>24 MPG)
            rotation = df[df['MIN'] > 24.0].copy()
            
            # 3. Find Best Defender (Lowest DEF_RATING)
            # We sort by Def Rtg to find the toughest matchup
            best_defenders = rotation.sort_values('DEF_RATING')
            
            for _, row in best_defenders.iterrows():
                def_name = row['PLAYER_NAME']
                def_rtg = row['DEF_RATING']
                
                # 4. Check if Healthy
                status = check_player_status(injury_report, def_name)
                
                # If they are active AND elite (< 110.0), they are a threat
                if def_rtg < 110.0:
                    if "out" in status: 
                        return False, def_name, f"OUT (Rtg {def_rtg})"
                    else: 
                        return True, def_name, f"ACTIVE (Rtg {def_rtg})"
                        
            return False, None, None
        except: return False, None, None

# --- UI LAYOUT ---
st.title("⚡ NBA Prop Master Pro")
st.caption("Auto-Scouting Enabled")

if 'data' not in st.session_state: st.session_state.data = None

player_name = st.text_input("Player Name:", placeholder="e.g. LeBron James")

if st.button("Analyze", type="primary"):
    if not player_name: st.warning("Enter name."); st.stop()
    
    predictor = NBAPredictorLogic()
    state, msg = predictor.train(player_name)
    
    if not state: st.error(msg)
    else:
        if msg == "Safe Mode": st.warning("⚠️ API Busy: Using Safe Mode")
        
        next_game = predictor.get_next_game(state['team_id'])
        injury_report = fetch_live_injury_report()
        is_tm_out, tm_name, tm_stat = predictor.check_teammates(state['team_id'], state['player_name'], injury_report)
        
        is_def_active = False
        def_name, def_stat = None, None
        if next_game:
            # Use the Dynamic Scout
            is_def_active, def_name, def_stat = predictor.check_dynamic_defender(
                next_game['opp_id'], injury_report
            )
        
        st.session_state.data = {
            'state': state, 'next_game': next_game,
            'is_tm_out': is_tm_out, 'tm_name': tm_name, 'tm_stat': tm_stat,
            'is_def_active': is_def_active, 'def_name': def_name, 'def_stat': def_stat,
            'predictor': predictor,
            'safe_mode': predictor.safe_mode
        }

if st.session_state.data:
    d = st.session_state.data
    state = d['state']
    next_game = d['next_game']
    predictor = d['predictor']

    if not next_game: st.warning("No upcoming games."); st.stop()

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
    
    # Dynamic Label
    def_label = "Penalty: Elite Matchup?"
    if d['def_name']: def_label = f"Penalty: {d['def_name']}?"
        
    def_val = c2.checkbox(def_label, value=d['is_def_active'])
    if d['def_name']: 
        if d['is_def_active']: c2.error(f"🔒 {d['def_name']} ({d['def_stat']})")
        else: c2.success(f"🔓 {d['def_name']} is OUT")
    
    tm_mod = 1.15 if tm_val else 1.0
    def_mod = 0.92 if def_val else 1.0

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
