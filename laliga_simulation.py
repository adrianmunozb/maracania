import pandas as pd
import numpy as np
from collections import defaultdict
import time
import json
import os

LEAGUES = {
    'La Liga': 'sofascore_la_liga',
    'Premier League': 'sofascore_premier_league',
    'Serie A': 'sofascore_serie_a',
    'Bundesliga': 'sofascore_bundesliga',
    'Ligue 1': 'sofascore_ligue_1'
}

def calculate_split_strength(row, league_avgs):
    """
    Calculates Home and Away Attack/Defense ratings using a weighted multi-stat model.
    Weights: 
    - Attack: 35% Goals, 35% Big Chances, 20% SoT, 10% Control (Possession/Corners)
    - Defense: 35% Goals Conc, 35% Big Chances Agst, 20% SoT Agst, 10% Suppression (Shots Agst)
    """
    
    # --- HELPER: Normalize ---
    def normalize(val, avg, min_val=0.1):
        return val / max(avg, min_val)

    # --- ATTACK METRICS ---
    # 1. Goals (Home/Away split available)
    total_goals = max(row['goalsScored'], 1)
    home_ratio = row['home_goles_favor'] / total_goals
    away_ratio = row['away_goles_favor'] / total_goals
    
    h_goals = row['home_goles_favor'] / max(row['home_partidos_jugados'], 1)
    a_goals = row['away_goles_favor'] / max(row['away_partidos_jugados'], 1)
    
    # 2. Big Chances (Total only -> Split by Goal Ratio)
    bc = row.get('bigChances', 0)
    h_bc = (bc * home_ratio) / max(row['home_partidos_jugados'], 1)
    a_bc = (bc * away_ratio) / max(row['away_partidos_jugados'], 1)
    
    # 3. Shots on Target (Total only -> Split by Goal Ratio)
    sot = row.get('shotsOnTarget', 0)
    h_sot = (sot * home_ratio) / max(row['home_partidos_jugados'], 1)
    a_sot = (sot * away_ratio) / max(row['away_partidos_jugados'], 1)
    
    # 4. Control: Possession & Corners (Total only -> Assume constant or slight home bias? Let's use simple distribution)
    # We'll use a 55/45 split assumption for possession/corners domination at home vs away if not specified, 
    # but strictly sticking to the data: we only have totals. 
    # Let's use the same goal ratio as a proxy for "playing better" at home.
    poss = row.get('averageBallPossession', 50.0)
    corn = row.get('corners', 0)
    
    # Possession isn't additive, it's an average. We assume they keep their average style home and away roughly, 
    # maybe slightly higher home. Let's just use the raw average for both to be safe, or slightly adjust.
    # Actually, teams dominate more at home. Let's start with raw avg to avoid noise.
    # Corners are additive.
    h_corn = (corn * home_ratio) / max(row['home_partidos_jugados'], 1)
    a_corn = (corn * away_ratio) / max(row['away_partidos_jugados'], 1)
    
    # Normalize Attack
    n_h_goals = normalize(h_goals, league_avgs['home_goals'])
    n_a_goals = normalize(a_goals, league_avgs['away_goals'])
    
    n_h_bc = normalize(h_bc, league_avgs['home_big_chances'])
    n_a_bc = normalize(a_bc, league_avgs['away_big_chances'])
    
    n_h_sot = normalize(h_sot, league_avgs['home_sot'])
    n_a_sot = normalize(a_sot, league_avgs['away_sot'])
    
    n_poss = normalize(poss, league_avgs['avg_possession'])
    n_h_corn = normalize(h_corn, league_avgs['home_corners'])
    n_a_corn = normalize(a_corn, league_avgs['away_corners'])
    
    # Composite Attack Ratings
    # Formula: 0.35 Goals + 0.35 BC + 0.20 SoT + 0.05 Poss + 0.05 Corners
    home_att_rating = (0.35 * n_h_goals) + (0.35 * n_h_bc) + (0.20 * n_h_sot) + (0.05 * n_poss) + (0.05 * n_h_corn)
    away_att_rating = (0.35 * n_a_goals) + (0.35 * n_a_bc) + (0.20 * n_a_sot) + (0.05 * n_poss) + (0.05 * n_a_corn)


    # --- DEFENSE METRICS ---
    # 1. Goals Conceded (Home/Away split available)
    total_conc = max(row['goalsConceded'], 1)
    # Home Defense Ratio: Goals conceded at HOME / Total Conceded
    h_conc_ratio = row['home_goles_contra'] / total_conc
    a_conc_ratio = row['away_goles_contra'] / total_conc
    
    h_conc = row['home_goles_contra'] / max(row['home_partidos_jugados'], 1)
    a_conc = row['away_goles_contra'] / max(row['away_partidos_jugados'], 1)
    
    # 2. Big Chances Against (Total -> Split by Conceded Ratio)
    bca = row.get('bigChancesAgainst', 0)
    h_bca = (bca * h_conc_ratio) / max(row['home_partidos_jugados'], 1)
    a_bca = (bca * a_conc_ratio) / max(row['away_partidos_jugados'], 1)
    
    # 3. SoT Against
    sota = row.get('shotsOnTargetAgainst', 0)
    h_sota = (sota * h_conc_ratio) / max(row['home_partidos_jugados'], 1)
    a_sota = (sota * a_conc_ratio) / max(row['away_partidos_jugados'], 1)
    
    # 4. Suppression: Total Shots Against
    sa = row.get('shotsAgainst', 0)
    h_sa = (sa * h_conc_ratio) / max(row['home_partidos_jugados'], 1)
    a_sa = (sa * a_conc_ratio) / max(row['away_partidos_jugados'], 1)
    
    # Normalize Defense (Lower is better usually, but here 'rating' is a multiplier for expected goals.
    # So High Rating = BAD Defense (concedes more).
    # We are calculating "Expected Conceded" multiplier.
    
    n_h_conc = normalize(h_conc, league_avgs['home_goals_conc'])
    n_a_conc = normalize(a_conc, league_avgs['away_goals_conc'])
    
    n_h_bca = normalize(h_bca, league_avgs['home_bca'])
    n_a_bca = normalize(a_bca, league_avgs['away_bca'])
    
    n_h_sota = normalize(h_sota, league_avgs['home_sota'])
    n_a_sota = normalize(a_sota, league_avgs['away_sota'])
    
    n_h_sa = normalize(h_sa, league_avgs['home_sa'])
    n_a_sa = normalize(a_sa, league_avgs['away_sa'])
    
    # Composite Defense Ratings
    # Formula: 0.35 Goals + 0.35 BC + 0.20 SoT + 0.10 Shots
    home_def_rating = (0.35 * n_h_conc) + (0.35 * n_h_bca) + (0.20 * n_h_sota) + (0.10 * n_h_sa)
    away_def_rating = (0.35 * n_a_conc) + (0.35 * n_a_bca) + (0.20 * n_a_sota) + (0.10 * n_a_sa)
    
    return home_att_rating, home_def_rating, away_att_rating, away_def_rating

def run_simulation(stats_df, fixtures_df, n_simulations=1000000):
    start_time = time.time()
    
    # 1. League Averages
    total_home_matches = stats_df['home_partidos_jugados'].mean()
    if pd.isna(total_home_matches) or total_home_matches == 0: total_home_matches = 1

    total_away_matches = stats_df['away_partidos_jugados'].mean()
    if pd.isna(total_away_matches) or total_away_matches == 0: total_away_matches = 1
    
    # Helper to safe mean per game
    def get_per_game(col):
        if col in stats_df.columns:
            return stats_df[col].sum() / (stats_df['partidos_jugados'].sum())
        return 1.0

    # Calculate global per-game averages for normalization
    avg_bc = get_per_game('bigChances')
    avg_sot = get_per_game('shotsOnTarget')
    avg_poss = stats_df['averageBallPossession'].mean() if 'averageBallPossession' in stats_df else 50.0
    avg_corn = get_per_game('corners')
    
    avg_bca = get_per_game('bigChancesAgainst')
    avg_sota = get_per_game('shotsOnTargetAgainst')
    avg_sa = get_per_game('shotsAgainst')

    # Base Goal Avgs
    league_home_goals = stats_df['home_goles_favor'].mean() / total_home_matches
    league_away_goals = stats_df['away_goles_favor'].mean() / total_away_matches
    league_home_conc = stats_df['home_goles_contra'].mean() / total_home_matches
    league_away_conc = stats_df['away_goles_contra'].mean() / total_away_matches
    
    league_avgs = {
        'home_goals': league_home_goals,
        'away_goals': league_away_goals,
        'home_goals_conc': league_home_conc,
        'away_goals_conc': league_away_conc,
        
        'home_big_chances': avg_bc * 1.1,
        'away_big_chances': avg_bc * 0.9,
        
        'home_sot': avg_sot * 1.1,
        'away_sot': avg_sot * 0.9,
        
        'avg_possession': avg_poss, # Not split
        
        'home_corners': avg_corn * 1.1,
        'away_corners': avg_corn * 0.9,
        
        'home_bca': avg_bca * 0.9, # Concede FEWER at home
        'away_bca': avg_bca * 1.1,
        
        'home_sota': avg_sota * 0.9,
        'away_sota': avg_sota * 1.1,
        
        'home_sa': avg_sa * 0.9,
        'away_sa': avg_sa * 1.1,
    }
    
    # 2. Build Profiles & Index Map
    team_profiles = {}
    team_indices = {} # map team_id -> 0..N
    reverse_indices = {} # map 0..N -> team_id
    
    idx = 0
    # Arrays for current state (vectorized)
    current_points_arr = []
    current_gd_arr = []
    
    for _, row in stats_df.iterrows():
        tid = row['id']
        h_att, h_def, a_att, a_def = calculate_split_strength(row, league_avgs)
        
        team_profiles[tid] = {
            'name': row['Equipo'],
            'home_attack': h_att,
            'home_defense': h_def,
            'away_attack': a_att,
            'away_defense': a_def,
            'current_points': row['puntos'],
            'current_gd': row['goles_favor'] - row['goles_contra'],
        }
        
        team_indices[tid] = idx
        reverse_indices[idx] = tid
        current_points_arr.append(row['puntos'])
        current_gd_arr.append(row['goles_favor'] - row['goles_contra'])
        idx += 1
        
    num_teams = len(team_profiles)
    current_points_vec = np.array(current_points_arr, dtype=np.int32)
    current_gd_vec = np.array(current_gd_arr, dtype=np.int32)

    # 3. Prepare Matches for Vectorization
    fixtures_df['local_id'] = fixtures_df['local_id'].astype(int)
    fixtures_df['visitante_id'] = fixtures_df['visitante_id'].astype(int)
    
    match_home_indices = []
    match_away_indices = []
    match_home_exp = []
    match_away_exp = []
    
    for _, match in fixtures_df.iterrows():
        lid = match['local_id']
        vid = match['visitante_id']
        
        if lid in team_profiles and vid in team_profiles:
            home_team = team_profiles[lid]
            away_team = team_profiles[vid]
            
            h_exp = league_avgs['home_goals'] * home_team['home_attack'] * away_team['away_defense']
            a_exp = league_avgs['away_goals'] * away_team['away_attack'] * home_team['home_defense']
            
            match_home_indices.append(team_indices[lid])
            match_away_indices.append(team_indices[vid])
            match_home_exp.append(h_exp)
            match_away_exp.append(a_exp)

    # Arrays: (n_matches,)
    m_h_idx = np.array(match_home_indices, dtype=np.int32)
    m_a_idx = np.array(match_away_indices, dtype=np.int32)
    m_h_exp = np.array(match_home_exp, dtype=np.float32)
    m_a_exp = np.array(match_away_exp, dtype=np.float32)
    
    num_matches = len(m_h_idx)
    
    results = defaultdict(lambda: {'1st': 0, 'top4': 0, 'uel': 0, 'uecl': 0, 'relegation': 0, 'total_points': 0, 'positions': [0]*25})

    print(f"  Simulating {n_simulations} seasons with {num_matches} remaining matches (Batched)...")

    # --- BATCHED VECTORIZED SIMULATION ---
    BATCH_SIZE = 100000 
    
    for start_idx in range(0, n_simulations, BATCH_SIZE):
        current_batch_size = min(BATCH_SIZE, n_simulations - start_idx)
        
        # 1. Generate Goals for Batch
        sim_home_goals = np.random.poisson(lam=m_h_exp, size=(current_batch_size, num_matches))
        sim_away_goals = np.random.poisson(lam=m_a_exp, size=(current_batch_size, num_matches))
        
        # 2. Points & GD
        sim_h_pts = np.where(sim_home_goals > sim_away_goals, 3, 
                             np.where(sim_home_goals == sim_away_goals, 1, 0))
        sim_a_pts = np.where(sim_away_goals > sim_home_goals, 3, 
                             np.where(sim_home_goals == sim_away_goals, 1, 0))
                             
        sim_h_gd = sim_home_goals - sim_away_goals
        sim_a_gd = sim_away_goals - sim_home_goals
        
        # 3. Aggregate to Teams
        batch_points = np.tile(current_points_vec, (current_batch_size, 1))
        batch_gd = np.tile(current_gd_vec, (current_batch_size, 1))
        
        for i in range(num_matches):
            batch_points[:, m_h_idx[i]] += sim_h_pts[:, i]
            batch_points[:, m_a_idx[i]] += sim_a_pts[:, i]
            batch_gd[:, m_h_idx[i]] += sim_h_gd[:, i]
            batch_gd[:, m_a_idx[i]] += sim_a_gd[:, i]

        # 4. Rank Teams
        sort_score = batch_points * 10000 + batch_gd
        ranked_indices = np.argsort(-sort_score, axis=1)
        
        # 5. Extract Statistics form Batch
        for rank in range(num_teams):
            teams_at_rank = ranked_indices[:, rank]
            counts = np.bincount(teams_at_rank, minlength=num_teams)
            
            for t_idx, count in enumerate(counts):
                if count > 0:
                    tid = reverse_indices[t_idx]
                    res = results[tid]
                    res['positions'][rank+1] += int(count)
                    
                    if rank == 0: res['1st'] += int(count)
                    if rank < 4: res['top4'] += int(count)
                    if rank == 4 or rank == 5: res['uel'] += int(count)
                    if rank == 6: res['uecl'] += int(count)
                    if rank >= num_teams - 3: res['relegation'] += int(count)

        # Total points aggregation
        total_pts_sum = batch_points.sum(axis=0)
        for t_idx in range(num_teams):
            tid = reverse_indices[t_idx]
            results[tid]['total_points'] += float(total_pts_sum[t_idx])
            
    print(f"  Simulation completed in {time.time() - start_time:.2f} seconds.")

    # 3. Format Output
    final_output = []
    for tid, info in team_profiles.items():
        res = results[tid]
        games_simulated = n_simulations
        
        att_disp = (info['home_attack'] + info['away_attack']) / 2
        def_disp = (info['home_defense'] + info['away_defense']) / 2
        score_att = 5.0 + (att_disp - 1.0) * 5.0
        score_def = 10 - (def_disp - 0.4) * 8
        
        final_output.append({
            'team_name': info['name'],
            'win_probability': round((res['1st'] / games_simulated) * 100, 1),
            'top4_probability': round((res['top4'] / games_simulated) * 100, 1),
            'uel_probability': round((res['uel'] / games_simulated) * 100, 1),
            'uecl_probability': round((res['uecl'] / games_simulated) * 100, 1),
            'relegation_probability': round((res['relegation'] / games_simulated) * 100, 1),
            'avg_points': round(res['total_points'] / games_simulated, 1),
            'attack_rating': max(1.0, min(9.9, round(score_att, 1))),
            'defense_rating': max(1.0, min(9.9, round(score_def, 1))),
            'position_distribution': [round((x/games_simulated)*100, 1) for x in res['positions'][1:]] 
        })
        
    final_output.sort(key=lambda x: x['avg_points'], reverse=True)
    return final_output

if __name__ == "__main__":
    
    all_results = {}
    
    print("--- STARTING EUROPEAN LEAGUES SIMULATION ---")
    
    for league_name, file_prefix in LEAGUES.items():
        stats_file = f"{file_prefix}_stats.csv"
        fixtures_file = f"{file_prefix}_fixtures.csv"
        
        if os.path.exists(stats_file) and os.path.exists(fixtures_file):
            print(f"\nProcessing {league_name}...")
            try:
                # Use simple read_csv
                stats = pd.read_csv(stats_file)
                fixtures = pd.read_csv(fixtures_file)
                
                # Check for necessary columns
                if 'home_goles_favor' not in stats.columns:
                    print(f"  SKIPPING {league_name}: Missing home/away split columns in CSV.")
                    continue
                    
                league_results = run_simulation(stats, fixtures)
                all_results[league_name] = league_results
                print(f"  {league_name} done. Top Team: {league_results[0]['team_name']} ({league_results[0]['win_probability']}%)")
                
            except Exception as e:
                print(f"  ERROR processing {league_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\nSkipping {league_name}: Data files not found ({stats_file})")

    # Serialize all results
    js_content = f"const simulationResults = {json.dumps(all_results, indent=2)};"
    with open("simulation_data.js", "w", encoding='utf-8') as f:
        f.write(js_content)
        
    print("\nAll simulations completed. Results saved to 'simulation_data.js'")
