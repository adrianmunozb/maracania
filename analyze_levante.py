import pandas as pd

try:
    df = pd.read_csv("sofascore_laliga_stats.csv")
    
    # Filter Levante
    levante = df[df['Equipo'] == 'Levante UD'].iloc[0]
    
    print("--- STATS FOR LEVANTE UD ---")
    print(f"Goals Conceded: {levante['goalsConceded']}")
    print(f"Big Chances Against: {levante.get('bigChancesAgainst', 'N/A')}")
    print(f"Shots On Target Against: {levante.get('shotsOnTargetAgainst', 'N/A')}")
    print(f"Matches Played: {levante['partidos_jugados']}")
    
    print("\n--- LEAGUE AVERAGES (TOTAL) ---")
    print(df[['goalsConceded', 'bigChancesAgainst', 'shotsOnTargetAgainst']].mean())
    
except Exception as e:
    print(f"Error: {e}")
