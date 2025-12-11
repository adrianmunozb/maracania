from curl_cffi import requests
import pandas as pd
import time
import random

# --- CONFIGURACIÓN ---
LEAGUES = {
    'La Liga': 8,
    'Premier League': 17,
    'Serie A': 23,
    'Bundesliga': 35,
    'Ligue 1': 34
}

# Headers para parecer un navegador real (Chrome)
HEADERS = {
    "authority": "api.sofascore.com",
    "accept": "*/*",
    "accept-language": "es-ES,es;q=0.9,en;q=0.8",
    "origin": "https://www.sofascore.com",
    "referer": "https://www.sofascore.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

def obtener_id_temporada_actual(tournament_id, league_name):
    """
    Consulta la API para obtener el ID de la temporada 24/25 (o la actual).
    """
    url = f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}/seasons"
    print(f"[{league_name}] Buscando temporada actual en: {url}")

    try:
        # Usamos impersonate="chrome" para engañar al sistema anti-bot
        r = requests.get(url, headers=HEADERS, impersonate="chrome")
        data = r.json()

        # La primera temporada de la lista suele ser la actual
        season_id = data['seasons'][0]['id']
        year = data['seasons'][0]['year']
        print(f"[{league_name}] Temporada encontrada: {year} (ID: {season_id})")
        return season_id
    except Exception as e:
        print(f"[{league_name}] Error obteniendo temporada: {e}")
        return None

def obtener_ids_equipos(season_id, tournament_id, league_name):
    """
    Obtiene la tabla de clasificación para sacar los IDs y datos de la tabla (Puntos, PJ, G, E, P).
    """
    print(f"[{league_name}] Obteniendo clasificaciones (Total, Local, Visitante)...")
    
    types = {'total': 'standings/total', 'home': 'standings/home', 'away': 'standings/away'}
    equipos_data = {} # Diccionario ID -> Info

    for type_name, endpoint_suffix in types.items():
        url = f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/{endpoint_suffix}"
        # print(f"  - Descargando tabla '{type_name}'...")
        
        try:
            r = requests.get(url, headers=HEADERS, impersonate="chrome")
            data = r.json()
            
            if 'standings' in data and len(data['standings']) > 0:
                tabla = data['standings'][0]['rows']
                
                for fila in tabla:
                    team_id = fila['team']['id']
                    
                    if team_id not in equipos_data:
                        equipos_data[team_id] = {
                            'id': team_id,
                            'nombre': fila['team']['name']
                        }
                    
                    # Guardamos con prefijo (ej: home_puntos, away_goals)
                    prefix = "" if type_name == 'total' else f"{type_name}_"
                    
                    equipos_data[team_id][f'{prefix}puntos'] = fila.get('points', 0)
                    equipos_data[team_id][f'{prefix}partidos_jugados'] = fila.get('matches', 0)
                    equipos_data[team_id][f'{prefix}goles_favor'] = fila.get('scoresFor', 0)
                    equipos_data[team_id][f'{prefix}goles_contra'] = fila.get('scoresAgainst', 0)
            
            time.sleep(1) # Pausa entre llamadas

        except Exception as e:
            print(f"[{league_name}] Error obteniendo tabla {type_name}: {e}")
            return {}

    print(f"[{league_name}] Se encontraron {len(equipos_data)} equipos en la liga.")
    return equipos_data

def obtener_partidos_restantes(season_id, tournament_id, league_name):
    """
    Obtiene los partidos no jugados de la temporada.
    """
    print(f"[{league_name}] Obteniendo calendario de partidos restantes...")
    # Primero obtenemos las rondas
    url_rounds = f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/rounds"
    
    partidos_pendientes = []
    
    try:
        r = requests.get(url_rounds, headers=HEADERS, impersonate="chrome")
        data = r.json()
        current_round = data.get('currentRound', {}).get('round', 1)
        rounds = data.get('rounds', [])
        
        # Filtramos rondas desde la actual hacia adelante
        rondas_futuras = [rnd['round'] for rnd in rounds if rnd['round'] >= current_round]
        
        print(f"[{league_name}] Ronda actual: {current_round}. Descargando rondas restantes: {len(rondas_futuras)} rondas.")
        
        for rnd in rondas_futuras:
            url_round = f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{rnd}"
            r_evt = requests.get(url_round, headers=HEADERS, impersonate="chrome")
            data_evt = r_evt.json()
            
            events = data_evt.get('events', [])
            for evt in events:
                if evt['status']['type'] == 'notstarted':
                    partido = {
                        'id': evt['id'],
                        'ronda': rnd,
                        'local_id': evt['homeTeam']['id'],
                        'local_nombre': evt['homeTeam']['name'],
                        'visitante_id': evt['awayTeam']['id'],
                        'visitante_nombre': evt['awayTeam']['name'],
                        'fecha': evt['startTimestamp']
                    }
                    partidos_pendientes.append(partido)
            
            time.sleep(random.uniform(0.5, 0.8)) # Pausa pequeña
            
        print(f"[{league_name}] Total de partidos pendientes encontrados: {len(partidos_pendientes)}")
        return partidos_pendientes

    except Exception as e:
        print(f"[{league_name}] Error obteniendo calendario: {e}")
        return []

def obtener_estadisticas_equipo(team_id, season_id, tournament_id, team_name):
    """
    Llama al endpoint de estadísticas "overall" de un equipo específico.
    """
    url = f"https://api.sofascore.com/api/v1/team/{team_id}/unique-tournament/{tournament_id}/season/{season_id}/statistics/overall"

    try:
        r = requests.get(url, headers=HEADERS, impersonate="chrome")
        if r.status_code == 200:
            data = r.json()
            stats = data.get('statistics', {})

            # Añadimos el nombre del equipo a los datos
            stats['Equipo'] = team_name
            stats['ID_Equipo'] = team_id
            return stats
        else:
            return None
    except Exception as e:
        print(f"   Excepción con {team_name}: {e}")
        return None

def process_league(league_name, tournament_id):
    print(f"\n--- PROCESANDO LIGA: {league_name.upper()} (ID: {tournament_id}) ---")
    
    # 1. Obtener ID de la temporada actual
    season_id = obtener_id_temporada_actual(tournament_id, league_name)
    if not season_id:
        print(f"Skipping {league_name}: No season ID found.")
        return

    # 2. Obtener lista de equipos y clasificación
    equipos_dict = obtener_ids_equipos(season_id, tournament_id, league_name)
    if not equipos_dict:
        print(f"Skipping {league_name}: No teams found.")
        return

    lista_stats_completa = []

    # 3. Iterar equipo por equipo
    print(f"[{league_name}] Descargando estadísticas detalladas ({len(equipos_dict)} equipos)...")
    equipos_list = list(equipos_dict.values())
    for i, eq in enumerate(equipos_list):
        if i % 5 == 0: print(f"[{league_name}] Progreso: {i}/{len(equipos_list)}") # Reduce spam

        stats = obtener_estadisticas_equipo(eq['id'], season_id, tournament_id, eq['nombre'])

        if stats:
            # Fusionamos los datos de la tabla (puntos, etc) con las estadísticas
            stats.update(eq)
            lista_stats_completa.append(stats)

        # Pausa aleatoria para evitar bloqueo (IMPORTANTE)
        time.sleep(random.uniform(0.8, 1.5))

    # 4. Obtener calendario restante
    partidos_pendientes = obtener_partidos_restantes(season_id, tournament_id, league_name)

    # Clean filename string
    safe_league_name = league_name.replace(" ", "_").lower()

    # 5. Guardar a CSV (Stats)
    if lista_stats_completa:
        df = pd.json_normalize(lista_stats_completa) # Aplana el JSON a columnas

        # Reordenamos para que el nombre salga primero
        cols_prioridad = [
            'Equipo', 'puntos', 'partidos_jugados', 'goles_favor', 'goles_contra',
            'home_puntos', 'home_goles_favor', 'home_goles_contra',
            'away_puntos', 'away_goles_favor', 'away_goles_contra'
        ]
        
        cols = [c for c in cols_prioridad if c in df.columns] + [c for c in df.columns if c not in cols_prioridad]
        df = df.loc[:, ~df.columns.duplicated()]
        df = df[cols]

        filename = f"sofascore_{safe_league_name}_stats.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"[{league_name}] ¡Éxito! Datos guardados en {filename}")
    else:
        print(f"[{league_name}] No se pudieron obtener datos de estadísticas.")

    # 6. Guardar a CSV (Fixtures)
    if partidos_pendientes:
        df_fixtures = pd.DataFrame(partidos_pendientes)
        filename_fixtures = f"sofascore_{safe_league_name}_fixtures.csv"
        df_fixtures.to_csv(filename_fixtures, index=False, encoding='utf-8-sig')
        print(f"[{league_name}] ¡Éxito! Calendario guardado en {filename_fixtures}")
    else:
        print(f"[{league_name}] No se encontraron partidos pendientes.")

def main():
    print("--- INICIANDO SCRAPER MULTI-LIGA DE SOFASCORE ---")
    
    for league_name, tournament_id in LEAGUES.items():
        try:
            process_league(league_name, tournament_id)
            print(f"Finalizado procesamiento de {league_name}.\n")
            time.sleep(2) # Pausa entre ligas
        except Exception as e:
            print(f"CRITICAL ERROR processing {league_name}: {e}")

if __name__ == "__main__":
    main()
