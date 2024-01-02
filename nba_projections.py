import pandas as pd
import datetime as dt
from nba_api.stats.static import teams
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelogs, teamestimatedmetrics, playerestimatedmetrics, teamgamelogs, commonteamroster, leaguedashptstats
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import matplotlib.pyplot as plt
import requests
from datetime import datetime

cur_season = '2023-24'

#set up historical dataframes

def get_rosters(cur_season):
    team_data = teams.get_teams()
    team_ids = [team['id'] for team in team_data]
    team_df = pd.DataFrame(team_data)
    # schedule_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2020/league/00_full_schedule_week_tbds.json'
    # schedule = requests.get(schedule_url).json()
    # schedule['lscd'][0]['mscd']['g'][0]['gid']
    # team_roster = commonteamroster.CommonTeamRoster(team_id='1610612740', season=cur_season).get_data_frames()[0]
    all_rosters = []
    for id in team_ids:
        try:
            team_roster = commonteamroster.CommonTeamRoster(team_id=id, season=cur_season).get_data_frames()[0]
            all_rosters.append(team_roster)
        except:
            pass
    roster_df = pd.concat(all_rosters)
    # join team_df and roster_df
    team_df_merged = team_df.merge(roster_df, how='left', left_on='id', right_on='TeamID')
    rosters = team_df_merged[['abbreviation', 'TeamID', 'PLAYER_ID', 'POSITION', 'HEIGHT', 'WEIGHT', 'AGE', 'EXP']].rename(columns={'abbreviation': 'TEAM'})
    players_data = players.get_players()
    players_df = pd.DataFrame(players_data)
    rosters = rosters.merge(players_df, how='left', left_on='PLAYER_ID', right_on='id')
    rosters = rosters[['TEAM', 'TeamID', 'PLAYER_ID', 'POSITION', 'HEIGHT', 'WEIGHT', 'AGE', 'EXP', 'full_name']].rename(columns={'full_name': 'PLAYER_NAME'})
    return rosters

def pace_calculator(FGA, FTA, OREB, OPPDREB, FG, TOV, MIN):
    return 48 * (FGA + 0.4 * FTA - 1.07 * (FGA-FG)* (OREB/(OREB + OPPDREB)) + TOV) / MIN

def remove_3(df):
    return df.iloc[3:]

def create_dictionary(window_size, prefix, lis):
    lis_modified = [item + f'_{prefix}LAST{window_size}GAMES' for item in lis]
    return dict(zip(lis, lis_modified))

def get_todays_games_player(game_date, rosters):
    url = 'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json'
    data = requests.get(url).json()
    data = data['leagueSchedule']['gameDates']
    df = pd.DataFrame(data)
    #change gameDate to datetime
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    game_date = dt.datetime.strptime(game_date, '%Y-%m-%d')
    today_game = df[df['gameDate'] == game_date]
    games_json = today_game['games'].values[0]
    games_df_list = []
    for game in games_json:
        home_team_id = game['homeTeam']['teamId']
        home_team_abbreviation = game['homeTeam']['teamTricode']
        away_team_id = game['awayTeam']['teamId']
        away_team_abbreviation = game['awayTeam']['teamTricode']
        game_id = game['gameId']
        game_date = game['gameDateEst']
        game_dict = {'GAME_ID': game_id, 'GAME_DATE': game_date, 'HOME_TEAM_ID': home_team_id, 'HOME_TEAM_ABBREVIATION': home_team_abbreviation, 'AWAY_TEAM_ID': away_team_id, 'AWAY_TEAM_ABBREVIATION': away_team_abbreviation}
        games_df_list.append(game_dict)
    games_df = pd.DataFrame(games_df_list)
    games_df['HOME_MATCHUP'] = games_df['HOME_TEAM_ABBREVIATION'] + ' vs. ' + games_df['AWAY_TEAM_ABBREVIATION']
    games_df['AWAY_MATCHUP'] = games_df['AWAY_TEAM_ABBREVIATION'] + ' @ ' + games_df['HOME_TEAM_ABBREVIATION']
    home_teams = games_df[['GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'HOME_TEAM_ABBREVIATION', 'HOME_MATCHUP']].copy()
    away_teams = games_df[['GAME_ID', 'GAME_DATE', 'AWAY_TEAM_ID', 'AWAY_TEAM_ABBREVIATION', 'AWAY_MATCHUP']].copy()
    home_teams.rename(columns={'HOME_TEAM_ID': 'TEAM_ID', 'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'HOME_MATCHUP': 'MATCHUP'}, inplace=True)
    away_teams.rename(columns={'AWAY_TEAM_ID': 'TEAM_ID', 'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'AWAY_MATCHUP': 'MATCHUP'}, inplace=True)
    team_df = pd.concat([home_teams, away_teams], ignore_index=True)
    roster_merged = team_df.merge(rosters, how='left', left_on='TEAM_ID', right_on='TeamID')
    game_date_parsed = datetime.fromisoformat(game_date.replace("Z", ""))
    formatted_game_date = game_date_parsed.strftime("%Y-%m-%d %H:%M:%S")
    roster_merged['GAME_DATE'] = pd.to_datetime(formatted_game_date)
    return roster_merged

def get_todays_games_team(game_date):
    url = 'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json'
    data = requests.get(url).json()
    data = data['leagueSchedule']['gameDates']
    df = pd.DataFrame(data)
    #change gameDate to datetime
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    game_date = dt.datetime.strptime(game_date, '%Y-%m-%d')
    today_game = df[df['gameDate'] == game_date]
    games_json = today_game['games'].values[0]
    games_df_list = []
    for game in games_json:
        home_team_id = game['homeTeam']['teamId']
        home_team_abbreviation = game['homeTeam']['teamTricode']
        away_team_id = game['awayTeam']['teamId']
        away_team_abbreviation = game['awayTeam']['teamTricode']
        game_id = game['gameId']
        game_date = game['gameDateEst']
        game_dict = {'GAME_ID': game_id, 'GAME_DATE': game_date, 'HOME_TEAM_ID': home_team_id, 'HOME_TEAM_ABBREVIATION': home_team_abbreviation, 'AWAY_TEAM_ID': away_team_id, 'AWAY_TEAM_ABBREVIATION': away_team_abbreviation}
        games_df_list.append(game_dict)
    games_df = pd.DataFrame(games_df_list)
    games_df['HOME_MATCHUP'] = games_df['HOME_TEAM_ABBREVIATION'] + ' vs. ' + games_df['AWAY_TEAM_ABBREVIATION']
    games_df['AWAY_MATCHUP'] = games_df['AWAY_TEAM_ABBREVIATION'] + ' @ ' + games_df['HOME_TEAM_ABBREVIATION']
    home_teams = games_df[['GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'HOME_TEAM_ABBREVIATION', 'HOME_MATCHUP']].copy()
    away_teams = games_df[['GAME_ID', 'GAME_DATE', 'AWAY_TEAM_ID', 'AWAY_TEAM_ABBREVIATION', 'AWAY_MATCHUP']].copy()
    home_teams.rename(columns={'HOME_TEAM_ID': 'TEAM_ID', 'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'HOME_MATCHUP': 'MATCHUP'}, inplace=True)
    away_teams.rename(columns={'AWAY_TEAM_ID': 'TEAM_ID', 'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'AWAY_MATCHUP': 'MATCHUP'}, inplace=True)
    team_df = pd.concat([home_teams, away_teams], ignore_index=True)
    game_date_parsed = datetime.fromisoformat(game_date.replace("Z", ""))
    formatted_game_date = game_date_parsed.strftime("%Y-%m-%d %H:%M:%S")
    team_df['GAME_DATE'] = pd.to_datetime(formatted_game_date)
    return team_df

def get_passing_stats(season, game_date):

    player_dash_info = leaguedashptstats.LeagueDashPtStats(
        last_n_games = 0,
        pt_measure_type = "Passing",
        player_or_team = "Player",
        per_mode_simple = "PerGame",
        season = season,
        season_type_all_star = "Regular Season",
        team_id_nullable = 0,
        opponent_team_id = 0,
        date_from_nullable = game_date,
        date_to_nullable = game_date
    )

    player_logs = player_dash_info.get_data_frames()[0]
    player_logs["GAME_DATE"] = game_date
    
    player_logs["GAME_DATE"] = pd.to_datetime(player_logs["GAME_DATE"])
    player_logs = player_logs[['PLAYER_ID','GAME_DATE','POTENTIAL_AST']]
    
    return player_logs

#only run this if you don't have player_passing_stats.csv
# passing_stats_list = []
# for game_date in game_logs['GAME_DATE'].unique():
#     player_passing_stats = get_passing_stats(cur_season, game_date)
#     passing_stats_list.append(player_passing_stats)
# player_passing_stats_df = pd.concat(passing_stats_list)
# player_passing_stats_df = player_passing_stats_df.sort_values(['PLAYER_ID', 'GAME_DATE'])
# player_passing_stats_df['POTENTIAL_AST_L5'] = player_passing_stats_df.groupby('PLAYER_ID')['POTENTIAL_AST'].transform(lambda x: x.shift().rolling(5, min_periods = 5).mean())
# player_passing_stats_df['PLAYER_ID'] = player_passing_stats_df['PLAYER_ID'].astype(str)
# player_passing_stats_df.to_csv('./Outputs/player_passing_stats.csv', index=False)

def get_player_data(cur_date):
    #Grab necessary columns
    team_gamelogs = teamgamelogs.TeamGameLogs(season_nullable=cur_season).get_data_frames()[0]
    all_game_logs = playergamelogs.PlayerGameLogs(season_nullable=cur_season, league_id_nullable='00', season_type_nullable='Regular Season').get_data_frames()[0]
    rosters = get_rosters(cur_season)
    game_logs = all_game_logs.merge(rosters, how='left', left_on='PLAYER_ID', right_on='PLAYER_ID')
    game_logs = game_logs[['SEASON_YEAR', 'PLAYER_ID', 'PLAYER_NAME_x', 'POSITION', 'HEIGHT', 'WEIGHT', 'AGE', 'EXP', 'TEAM_ID', 'TEAM', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN','FGM',
                        'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']].rename(index=str, columns={'PLAYER_NAME_x': 'PLAYER_NAME'})
    game_logs = game_logs.sort_values(by=['PLAYER_ID', 'GAME_DATE']).set_index(['GAME_ID'])
    player_games_cleaned = game_logs[['PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'TEAM_ID',
        'TEAM', 'GAME_DATE', 'MATCHUP',
        'WL', 'PTS','MIN','FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF']].reset_index()
    #PLAYER_ID got switched to integer, switching it back to a string for merge purposes.
    # player_games_cleaned = game_logs.reset_index()
    player_games_cleaned['PLAYER_ID'] = player_games_cleaned['PLAYER_ID'].astype(str)
    player_games_cleaned['TEAM'] = player_games_cleaned['MATCHUP'].str[:3]
    
    todays_games = get_todays_games_player(cur_date, rosters)
    todays_games = todays_games[['GAME_ID', 'PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'TEAM_ID', 'TEAM', 'GAME_DATE', 'MATCHUP']]
    merged_games_df = player_games_cleaned.merge(todays_games, how = 'outer', on = 'GAME_ID')
    columns_to_fill_na = ['PLAYER_ID_x', 'PLAYER_NAME_x', 'POSITION_x', 'TEAM_ID_x', 'TEAM_x', 'GAME_DATE_x', 'MATCHUP_x']
    for column in columns_to_fill_na:
        merged_games_df[column] = merged_games_df[column].fillna(merged_games_df[column.replace('_x', '_y')])
    #drop columns that were created from the merge and rename the columns that were kept
    merged_games_df = merged_games_df.drop(['PLAYER_ID_y', 'PLAYER_NAME_y', 'POSITION_y', 'TEAM_ID_y', 'TEAM_y', 'GAME_DATE_y', 'MATCHUP_y'], axis=1)
    merged_games_df = merged_games_df.rename(columns={'PLAYER_ID_x': 'PLAYER_ID', 'PLAYER_NAME_x': 'PLAYER_NAME', 'POSITION_x': 'POSITION', 'TEAM_ID_x': 'TEAM_ID', 'TEAM_x': 'TEAM', 'GAME_DATE_x': 'GAME_DATE', 'MATCHUP_x': 'MATCHUP'})
    
    current_player_passing_stats = pd.read_csv('./Outputs/player_passing_stats.csv')
    current_player_passing_stats['GAME_DATE'] = pd.to_datetime(current_player_passing_stats['GAME_DATE'])
    #get list of game_dates in palyer_games_cleaned but not in current_player_passing_stats
    game_dates = player_games_cleaned['GAME_DATE'].unique()
    current_game_dates = current_player_passing_stats['GAME_DATE'].unique()
    missing_game_dates = [date for date in game_dates if date not in current_game_dates]
    passing_stats_list = []
    passing_stats_list.append(current_player_passing_stats)
    for game_date in missing_game_dates:
        player_passing_stats = get_passing_stats("2023-24", game_date)
        passing_stats_list.append(player_passing_stats)
    player_passing_stats_df = pd.concat(passing_stats_list)
    player_passing_stats_df = player_passing_stats_df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    player_passing_stats_df['POTENTIAL_AST_L5'] = player_passing_stats_df.groupby('PLAYER_ID')['POTENTIAL_AST'].transform(lambda x: x.shift().rolling(5, min_periods = 5).mean())
    player_passing_stats_df['PLAYER_ID'] = player_passing_stats_df['PLAYER_ID'].astype(str)
    player_passing_stats_df.to_csv('./Outputs/player_passing_stats.csv', index=False)
    merged_games_df['GAME_DATE'] = pd.to_datetime(merged_games_df['GAME_DATE'])
    merged_player_data = pd.merge(merged_games_df, player_passing_stats_df, how="left", left_on=['PLAYER_ID', 'GAME_DATE'], right_on=['PLAYER_ID', 'GAME_DATE'])
    player_data_cleaned = merged_player_data.reset_index(drop=True)
    
    team_games = team_gamelogs.sort_values(['TEAM_ID','GAME_DATE']).set_index(['GAME_ID'], drop=False)
    team_games['VS_TEAM_ID'] = team_games.loc[:,('MATCHUP')].str[-3:]
    team_games = team_games[team_games.columns.drop(list(team_games.filter(regex='RANK')))]
    team_games = team_games.reset_index(drop=True)
    today_team_games = get_todays_games_team(cur_date)
    team_games_merged = team_games.merge(today_team_games, how = 'outer', on = 'GAME_ID')
    columns_to_fill_na = ['TEAM_ID_x', 'TEAM_ABBREVIATION_x', 'GAME_DATE_x', 'MATCHUP_x']
    for column in columns_to_fill_na:
        team_games_merged[column] = team_games_merged[column].fillna(team_games_merged[column.replace('_x', '_y')])
    #drop columns that were created from the merge and rename the columns that were kept
    team_games_merged = team_games_merged.drop(['TEAM_ID_y', 'TEAM_ABBREVIATION_y', 'GAME_DATE_y', 'MATCHUP_y'], axis=1)
    team_games_merged = team_games_merged.rename(columns={'TEAM_ID_x': 'TEAM_ID', 'TEAM_ABBREVIATION_x': 'TEAM_ABBREVIATION', 'GAME_DATE_x': 'GAME_DATE', 'MATCHUP_x': 'MATCHUP'})
    team_games = team_games_merged.reset_index(drop=True)
    team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])
    team_games = team_games.sort_values(['TEAM_ID','GAME_DATE']).set_index(['GAME_ID'], drop=False)
    
    team_cols = ['TEAM_NAME', 'GAME_ID','GAME_DATE', 'MATCHUP','TEAM_ABBREVIATION']
    team_cols2 = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'BLKA', 'PFD', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']
    for item in team_cols2:
        team_cols.append('TEAM_'+item)
    own_team_games = team_games[['TEAM_NAME', 'GAME_ID','GAME_DATE', 'MATCHUP', 'TEAM_ABBREVIATION', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'BLKA', 'PFD', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']].copy()
    own_team_games.columns = team_cols
    own_team_games['VS_TEAM_ID'] = own_team_games['MATCHUP'].str[-3:]
    own_team_games['PREV_GAME_DATE'] = own_team_games.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(1)
    own_team_games['DAYS_SINCE_LAST_GAME'] = (pd.to_datetime(own_team_games['GAME_DATE']) - pd.to_datetime(own_team_games['PREV_GAME_DATE'])).dt.days
    own_team_games['B2B_FLAG'] = [1 if days == 1 else 0 for days in own_team_games['DAYS_SINCE_LAST_GAME']]
    own_team_games['HOME_FLAG'] = [1 if 'vs.' in matchup else 0 for matchup in team_games['MATCHUP']]
    own_team_games = own_team_games.drop(['PREV_GAME_DATE', 'DAYS_SINCE_LAST_GAME'], axis=1)
    #creating the vs team's columns
    cols = ['GAME_DATE', 'MATCHUP','TEAM_ABBREVIATION']
    cols2 = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']
    for item in cols2:
        cols.append('VS_TEAM_'+item)
    temp_df = team_games[['GAME_DATE', 'MATCHUP','TEAM_ABBREVIATION', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']]    
    temp_df.columns = cols
    #merging the two columns together to create merged_tg which will have both the specified team's data as well as the opponent's data
    merged_tg = pd.merge(own_team_games, temp_df, how='inner', left_on=['VS_TEAM_ID', 'GAME_DATE'], right_on=['TEAM_ABBREVIATION', 'GAME_DATE'])
    # now we'll calculate pace for last 10 games
    pace_lis = ['TEAM_FGA', 'TEAM_FTA', 'TEAM_OREB', 'VS_TEAM_DREB', 'TEAM_FGM', 'TEAM_TOV', 'TEAM_MIN']
    pre_group_teams = merged_tg.set_index(['GAME_ID'], drop=True)
    teams_grouped = pre_group_teams.groupby('TEAM_ABBREVIATION_x')
    team_dict = create_dictionary(10, "AVG", pace_lis)
    windowed_sums = pd.DataFrame(teams_grouped.rolling(center=False, window=10, win_type='triang')[pace_lis].mean().shift()).rename(index=str, columns=team_dict).reset_index()
    #We can apply our pace_calculator helper now for L10 games
    windowed_sums['PACE_L10'] = pace_calculator(FGA=windowed_sums['TEAM_FGA_AVGLAST10GAMES'], FTA=windowed_sums['TEAM_FTA_AVGLAST10GAMES'],OREB=windowed_sums['TEAM_OREB_AVGLAST10GAMES'], OPPDREB=windowed_sums['VS_TEAM_DREB_AVGLAST10GAMES'], FG=windowed_sums['TEAM_FGM_AVGLAST10GAMES'], TOV=windowed_sums['TEAM_TOV_AVGLAST10GAMES'], MIN= windowed_sums['TEAM_MIN_AVGLAST10GAMES'] )
    #now join back to the original dataframe and drop the columns we don't need
    merged_tg = merged_tg.merge(windowed_sums, how='inner', left_on=['TEAM_ABBREVIATION_x', 'GAME_ID'], right_on=['TEAM_ABBREVIATION_x', 'GAME_ID'])
    merged_tg = merged_tg.drop(['TEAM_FGA_AVGLAST10GAMES', 'TEAM_FTA_AVGLAST10GAMES', 'TEAM_OREB_AVGLAST10GAMES', 'VS_TEAM_DREB_AVGLAST10GAMES', 'TEAM_FGM_AVGLAST10GAMES', 'TEAM_TOV_AVGLAST10GAMES', 'TEAM_MIN_AVGLAST10GAMES'], axis=1).set_index(['GAME_ID'], drop= True)
    # merged_tg['PACE'] =  pace_calculator(FGA=merged_tg['TEAM_FGA'], FTA=merged_tg['TEAM_FTA'],OREB=merged_tg['TEAM_OREB'], OPPDREB=merged_tg['VS_TEAM_DREB'], FG=merged_tg['TEAM_FGM'], TOV=merged_tg['TEAM_TOV'], MIN= merged_tg['TEAM_MIN'] )
    #Applying the Y-T-D mean
    expander = merged_tg.groupby('TEAM_ABBREVIATION_x').PACE_L10.expanding(min_periods=2).mean().reset_index()
    merged_tg = merged_tg.reset_index()
    #Merging this Y-T-D PACE mean with other team data
    # temp_team_df = pd.merge(left = merged_tg[['TEAM_NAME','TEAM_ABBREVIATION_x','GAME_ID', 'GAME_DATE','TEAM_PTS', 'TEAM_ABBREVIATION_y', 'VS_TEAM_PTS']], right= expander, how="inner", left_on=['TEAM_ABBREVIATION_x', 'GAME_ID'], right_on=['TEAM_ABBREVIATION_x','GAME_ID'])
    temp_team_df = pd.merge(left = merged_tg, right= expander, how="inner", left_on=['TEAM_ABBREVIATION_x', 'GAME_ID'], right_on=['TEAM_ABBREVIATION_x','GAME_ID'])
    final_team_df = pd.merge(left = temp_team_df, right= expander, how="inner", left_on=['TEAM_ABBREVIATION_y', 'GAME_ID'], right_on=['TEAM_ABBREVIATION_x','GAME_ID']).rename(index=str, columns={"PACE_L10_x": "GAME_PACE","PACE_L10_y": "OWN_TEAM_PACE_L10", "PTS_x":"PTS_ACTUAL", "PACE_L10": "VS_TEAM_PACE_L10"})
    final_team_df = final_team_df[final_team_df.columns.drop(list(final_team_df.filter(regex='RANK')))].reset_index(drop=True)
    #Merge together with player_data_cleaned
    merged_player_data = pd.merge(player_data_cleaned, final_team_df, how="inner", left_on=['TEAM', 'GAME_ID'], right_on=['TEAM_ABBREVIATION_x_x', 'GAME_ID']).rename(index=str, columns={"GAME_DATE_x": "GAME_DATE"})
    merged_player_data['GAME_DATE'] = pd.to_datetime(merged_player_data['GAME_DATE'])
    merged_player_data['PLAYER_ID'] = merged_player_data['PLAYER_ID'].astype(int)
    merged_player_data = merged_player_data.sort_values(['PLAYER_ID', 'GAME_DATE']).rename(index=str, columns={"TEAM_ABBREVIATION_y": "VS_TEAM_ABBREVIATION", "PTS_x":"PTS_ACTUAL", "WL_x":"WL"})
    merged_player_data = merged_player_data.drop(['GAME_DATE_y','TEAM_ABBREVIATION_x_x','TEAM_ABBREVIATION_x_y','GAME_PACE'],axis=1).reset_index(drop=True)
    merged_player_data['PACE'] = pace_calculator(FGA=merged_player_data['TEAM_FGA'], FTA=merged_player_data['TEAM_FTA'],OREB=merged_player_data['TEAM_OREB'], OPPDREB=merged_player_data['VS_TEAM_DREB'], FG=merged_player_data['TEAM_FGM'], TOV=merged_player_data['TEAM_TOV'], MIN= merged_player_data['TEAM_MIN'] )
    merged_player_data = merged_player_data.reset_index(drop=True)
    # add in defensive stats for L10 games
    feature_list = ['MIN','FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'POTENTIAL_AST']
    temp_merged_player_df = merged_player_data.copy()
    temp_merged_player_df = temp_merged_player_df.reset_index()
    temp_merged_player_df['GAME_DATE'] = pd.to_datetime(temp_merged_player_df['GAME_DATE'])
    temp_merged_player_df.sort_values(['VS_TEAM_ABBREVIATION', 'POSITION','GAME_DATE'], inplace=True)
    summed_by_position_date = temp_merged_player_df.groupby(['GAME_DATE', 'VS_TEAM_ABBREVIATION', 'POSITION'])[feature_list].sum()
    columns_to_divide = [col for col in summed_by_position_date.columns if col not in ['MIN']]
    summed_by_position_date[columns_to_divide] = summed_by_position_date[columns_to_divide].div(summed_by_position_date['MIN'], axis=0)
    summed_by_position_date[columns_to_divide] = summed_by_position_date[columns_to_divide]
    summed_by_position_date = summed_by_position_date.reset_index()

    #now turn them into actual averages
    for feature in feature_list:
        feature_avg_last_10 = f'VS_{feature}_AVGPM_LAST10G'
        summed_by_position_date[feature_avg_last_10] = summed_by_position_date.groupby(['VS_TEAM_ABBREVIATION', 'POSITION'])[feature] \
            .transform(lambda x: x.rolling(window=10, min_periods=3).mean().shift())

    feature_name_list = []
    for feature in feature_list:
        feature_name_list.append(f'VS_{feature}_AVGPM_LAST10G')
    #filter df to only include the columns we want
    summed_by_position_date = summed_by_position_date[['GAME_DATE', 'VS_TEAM_ABBREVIATION', 'POSITION'] + feature_name_list]
    summed_by_position_date = summed_by_position_date.rename(columns={'VS_MIN_AVGPM_LAST10G': 'VS_MIN_AVG_LAST10G'})
    merged_player_data = pd.merge(merged_player_data, summed_by_position_date, how="left", left_on=['GAME_DATE', 'VS_TEAM_ABBREVIATION', 'POSITION'], right_on=['GAME_DATE', 'VS_TEAM_ABBREVIATION', 'POSITION'])
    merged_player_data['REB'] = merged_player_data['OREB'] + merged_player_data['DREB']
    merged_player_data = merged_player_data.drop(['MATCHUP_x', 'VS_TEAM_ID'],axis=1).reset_index(drop=True)
    return merged_player_data

# player_games_cleaned = game_logs[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID',
#     'TEAM', 'GAME_DATE', 'MATCHUP',
#     'WL', 'PTS','MIN','FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF']].reset_index()
# #PLAYER_ID got switched to integer, switching it back to a string for merge purposes.
# # player_games_cleaned = game_logs.reset_index()
# player_games_cleaned['PLAYER_ID'] = player_games_cleaned['PLAYER_ID'].astype(str)

def get_model_data (cur_date):
    
    window_sizes = [3, 5, 10]
    rolling_data = pd.DataFrame()

    merged_player_data = get_player_data(cur_date)
    feature_list = ['MIN','FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'POTENTIAL_AST']
    merged_player_data_temp = merged_player_data.set_index(['GAME_ID'], drop=True)
    player_games_grouped = merged_player_data_temp.groupby(['PLAYER_ID'])
    # player_games_mean = pd.DataFrame(player_games_grouped.rolling(center=False, window=3, win_type='triang')[feature_list].mean().shift()).rename(index=str, columns=dictionary_avg)
    for window_size in window_sizes:
        dictionary_avg = create_dictionary(window_size, "AVG", feature_list)
        dictionary_std = create_dictionary(window_size, "STD", feature_list)
        
        player_games_mean = pd.DataFrame(player_games_grouped.rolling(center=False, window=window_size, win_type='triang')[feature_list].mean().shift()).rename(index=str, columns=dictionary_avg).reset_index()
        player_games_stdv = pd.DataFrame(player_games_grouped.rolling(center=False, window=window_size, win_type='triang')[feature_list].std().shift()).rename(index=str, columns=dictionary_std).reset_index()
        
        if rolling_data.empty:
            rolling_data = player_games_mean
            rolling_data = rolling_data.merge(player_games_stdv, how="inner", left_on=['PLAYER_ID', 'GAME_ID'], right_on=['PLAYER_ID', 'GAME_ID'])
        else:
            rolling_data = rolling_data.merge(player_games_mean, how="inner", left_on=['PLAYER_ID', 'GAME_ID'], right_on=['PLAYER_ID', 'GAME_ID'])
            rolling_data = rolling_data.merge(player_games_stdv, how="inner", left_on=['PLAYER_ID', 'GAME_ID'], right_on=['PLAYER_ID', 'GAME_ID'])
    # Merge rolling data with player_games_cleaned
    rolling_data['PLAYER_ID'] = rolling_data['PLAYER_ID'].astype(int)
    player_data_mod = pd.merge(merged_player_data, rolling_data, how="inner", left_on=['PLAYER_ID', 'GAME_ID'], right_on=['PLAYER_ID', 'GAME_ID'])
    # Set GAME_ID and PLAYER_ID as multi-index
    # player_data_temp = player_data.set_index(['GAME_ID'], drop=False)
    # player_pts_ytd = player_data_temp.groupby('PLAYER_ID').PTS.expanding(min_periods=2).mean().reset_index()
    # player_data_merged = pd.merge(player_data, player_pts_ytd, how="inner", left_on=['PLAYER_ID', 'GAME_ID'], right_on=['PLAYER_ID', 'GAME_ID']).rename(index=str, columns={"PTS_y": "PTS_YTD"})
    player_data_cleaned_mod = player_data_mod.set_index(['PLAYER_ID','GAME_ID']).groupby(level=0, group_keys=False).apply(remove_3).reset_index()
    for column_name in player_data_cleaned_mod.columns:
        if '_AVGPM_' in column_name:
            player_data_cleaned_mod[column_name] = player_data_cleaned_mod[column_name] * 36
    return player_data_cleaned_mod


