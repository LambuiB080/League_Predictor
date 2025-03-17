import pandas as pd
from datetime import date


class DataPreparation:

    teams_as_numbers = {
        'Liverpool': 1,
        'Man City': 2,
        'Arsenal': 3,
        'Chelsea': 4,
        'Man United': 5,
        'Brentford': 6,
        'Everton': 7,
        'Ipswich': 8,
        'Leicester': 9,
        r"Nott'm Forest": 10,
        'West Ham': 11,
        'Newcastle': 12,
        'Southampton': 13,
        'Brighton': 14,
        'Crystal Palace': 15,
        'Fulham': 16,
        'Aston Villa': 17,
        'Bournemouth': 18,
        'Tottenham': 19,
        'Wolves': 20,
        'Sheffield United': 21,
        'Luton': 22,
        'Burnley': 23,
        'Leeds': 24,
        'Watford': 25,
        'Norwich': 26,
        'West Brom': 27
    }

    match_result_as_number = {
        'H': 1,
        'A': 2,
        'D': 0
    }

    def __init__(self, league_name, *args):
        self.league_name = league_name
        self.csv_databases = args
        self.df, self.df_copy = self._create_dataframe()
        #df means dataframe


    def _create_dataframe(self):
        df = self._concatenate_databases()
        df = self._delete_columns(df)
        df = self._add_columns(df)
        df = self._replace_names_with_numbers(df)
        return df, df.copy()

    def calculate_data_for_model(self, home_team=None, away_team=None):
        if home_team is not None and away_team is not None: x_teams_as_numbers = [home_team, away_team]
        else: x_teams_as_numbers = self.teams_as_numbers.values()

        for team_num in x_teams_as_numbers:
            if not self._is_team_in_df(team_num):
                continue

            df_home_team, df_away_team, df_home_away_team = self._get_team_dataframes(team_num)

            df_home_away_team.apply(lambda row: self._count_data_for_every_team_match(df_home_away_team, row.name, team_num), axis=1)
            self._count_home_away_data_for_teams(df_away_team, df_home_team, team_num)

    def prepare_df_for_training(self):
        self.df = self.df.drop(['Date'], axis=1, errors='ignore')
        self.df = self.df.astype('float64')
        self.df = self.df.dropna()
        self.df.reset_index(drop=True, inplace=True)

    def _count_home_away_data_for_teams(self, df_away_team, df_home_team, team_num):
        self._count_last_5_matches_points_for_team_where_they_play(df_home_team, 'H')
        self._count_average_goals_for_team_where_they_play(df_home_team, team_num, 'H')
        self._count_last_5_matches_points_for_team_where_they_play(df_away_team, 'A')
        self._count_average_goals_for_team_where_they_play(df_away_team, team_num, 'A')

    def _count_data_for_every_team_match(self, df_home_away_team, row_id, team_num):
        self._count_team_last_5_matches_points(df_home_away_team, row_id, team_num)
        self._count_team_param_average(df_home_away_team, team_num, row_id, 'FTHG', 'FTAG', 'Avg5HTG', 'Avg5ATG', avg_last_5=True)
        self._count_team_param_average(df_home_away_team, team_num, row_id, 'HS', 'AS', 'HAvgS', 'AAvgS', avg_last_5=False)
        self._count_team_param_average(df_home_away_team, team_num, row_id, 'HST', 'AST', 'HAvgST', 'AAvgST', avg_last_5=False)
        self._count_team_param_average(df_home_away_team, team_num, row_id, 'HY', 'AY', 'HAvgY', 'AAvgY', avg_last_5=False)

    def _count_last_5_matches_points_for_team_where_they_play(self, df_team, home_or_away):
        if home_or_away == 'H':
            column = 'PH5H'
            H = 3
            A = 0
        else:
            column = 'PA5A'
            H = 0
            A = 3
        for z in range(1, 6):
            shifted = df_team['FTR'].shift(z)
            self.df.loc[df_team.index[shifted == 1], column] += H
            self.df.loc[df_team.index[shifted == 2], column] += A
            self.df.loc[df_team.index[shifted == 0], column] += 1

    def _count_average_goals_for_team_where_they_play(self, df_team, team_num, home_away):
        if home_away == 'H':
            home_goals = 'FTHG'
            team = 'HomeTeam'
            new_column = 'AvgHTHG'
        else:
            home_goals = 'FTAG'
            team = 'AwayTeam'
            new_column = 'AvgATAG'
        avg_hg = df_team[home_goals].shift(1).rolling(window=len(df_team) - 1, min_periods=1).mean()
        self.df.loc[self.df[team] == team_num, new_column] = avg_hg

    def _count_team_last_5_matches_points(self, df_team, row_id, team_num):
        if row_id < 1: return
        recent_5_matches = df_team.iloc[max(0, row_id - 5): row_id]
        val = recent_5_matches.apply(
            lambda row: (
                3 if row['FTR'] == 1 and row['HomeTeam'] == team_num else
                3 if row['FTR'] == 2 and row['AwayTeam'] == team_num else
                1 if row['FTR'] == 0 else
                0
            ),
            axis=1
        ).sum()

        if df_team.loc[row_id, 'HomeTeam'] == team_num:
            self.df.loc[(self.df['HomeTeam'] == team_num) & (self.df['AwayTeam'] == df_team.loc[row_id, 'AwayTeam']) & (
                    self.df['Date'] == df_team.loc[row_id, 'Date']), 'PH5'] = val
        elif df_team.loc[row_id, 'AwayTeam'] == team_num:
            self.df.loc[(self.df['AwayTeam'] == team_num) & (self.df['HomeTeam'] == df_team.loc[row_id, 'HomeTeam']) & (
                    self.df['Date'] == df_team.loc[row_id, 'Date']), 'PA5'] = val

    def _count_team_param_average(self, df_team, team_num, row_id, home_param, away_param, home_param_result, away_param_result, avg_last_5):
        if row_id < 1: return

        avg_denominator = len(df_team)
        if avg_last_5: avg_denominator = 5

        recent_matches = df_team.iloc[max(0, row_id - avg_denominator): row_id]
        val = recent_matches.apply(
            lambda row: (
                row[home_param] if row['HomeTeam'] == team_num else
                row[away_param] if row['AwayTeam'] == team_num else
                0
            ),
            axis=1
        ).sum()

        if df_team.loc[row_id, 'HomeTeam'] == team_num and row_id != 0:
            self.df.loc[(self.df['HomeTeam'] == team_num) & (self.df['AwayTeam'] == df_team.loc[row_id, 'AwayTeam']) & (self.df['Date'] == df_team.loc[row_id, 'Date']), home_param_result] = val / min(row_id, avg_denominator)
        elif df_team.loc[row_id, 'AwayTeam'] == team_num and row_id != 0:
            self.df.loc[(self.df['AwayTeam'] == team_num) & (self.df['HomeTeam'] == df_team.loc[row_id, 'HomeTeam']) & (self.df['Date'] == df_team.loc[row_id, 'Date']), away_param_result] = val / min(row_id, avg_denominator)

    def _concatenate_databases(self):
        df = None
        try:
            df = pd.concat([pd.read_csv(database, header=0) for database in self.csv_databases], ignore_index=True)
        except FileNotFoundError:
            print('Check your Internet connection')
        return df

    def _delete_columns(self, df):
        df = df.drop(df.columns[-108:], axis=1, errors='ignore')
        df = df.drop(['Div', 'Time', 'Referee', 'HTHG', 'HTAG', 'HTR', 'HC', 'AC', 'HR', 'AR'], axis=1,
                     errors='ignore')
        return df

    def _add_columns(self, df):
        df['PH5H'] = 0
        df['PA5A'] = 0
        return df

    def _replace_names_with_numbers(self, df):
        pd.set_option('future.no_silent_downcasting', True)
        df = df.replace({
            'HomeTeam': self.teams_as_numbers,
            'AwayTeam': self.teams_as_numbers,
            'FTR': self.match_result_as_number
        })
        return df

    def replace_number_with_team(self, team_num):
        for team in self.teams_as_numbers.keys():
            if self.teams_as_numbers[team] == team_num:
                return team

    def calculate_data_for_inserted_teams(self, home_team, away_team):
        self.df = self.df_copy.copy()
        self.df.loc[len(self.df)] = {'Date': date.today().strftime("%d/%m/%Y"), 'HomeTeam': home_team, 'AwayTeam': away_team, 'PH5H': 0.0, 'PA5A': 0.0}
        self.calculate_data_for_model(home_team=home_team, away_team= away_team)

    def _is_team_in_df(self, team_num):
        return team_num in self.df['HomeTeam'].values or team_num in self.df['AwayTeam'].values

    def _get_team_dataframes(self, team_num):
        df_home = self.df[self.df['HomeTeam'] == team_num]
        df_away = self.df[self.df['AwayTeam'] == team_num]
        df_f = pd.concat([df_home, df_away]).sort_index().reset_index(drop=True)
        return df_home, df_away, df_f

    def get_df(self):
        return self.df

    def get_teams_as_numbers(self):
        return self.teams_as_numbers

    def get_match_result_as_number(self):
        return self.match_result_as_number