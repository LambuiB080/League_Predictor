import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from DataPreparation import DataPreparation


class Model:

    def __init__(self, league_name):
        self.DataPreparation = self._select_league(league_name)
        self.X = None
        self.Y = None
        self.X_train = None
        self.y_pred = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.prediction_probability = None
        self.prediction_params = [
            'HomeTeam', 'AwayTeam', 'PH5H', 'PA5A',
            'AAvgST', 'HAvgST', 'PA5', 'PH5',
            'AvgHTHG', 'AvgATAG', 'Avg5HTG','Avg5ATG',
            'HAvgY', 'AAvgY', 'HAvgS', 'AAvgS'
        ]

    def _select_league(self, league_name):
        if league_name == 'Premier League':
            CSV_URL1 = 'https://www.football-data.co.uk/mmz4281/2122/E0.csv'
            CSV_URL2 = 'https://www.football-data.co.uk/mmz4281/2223/E0.csv'
            CSV_URL3 = 'https://www.football-data.co.uk/mmz4281/2324/E0.csv'
            CSV_URL4 = 'https://www.football-data.co.uk/mmz4281/2425/E0.csv'
            return DataPreparation(league_name, CSV_URL1, CSV_URL2, CSV_URL3, CSV_URL4)
        else:
            raise ValueError(f"{league_name} is not supported league")
            #MORE LEAGUES WILL BE ADDED SOON

    def train_model(self, label):
        self.DataPreparation.calculate_data_for_model()
        self.DataPreparation.prepare_df_for_training()
        self.X = self.DataPreparation.get_df()[self.prediction_params]
        self.Y = self.DataPreparation.get_df()['FTR']
        self._is_training_available()
        most_effective_random_state = self._find_best_model()
        self._execute_training(most_effective_random_state)
        joblib.dump({'model': self.model, 'X_test': self.X_test, 'y_test': self.y_test, 'X_train': self.X_train}, label)

    def _find_best_model(self):
        trained_models_ranking = []
        for n in range(1, 300):
            self._execute_training(n)
            trained_models_ranking.append([format(accuracy_score(self.y_test, self.y_pred)), n])
            print(round(n / 300 * 100, 1), '%')
        trained_models_ranking = sorted(trained_models_ranking)
        return trained_models_ranking[-1][1]

    def _execute_training(self, n):
        self.X_train, self.X_test, y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.33, random_state=n)
        self.model = RandomForestClassifier(n_estimators=100, random_state=n)
        self.model.fit(self.X_train, y_train)
        self.y_pred = self.model.predict(self.X_test)

    def get_model(self, label):
        loaded = joblib.load(label)
        self._is_model_available_for_download(loaded)
        self.model = loaded['model']
        self.X_test = loaded['X_test']
        self.y_test = loaded['y_test']
        self.X_train = loaded['X_train']
        self.y_pred = self.model.predict(self.X_test)
        return self.model

    def predict_result(self, home_team, away_team):
        self._is_model_loaded()
        data_to_predict_df = self._prepare_data_for_prediction(away_team, home_team)
        prediction_probabilities = self.model.predict_proba(data_to_predict_df)
        self.prediction_probability = [prediction_probabilities, home_team, away_team]
        return prediction_probabilities

    def _prepare_data_for_prediction(self, away_team, home_team):
        self.DataPreparation.calculate_data_for_inserted_teams(home_team=home_team, away_team=away_team)
        df = self.DataPreparation.get_df()
        data_for_prediction = [df.at[len(df) - 1, param] for param in self.prediction_params]
        data_for_prediction_df = pd.DataFrame([data_for_prediction], columns=self.prediction_params)
        return data_for_prediction_df

    def display_prediction(self):
        self._is_model_loaded()
        home_team = self.DataPreparation.replace_number_with_team(self.prediction_probability[1])
        away_team = self.DataPreparation.replace_number_with_team(self.prediction_probability[2])

        print(f"\n{home_team}      vs      {away_team}\n")
        print(f"Probability of {home_team} winning: {round(self.prediction_probability[0][0][1] * 100, 2)}%")
        print(f"Probability of draw: {round(self.prediction_probability[0][0][0] * 100, 2)}%")
        print(f"Probability of {away_team} winning: {round(self.prediction_probability[0][0][2] * 100, 2)}%")

    def display_model_accuracy(self):
        self._is_model_loaded()
        print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(self.y_test, self.y_pred)))

    def display_feature_scores(self):
        self._is_model_loaded()
        print(pd.Series(self.model.feature_importances_, index=self.X_train.columns).sort_values(ascending=False))

    def display_classification_report(self):
        self._is_model_loaded()
        print(classification_report(self.y_test, self.y_pred))

    def display_confusion_matrix(self):
        self._is_model_loaded()
        print('Confusion matrix\n\n', confusion_matrix(self.y_test, self.y_pred))

    def display_all_model_data(self):
        self.display_model_accuracy()
        self.display_feature_scores()
        self.display_classification_report()
        self.display_confusion_matrix()

    def get_df(self):
        return self.DataPreparation.get_df()

    def _is_model_loaded(self):
        if self.model is None: raise ValueError("Model has not been trained or loaded.")

    def _is_model_available_for_download(self, loaded):
        if loaded is None: raise ValueError('No trained model available, train new one')

    def _is_training_available(self):
        if self.X.empty or self.Y.empty: raise ValueError("Training data is empty. Check your data source.")