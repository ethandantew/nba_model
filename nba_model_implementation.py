import pandas as pd
import datetime as dt
import nba_projections
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
import matplotlib.pyplot as plt
import nba_projections_import
import numpy as np

#get columns to be used in the model
raw_model_data = nba_projections.get_model_data()
raw_model_data.to_csv('./Outputs/raw_model_data.csv', index=False)

clean_model_data = raw_model_data.dropna()

features = [
    'MIN', 'POTENTIAL_AST_L5', 'TEAM_PLUS_MINUS', 'B2B_FLAG', 'HOME_FLAG',
    'OWN_TEAM_PACE_L10', 'VS_TEAM_PACE_L10', 'VS_FGM_AVGPM_LAST10G', 'VS_FGA_AVGPM_LAST10G',
    'VS_FG3M_AVGPM_LAST10G', 'VS_FG3A_AVGPM_LAST10G', 'VS_FTM_AVGPM_LAST10G', 'VS_FTA_AVGPM_LAST10G',
    'VS_OREB_AVGPM_LAST10G', 'VS_DREB_AVGPM_LAST10G', 'VS_AST_AVGPM_LAST10G', 'VS_STL_AVGPM_LAST10G',
    'VS_BLK_AVGPM_LAST10G', 'VS_TOV_AVGPM_LAST10G', 'VS_PF_AVGPM_LAST10G', 'VS_PTS_AVGPM_LAST10G',
    'VS_POTENTIAL_AST_AVGPM_LAST10G', 'MIN_AVGLAST3GAMES', 'FGM_AVGLAST3GAMES', 'FGA_AVGLAST3GAMES',
    'FG3M_AVGLAST3GAMES', 'FG3A_AVGLAST3GAMES', 'FTM_AVGLAST3GAMES', 'FTA_AVGLAST3GAMES',
    'REB_AVGLAST3GAMES', 'AST_AVGLAST3GAMES', 'STL_AVGLAST3GAMES',
    'BLK_AVGLAST3GAMES', 'TOV_AVGLAST3GAMES', 'PF_AVGLAST3GAMES', 'PTS_AVGLAST3GAMES',
    'POTENTIAL_AST_AVGLAST3GAMES', 'MIN_STDLAST3GAMES', 'FGM_STDLAST3GAMES', 'FGA_STDLAST3GAMES',
    'FG3M_STDLAST3GAMES', 'FG3A_STDLAST3GAMES', 'FTM_STDLAST3GAMES', 'FTA_STDLAST3GAMES',
    'REB_STDLAST3GAMES', 'AST_STDLAST3GAMES', 'STL_STDLAST3GAMES',
    'BLK_STDLAST3GAMES', 'TOV_STDLAST3GAMES', 'PF_STDLAST3GAMES', 'PTS_STDLAST3GAMES',
    'POTENTIAL_AST_STDLAST3GAMES', 'MIN_AVGLAST5GAMES', 'FGM_AVGLAST5GAMES', 'FGA_AVGLAST5GAMES',
    'FG3M_AVGLAST5GAMES', 'FG3A_AVGLAST5GAMES', 'FTM_AVGLAST5GAMES', 'FTA_AVGLAST5GAMES',
    'REB_AVGLAST5GAMES', 'AST_AVGLAST5GAMES', 'STL_AVGLAST5GAMES',
    'BLK_AVGLAST5GAMES', 'TOV_AVGLAST5GAMES', 'PF_AVGLAST5GAMES', 'PTS_AVGLAST5GAMES',
    'POTENTIAL_AST_AVGLAST5GAMES', 'MIN_STDLAST5GAMES', 'FGM_STDLAST5GAMES', 'FGA_STDLAST5GAMES',
    'FG3M_STDLAST5GAMES', 'FG3A_STDLAST5GAMES', 'FTM_STDLAST5GAMES', 'FTA_STDLAST5GAMES',
    'REB_STDLAST5GAMES', 'AST_STDLAST5GAMES', 'STL_STDLAST5GAMES',
    'BLK_STDLAST5GAMES', 'TOV_STDLAST5GAMES', 'PF_STDLAST5GAMES', 'PTS_STDLAST5GAMES',
    'POTENTIAL_AST_STDLAST5GAMES', 'MIN_AVGLAST10GAMES', 'FGM_AVGLAST10GAMES', 'FGA_AVGLAST10GAMES',
    'FG3M_AVGLAST10GAMES', 'FG3A_AVGLAST10GAMES', 'FTM_AVGLAST10GAMES', 'FTA_AVGLAST10GAMES',
    'REB_AVGLAST10GAMES', 'AST_AVGLAST10GAMES', 'STL_AVGLAST10GAMES',
    'BLK_AVGLAST10GAMES', 'TOV_AVGLAST10GAMES', 'PF_AVGLAST10GAMES', 'PTS_AVGLAST10GAMES',
    'POTENTIAL_AST_AVGLAST10GAMES'
]

prop = 'AST'

X = clean_model_data[features]
y = clean_model_data[prop]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost regressor
mean_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.05,
                            max_depth=5, alpha=10, n_estimators=100)

# Train the model
mean_model.fit(X_train, y_train)

importance = mean_model.feature_importances_

# Create a Series with feature importances and index as feature names
feature_importance = pd.Series(importance, index=X.columns)

# Sort the feature importances in descending order
sorted_importance = feature_importance.sort_values(ascending=False)

# Visualize the feature importances
plt.figure(figsize=(12, 8))
sorted_importance.plot(kind='bar')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('F score')
plt.show()

y_pred = mean_model.predict(X_test)


# add the predicitons to the df
player_data_temp = clean_model_data.copy()
X_test_temp = X_test.copy()
X_test_temp[f'pred_{prop}'] = y_pred

# Initialize XGBoost regressor
lower_model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.05)
lower_model.fit(X_train, y_train)

# Train the upper quantile model
upper_model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.95)
upper_model.fit(X_train, y_train)

# Predict the quantiles
lower_pred = lower_model.predict(X_test)
upper_pred = upper_model.predict(X_test)

mean_predictions = mean_model.predict(raw_model_data[features])
lower_predictions = lower_model.predict(raw_model_data[features])
upper_predictions = upper_model.predict(raw_model_data[features])

# Add the predictions to the dataframe
raw_model_data[f'predicted_{prop}'] = mean_predictions
raw_model_data[f'lower_bound_{prop}'] = lower_predictions
raw_model_data[f'upper_bound_{prop}'] = upper_predictions

#calculate prediction accuracy
raw_model_data['pred_accuracy'] = raw_model_data[prop] - raw_model_data[f'predicted_{prop}']

# Calculate coverage probability for lower quantile
lower_quantile = 0.05  # Adjust this value according to your lower quantile
coverage_lower = np.mean((y_test >= lower_pred) & (y_test <= y_pred))

# Calculate coverage probability for upper quantile
upper_quantile = 0.95  # Adjust this value according to your upper quantile
coverage_upper = np.mean((y_test >= y_pred) & (y_test <= upper_pred))

print(f'Coverage Probability for Lower Quantile ({lower_quantile}): {coverage_lower:.2%}')
print(f'Coverage Probability for Upper Quantile ({upper_quantile}): {coverage_upper:.2%}')

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'RMSE: {rmse:.2f}')
print(f'R2: {r2:.2f}')
print(f'MAE: {mae:.2f}')

# Add the lower and upper bounds to the dataframe
# raw_model_data['lower_bound_points'] = lower_predictions
# raw_model_data['upper_bound_points'] = upper_predictions

# Calculate the estimated standard deviation and add it to the dataframe
# raw_model_data['estimated_std_dev'] = (raw_model_data['upper_bound_points'] - raw_model_data['lower_bound_points']) / (2 * 1.645)

# Calculate the interval width and estimate the standard deviation
# interval_width = upper_pred - lower_pred
# estimated_std_dev = interval_width / (2 * 1.645)

# threshold = 20
# probabilities = []

# for mean, std in zip(y_pred, estimated_std_dev):
#     prob = 1 - norm(mean, std).cdf(threshold)
#     probabilities.append(prob)
    
def predict_and_get_probabilities(player_name, vs_team, threshold):
    # Filter the player and opponent team from the latest data
    player_data = raw_model_data[(raw_model_data['PLAYER_NAME'] == player_name) &
                                    (raw_model_data['VS_TEAM_ABBREVIATION'] == vs_team)].tail(1)
    
    if player_data.empty:
        return f"No recent games found for player {player_name} against {vs_team}."
    # Get the features for prediction
    X_player = player_data[features]
    
    # Make predictions
    mean_pred = mean_model.predict(X_player)
    lower_pred = lower_model.predict(X_player)
    upper_pred = upper_model.predict(X_player)
    
    # Calculate the estimated standard deviation
    interval_width = upper_pred - lower_pred
    estimated_std_dev = interval_width / (2 * 1.645)  # Approximation for 90% CI
    
    # Calculate the probability of exceeding the threshold
    probability_over_threshold = 1 - norm(mean_pred, estimated_std_dev).cdf(threshold)
    probability_under_threshold = norm(mean_pred, estimated_std_dev).cdf(threshold)
    
    return {
        "mean_prediction": mean_pred[0],
        "lower_bound": lower_pred[0],
        "upper_bound": upper_pred[0],
        "estimated_standard_deviation": estimated_std_dev[0],
        "probability_over_threshold": probability_over_threshold[0],
        "probability_under_threshold": probability_under_threshold[0]
    }

player_name = "Devin Booker"  # Replace with actual player name
vs_team = "WAS"            # Replace with actual opponent team abbreviation
threshold = 7.5             # Replace with the desired threshold for points
result = predict_and_get_probabilities(player_name, vs_team, threshold)
print(result)

imported_projections = nba_projections_import.sportsLine()
team_lines = nba_projections_import.getLines()