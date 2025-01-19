from statistics import LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv("wine+quality/winequality-red.csv", delimiter=";")


# 5. Model Building
# 	15.	Split the dataset into training and testing sets (80-20 split). What is the distribution of quality in the training and testing sets?
# 	16.	Train a Linear Regression model to predict quality. What is the R² score on the test set?
# 	17.	Train a Random Forest Regressor on the same dataset. How does its performance compare to Linear Regression?
# 	18.	Use GridSearchCV to optimize the hyperparameters of the Random Forest model. Which parameters lead to the best performance?



# Print column names for debugging
print("Columns in dataset:", data.columns)

# Drop the target column to create feature matrix (X) and target vector (y)
X = data.drop(columns=["quality"])  # Ensure 'quality' is excluded
y = data["quality"]  # Target variable

# Double-check the shapes of X and y
print(f"Shape of X (features): {X.shape}")
print(f"Shape of y (target): {y.shape}")
print()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Print the shapes of the splits
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print()
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Distribution of quality across training / test
print("\nDistribution in Training Set (y_train):")
print(y_train.value_counts(normalize=True))

print("\nDistribution in Testing Set (y_test):")
print(y_test.value_counts(normalize=True))

# Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# 	17.	Train a Random Forest Regressor on the same dataset. How does its performance compare to Linear Regression?
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results (before optimization):")
print(f"Mean Squared Error: {rf_mse:.3f}")
print(f"R² Score: {rf_r2:.3f}")

# Grid Search for Random Forest optimization
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=1),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

# Get best model predictions
best_rf_pred = rf_grid.predict(X_test)
best_rf_mse = mean_squared_error(y_test, best_rf_pred)
best_rf_r2 = r2_score(y_test, best_rf_pred)

print("\nOptimized Random Forest Results:")
print(f"Best Parameters: {rf_grid.best_params_}")
print(f"Mean Squared Error: {best_rf_mse:.3f}")
print(f"R² Score: {best_rf_r2:.3f}")

# Feature importance for the best Random Forest model
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_grid.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# 6. Evaluation and Visualization
# 	19.	Plot the predicted vs. actual wine quality for the test set. How well does the model perform?
# 	20.	Use Mean Squared Error (MSE) and Mean Absolute Error (MAE) to evaluate the model. Which metric provides more useful information in this context?
# 	21.	Plot feature importance from the Random Forest model. Which features contribute most to the prediction?


def plot_predictions(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: Simple scatterplot comparing predicted vs actual wine quality
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Adding the prediction line
    plt.plot([3, 8], [3, 8], 'r--', label='Perfect Predictions')

    # Label the plot
    plt.title('Predicted vs Actual Wine Quality')
    plt.xlabel('Actual Quality')
    plt.ylabel('Predicted Quality')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_predictions(y_test, best_rf_pred)


def calculate_errors(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: MSE, MAE to evaluate model, from this we will choose the best metric.
    """

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print("\n Model Error Measurements:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    # Interpreting
    print("\nWhat do these numbers mean?")
    print(f"On average, our predictions are off by {mae:.2f}")


calculate_errors(y_test, best_rf_pred)


def plot_top_features(model, feature_names, top_n=5):
    """

    :param model:
    :param feature_names:
    :param top_n:
    :return: Plotting the RFmodel with the top 5 features ascending.
    """
    # get feature importance
    importance = model.feature_importances_

    # create a df with features and importance as key and their value as the values xd
    features_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    # Sort and get top features
    top_features = features_df.sort_values('importance', ascending=True).tail(top_n)

    #Create bar plot
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title(f'Top {top_n} Most important features')
    plt.xlabel('importance')
    plt.tight_layout()
    plt.show()


plot_top_features(rf_grid.best_estimator_, X.columns)

