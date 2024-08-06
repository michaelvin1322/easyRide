import pandas as pd
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle


def train_model():
    df = pd.read_csv('data/modified_dataset.csv')
    x = df[
        [
            'trip_distance', 'puborough', 'puzone', 'puservicezone',
            'doborough', 'dozone', 'doservicezone', 'pickup_hour',
            'pickup_weekday'
         ]
    ]
    y = df['trip_duration']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize and fit the linear regression model
    model = CatBoostRegressor(iterations=10000, learning_rate=0.01, depth=8, cat_features=categorical_features, verbose=100)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate and print accuracy and determination coefficient
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    # TODO: ADD MAPE

    # Save the model as a pickle file
    with open('models/catboost_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model has been trained and saved as 'catboost_model.pkl'")


if __name__ == "__main__":
    train_model()
