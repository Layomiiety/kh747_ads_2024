# This file contains code for suporting addressing questions in the data

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

def predict_census_variable_from_osm_features(feature_name, osm_features, training_data):
    """
    Train and evaluate a linear regression model using the provided features and training data.

    Parameters:
    - features (list): A list of feature column names to be used as predictors.
    - training_data (DataFrame): A pandas DataFrame containing the features and target variable.

    Returns:
    - dict: A dictionary containing R² score and correlation.
    """
    # Prepare feature matrix (X) and target vector (y)
    X = training_data[osm_features]
    y = training_data[feature_name]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit OLS model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate R² score and correlation
    r2 = r2_score(y_test, y_pred)
    correlation = np.corrcoef(y_test, y_pred)[0, 1]

    # Plot predicted vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
    plt.xlabel("Actual Student Percentage")
    plt.ylabel("Predicted Student Percentage")
    plt.title(f"Predicted vs Actual with R² = {r2:.4f}, Correlation = {correlation:.4f}")
    plt.legend()
    plt.grid()
    plt.show()

    # Print scores
    print(f"R² Score: {r2:.4f}")
    print(f"Correlation: {correlation:.4f}")

    return {"r2_score": r2, "correlation": correlation}


def visualize_map(buildings,buildings_with_address, final_merged):
   # Identify matched and unmatched buildings
    matched_buildings = buildings_with_address[buildings_with_address['address_key'].isin(final_merged['address_key'])]
    unmatched_buildings = buildings_with_address[~buildings_with_address['address_key'].isin(final_merged['address_key'])]

    #  Plot the buildings in the Cambridge area
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all buildings without full addresses in light gray
    buildings.plot(ax=ax, color='lightgray', edgecolor='black', label='Buildings without Full Address')

    # Plot matched buildings (green)
    matched_buildings.plot(ax=ax, color='green', edgecolor='black', label='Matched Buildings')

    # Plot unmatched buildings (red)
    unmatched_buildings.plot(ax=ax, color='red', edgecolor='black', label='Unmatched Buildings')

    # Customize the plot
    ax.set_title("OSM Buildings in Cambridge Area - Address Matches and Non-Matches")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

def find_price_area_correlation(final_merged):
    # Filter the dataset to ensure price and area data are available
    filtered_data = final_merged.dropna(subset=['price', 'area_sqm'])
    # Calculate the correlation coefficient between price and area
    correlation = filtered_data['price'].corr(filtered_data['area_sqm'])
    return correlation