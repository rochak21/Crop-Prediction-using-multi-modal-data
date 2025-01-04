import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Define paths
train_json_path = "./data/soybean_train.json"
val_json_path = "./data/soybean_val.json"

# Load JSON data
def load_json_data(json_path):
    with open(json_path, "r") as file:
        return json.load(file)

train_data = load_json_data(train_json_path)
val_data = load_json_data(val_json_path)

# Function to load CSV data based on paths
def load_csv_data(paths):
    combined_df = pd.DataFrame()
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading file {path}: {e}")
        else:
            print(f"File not found: {path}")
    return combined_df

# Extract paths for USDA, Sentinel-2, and HRRR datasets
usda_paths = [obj['data']['USDA'] for obj in train_data]
sentinel_paths = [path for obj in train_data for path in obj['data']['sentinel']]
short_term_hrrr_paths = [path for obj in train_data for path in obj['data']['HRRR']['short_term']]
long_term_hrrr_paths = [paths for obj in train_data for paths in obj['data']['HRRR']['long_term']]

# Load datasets
usda_data = load_csv_data(usda_paths)
sentinel_data = load_csv_data(sentinel_paths)
short_term_hrrr_data = load_csv_data(short_term_hrrr_paths)
long_term_hrrr_data = pd.concat(
    [load_csv_data(paths) for paths in long_term_hrrr_paths], ignore_index=True
)

# Function to plot crop yield trends
def plot_crop_yield_trends(data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="year", y="YIELD", hue="state_name", marker="o")
    plt.title("Yearly Crop Yield Trends by State")
    plt.xlabel("Year")
    plt.ylabel("Yield (bushels/acre)")
    plt.legend(title="State")
    plt.grid(True)
    plt.show()

# Function to plot weather correlations
def plot_weather_correlations(data):
    weather_features = [
        'Avg Temperature (K)', 'Precipitation (kg m**-2)', 'Relative Humidity (%)',
        'Wind Speed (m s**-1)', 'Vapor Pressure Deficit (kPa)'
    ]
    plt.figure(figsize=(12, 8))
    corr_matrix = data[weather_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Weather Variables")
    plt.show()

# Function to visualize NDVI spatial distribution
def plot_ndvi_spatial_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Normalized_NDVI'], kde=True, bins=30)
    plt.title("Distribution of NDVI Values Across Grids")
    plt.xlabel("Normalized NDVI")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# USDA Crop Data EDA
print("\n--- USDA Crop Data Overview ---")
print(usda_data.head())
print(usda_data.describe())
plot_crop_yield_trends(usda_data)

# Weather Data EDA
print("\n--- Short-Term Weather Data Overview ---")
print(short_term_hrrr_data.head())
print(short_term_hrrr_data.describe())
plot_weather_correlations(short_term_hrrr_data)

# Sentinel NDVI Data EDA
print("\n--- Sentinel NDVI Data Overview ---")
print(sentinel_data.head())
print(sentinel_data.describe())
plot_ndvi_spatial_distribution(sentinel_data)
