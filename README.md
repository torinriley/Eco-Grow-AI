
# Crop Resilience Optimization Model

This repository contains a comprehensive machine learning pipeline designed to assess and optimize crop resilience based on various environmental and agricultural factors. The model utilizes data related to soil type, irrigation methods, pest management, fertilizer usage, and climate data to predict and improve the resilience of different crop varieties.

## Features

- **Data Preprocessing**: Handles data cleaning, missing value imputation, and normalization of numerical data.
- **Machine Learning Model**: Uses a combination of Random Forest, XGBoost, and Bayesian optimization for predicting resilience and optimizing crop-specific factors.
- **Crop Variety Optimization**: Provides recommendations for optimizing specific crop varieties based on real-world data.
- **Custom Scoring Algorithm**: Calculates a resilience score based on multiple factors, giving users an actionable insight into how resilient their crops will be under certain conditions.
- **Integration-Ready**: The model is designed to allow further integration with climate data APIs, satellite imagery, IoT devices, and real-time weather sensors.

## Installation

### Requirements

To run the project, you will need the following libraries installed:

```bash
pip install numpy pandas scikit-learn xgboost bayesian-optimization joblib scipy
```

### Clone the Repository

```bash
git clone https://github.com/your-username/crop-resilience-optimization.git
cd crop-resilience-optimization
```

### Running the Model

1. Ensure you have your crop and climate data available in CSV format. Example files (`crop_data.csv`, `climate_data.csv`) are referenced in the code but **are not included in this repository**. These datasets are proprietary and not publicly available due to licensing restrictions.
   
2. Once your data is prepared, you can run the model:

```python
python main.py
```

3. The program will prompt you to enter a crop type for optimization. Based on the provided data, the model will output the optimal factors and corresponding resilience score for that crop.

## Data

The data used in this project includes:

- **Crop Data**: This includes variables such as crop type, variety, soil type, irrigation, fertilizer usage, pest management, and yield.
- **Climate Data**: Climate factors such as temperature, rainfall, humidity, wind speed, and CO2 concentration.

Please note that the crop and climate datasets used in this project are **not publicly available** in this repository. You will need to source your own data, adhering to local or global datasets where appropriate.

## Example Data Format

You will need to structure your data in CSV files with the following formats:

### `crop_data.csv`
| Crop Type | Variety         | Planting Date | Harvest Date | Yield (kg/ha) | Soil Type | Irrigation | Fertilizer | Pest Management | Resilience Score |
|-----------|-----------------|---------------|--------------|---------------|-----------|------------|------------|-----------------|------------------|
| Wheat     | Hard Red Winter  | 2023-03-01    | 2023-08-01   | 4000          | Loamy     | Drip       | NPK        | Integrated Pest  | 8.5              |
| Corn      | Yellow Dent      | 2023-04-15    | 2023-10-10   | 6500          | Sandy     | Sprinkler  | Urea       | Organic          | 7.2              |

### `climate_data.csv`
| Date       | Temp Max (°C) | Temp Min (°C) | Rainfall (mm) | Humidity (%) | Wind Speed (km/h) | CO2 Concentration (ppm) |
|------------|---------------|---------------|---------------|--------------|-------------------|-------------------------|
| 2023-03-01 | 25            | 15            | 5             | 80           | 10                | 400                     |
| 2023-03-02 | 26            | 16            | 6             | 78           | 12                | 402                     |

**Note**: The data must be cleaned and pre-processed before running the model. Any missing values should be handled appropriately (either imputed or dropped).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
