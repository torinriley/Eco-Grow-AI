import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.optimize import differential_evolution

model_path = 'xgb_model.pkl'
xgb_model = joblib.load(model_path)

crop_data = pd.read_csv('crop_data.csv')
climate_data = pd.read_csv('climate_data.csv')

crop_data = crop_data.dropna()
climate_data = climate_data.dropna()

merged_data = pd.merge(crop_data, climate_data, how='left', left_on='Planting Date', right_on='Date')

numerical_columns = ['Yield (kg/ha)', 'Temp Max (°C)', 'Temp Min (°C)', 'Rainfall (mm)', 'Humidity (%)', 'Wind Speed (km/h)', 'CO2 Concentration (ppm)']

imputer = SimpleImputer(strategy='mean')
merged_data[numerical_columns] = imputer.fit_transform(merged_data[numerical_columns])

categorical_columns = ['Crop Type', 'Variety', 'Soil Type', 'Irrigation', 'Fertilizer', 'Pest Management']
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    merged_data[col] = label_encoders[col].fit_transform(merged_data[col])

scaler = StandardScaler()
scaled_numericals = scaler.fit_transform(merged_data[numerical_columns])

crop_variety_mapping = merged_data[['Crop Type', 'Variety']].drop_duplicates()
crop_variety_dict = crop_variety_mapping.groupby('Crop Type')['Variety'].apply(list).to_dict()

def optimize_resilience_for_crop(params, crop_type, label_encoders, scaled_numericals):
    params = np.array(params).reshape(1, -1)
    crop_index = label_encoders['Crop Type'].transform([crop_type])[0]
    params[0][0] = crop_index
    sample = np.hstack((params, scaled_numericals.mean(axis=0).reshape(1, -1)))
    score = xgb_model.predict(sample)
    return -score

def optimize_for_specific_crop(crop_type, label_encoders, scaled_numericals, categorical_columns, crop_variety_dict):
    bounds = [(0, len(label_encoders[col].classes_)-1) if col != 'Crop Type' else (label_encoders['Crop Type'].transform([crop_type])[0], label_encoders['Crop Type'].transform([crop_type])[0]) for col in categorical_columns]

    def variety_constraint(params):
        params = np.array(params).reshape(1, -1)
        crop_type_encoded = int(params[0][0])
        variety_encoded = int(params[0][1])
        valid_varieties = crop_variety_dict[crop_type_encoded]
        return variety_encoded in valid_varieties

    result = differential_evolution(optimize_resilience_for_crop, bounds, args=(crop_type, label_encoders, scaled_numericals), maxiter=100, popsize=15)
    optimized_params = result.x
    optimized_params_decoded = {col: label_encoders[col].inverse_transform([int(round(optimized_params[i]))])[0] for i, col in enumerate(categorical_columns)}

    # Ensure variety matches the crop type
    crop_type_encoded = label_encoders['Crop Type'].transform([crop_type])[0]
    if optimized_params_decoded['Variety'] not in crop_variety_dict[crop_type_encoded]:
        optimized_params_decoded['Variety'] = label_encoders['Variety'].inverse_transform([crop_variety_dict[crop_type_encoded][0]])[0]  # Assign first valid variety

    return optimized_params_decoded, -result.fun

while True:
    crop_type_to_optimize = input("Enter the crop type to optimize (or 'exit' to quit): ")
    if crop_type_to_optimize.lower() == 'exit':
        break
    optimized_params_decoded, optimized_resilience_score = optimize_for_specific_crop(crop_type_to_optimize, label_encoders, scaled_numericals, categorical_columns, crop_variety_dict)
    print(f'Optimized Crop Factors for {crop_type_to_optimize}: {optimized_params_decoded}')
    print(f'Optimized Resilience Score: {optimized_resilience_score}')
