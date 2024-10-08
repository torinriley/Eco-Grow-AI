import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from scipy.optimize import differential_evolution
import joblib

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

X = np.hstack((merged_data[categorical_columns], scaled_numericals))
y = merged_data['Resilience Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

def xgb_evaluate(max_depth, gamma, colsample_bytree):
    params = {
        'max_depth': int(max_depth),
        'gamma': gamma,
        'colsample_bytree': colsample_bytree,
        'n_estimators': 500,
        'learning_rate': 0.05
    }
    xgb_model.set_params(**params)
    cv_result = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    return cv_result

xgb_bo = BayesianOptimization(
    xgb_evaluate,
    {'max_depth': (3, 10),
     'gamma': (0, 1),
     'colsample_bytree': (0.3, 0.9)}
)

xgb_bo.maximize(init_points=10, n_iter=50)

best_params = xgb_bo.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
xgb_model.set_params(**best_params)
xgb_model.fit(X_train, y_train)

joblib.dump(xgb_model, 'best_xgb_model.pkl')

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

    result = differential_evolution(optimize_resilience_for_crop, bounds, args=(crop_type, label_encoders, scaled_numericals), maxiter=100, popsize=15, constraints=variety_constraint)
    optimized_params = result.x
    optimized_params_decoded = {col: label_encoders[col].inverse_transform([int(round(optimized_params[i]))])[0] for i, col in enumerate(categorical_columns)}

    crop_type = optimized_params_decoded['Crop Type']
    if optimized_params_decoded['Variety'] not in crop_variety_dict[crop_type]:
        optimized_params_decoded['Variety'] = crop_variety_dict[crop_type][0] 

    return optimized_params_decoded, -result.fun

while True:
    crop_type_to_optimize = input("Enter the crop type to optimize (or 'exit' to quit): ")
    if crop_type_to_optimize.lower() == 'exit':
        break
    optimized_params_decoded, optimized_resilience_score = optimize_for_specific_crop(crop_type_to_optimize, label_encoders, scaled_numericals, categorical_columns, crop_variety_dict)
    print(f'Optimized Crop Factors for {crop_type_to_optimize}: {optimized_params_decoded}')
    print(f'Optimized Resilience Score: {optimized_resilience_score}')
