from sklearn.preprocessing import MinMaxScaler

features = ['Yield (kg/ha)', 'Soil Type', 'Irrigation', 'Fertilizer', 'Pest Management']

scaler = MinMaxScaler()
crop_data[features] = scaler.fit_transform(crop_data[features])

weights = {
    'Yield (kg/ha)': 0.4,
    'Soil Type': 0.2,      
    'Irrigation': 0.15,    
    'Fertilizer': 0.15,    
    'Pest Management': 0.1 
}

def calculate_resilience_score(row):
    score = 0
    for feature, weight in weights.items():
        score += row[feature] * weight
    return score * 100  # Normalize to a 100-point scale

crop_data['Calculated Resilience Score'] = crop_data.apply(calculate_resilience_score, axis=1)

tools.display_dataframe_to_user(name="Crop Data with Calculated Resilience Scores", dataframe=crop_data)
crop_data[['Crop Type', 'Variety', 'Yield (kg/ha)', 'Calculated Resilience Score']]
