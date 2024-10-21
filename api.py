from flask import Flask, request, jsonify
import joblib
import gzip
import shutil
import pandas as pd
import os

app = Flask(__name__)


compressed_filename = 'best_random_forest.joblib.gz'
decompressed_filename = 'best_random_forest.joblib'

# Descomprimir el archivo
with gzip.open(compressed_filename, 'rb') as f_in, open(decompressed_filename, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)

# Cargar el modelo
with open(decompressed_filename, 'rb') as file:
    model_pipeline = joblib.load(file)

# Define the expected input features
categorical_features = ['city', 'allows_animals', 'is_furnished']
numerical_features = [
    'area_sqm',
    'num_rooms',
    'num_bathrooms',
    'num_parking_spaces',
    'floor_level',
    'fire_insurance_brl',
    'property_tax_brl',
]

@app.route('/')
def home():
    return "Welcome to the Total Monthly Cost Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Ensure all required features are present
        input_data = {}
        for feature in categorical_features + numerical_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            input_data[feature] = data[feature]

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Make prediction using the loaded pipeline
        prediction = model_pipeline.predict(input_df)

        # Return the prediction as JSON
        return jsonify({'total_monthly_cost_brl': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    try:
        # Verificar si el archivo CSV existe
        fi_filepath = 'feature_importances.csv'
        if not os.path.exists(fi_filepath):
            return jsonify({'error': 'Feature importances file not found.'}), 404

        # Leer el CSV
        fi_df = pd.read_csv(fi_filepath)

        # Convertir a lista de diccionarios
        fi_data = fi_df.to_dict(orient='records')

        return jsonify({'feature_importances': fi_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rental_trends', methods=['GET'])
def rental_trends():
    try:
        # Verificar si el archivo CSV existe
        trends_filepath = 'rental_trends.csv'
        if not os.path.exists(trends_filepath):
            return jsonify({'error': 'Rental trends file not found.'}), 404

        # Leer el CSV
        trends_df = pd.read_csv(trends_filepath)

        # Convertir a lista de diccionarios
        trends_data = trends_df.to_dict(orient='records')

        return jsonify({'rental_trends': trends_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)

