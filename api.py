from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import gzip
import shutil
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Read allowed origin from environment variable or use default
ALLOWED_ORIGIN = os.getenv('ALLOWED_ORIGIN', 'http://localhost:5173')

# Configure CORS
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})

compressed_filename = 'best_random_forest.joblib.gz'
decompressed_filename = 'best_random_forest.joblib'

# Decompress the file
with gzip.open(compressed_filename, 'rb') as f_in, open(decompressed_filename, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)

# Load the model
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


        if input_data['city'] not in ['São Paulo', 'Porto Alegre', 'Rio de Janeiro', 'Campinas',
       'Belo Horizonte']:
            return jsonify({'error': 'Invalid city'}), 400

        if input_data['allows_animals'] == True:
            input_data['allows_animals'] = 'acept'
        else:
            input_data['allows_animals'] = 'not acept'

        if input_data['is_furnished'] == True:
            input_data['is_furnished'] = 'furnished'
        else:
            input_data['is_furnished'] = 'not furnished'

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
        # Verify if the CSV file exists
        fi_filepath = 'feature_importances.csv'
        if not os.path.exists(fi_filepath):
            return jsonify({'error': 'Feature importances file not found.'}), 404

        # Read the CSV
        fi_df = pd.read_csv(fi_filepath)

        # Convert to list of dictionaries
        fi_data = fi_df.to_dict(orient='records')

        return jsonify({'feature_importances': fi_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rental_trends', methods=['GET'])
def rental_trends():
    try:
        # Verify if the CSV file exists
        trends_filepath = 'rental_trends.csv'
        if not os.path.exists(trends_filepath):
            return jsonify({'error': 'Rental trends file not found.'}), 404

        # Read the CSV
        trends_df = pd.read_csv(trends_filepath)

        # Convert to list of dictionaries
        trends_data = trends_df.to_dict(orient='records')

        return jsonify({'rental_trends': trends_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
