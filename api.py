from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model at startup
model_filename = 'best_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

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

        # Make prediction using the loaded model
        prediction = model.predict(input_df)

        # Return the prediction as JSON
        return jsonify({'total_monthly_cost_brl': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
