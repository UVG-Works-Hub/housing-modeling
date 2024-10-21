import requests

# Define the API endpoint
url = 'http://127.0.0.1:5000/predict'

# Define the input data
input_data = {
    "city": "São Paulo",
    "allows_animals": "acept",
    "is_furnished": "furnished",
    "area_sqm": 85,
    "num_rooms": 3,
    "num_bathrooms": 2,
    "num_parking_spaces": 1,
    "floor_level": 5,
    "fire_insurance_brl": 150,
    "property_tax_brl": 200
}

# Make the POST request
response = requests.post(url, json=input_data)

# Print the response
print(response.json())
