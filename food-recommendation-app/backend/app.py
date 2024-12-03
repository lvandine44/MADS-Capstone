# Imports
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__, template_folder='../templates',  static_folder='../static')

# Load pre-trained models
menu_model = joblib.load('../../models/menu_xgb_model.pkl')
food_model = joblib.load('../../models/food_xgb_model.pkl')

# Load fitted scalers
menu_scaler = joblib.load('../../models/scaler_menu.pkl')
food_scaler = joblib.load('../../models/scaler_food.pkl')

# Load datasets
menu_dataset = pd.read_csv("../../preprocessed_data/menu_recs_samp.csv")
food_dataset = pd.read_csv("../../preprocessed_data/individual_foods_samp.csv")

# Define features used during training
menu_features = [
    'Patient_ID', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'GenHlth', 'MentHlth', 'PhysHlth',
    'Sex', 'Age', 'Glucose Value', 'Cluster', 'Calories', 'Carbohydrates',
    'Sugars', 'Fats', 'Saturated_Fats', 'Cholesterol', 'Sodium', 'Fiber',
    'Potassium', 'Proteins', 'General_Score', 'Hour', 'Day', 'Month', 'Weekday',
    'GlucoseRank_Low', 'GlucoseRank_Norm'
]
food_features = [
    'Patient_ID', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'GenHlth', 'MentHlth', 'PhysHlth',
    'Sex', 'Age', 'Glucose Value', 'Cluster', 'Calories', 'Carbohydrates',
    'Sugars', 'Fats', 'Fiber', 'Proteins', 'General_Score', 'Hour', 'Day',
    'Month', 'Weekday', 'GlucoseRank_Low', 'GlucoseRank_Norm'
]

# Home route
@app.route('/')
def home():
    return "Welcome to our Food Recommendation Application!"

# Menu recommendation endpoint
@app.route('/predict-menu', methods=['POST'])
def predict_menu():
    try:
        input_data = request.json["data"]
        food_name = input_data.get("food")
        bmi = input_data.get("bmi")
        glucose_value = input_data.get("glucose")

        # Lookup the food in the dataset
        food_row = menu_dataset[menu_dataset["Food_Name"] == food_name]
        if food_row.empty:
            return jsonify({"error": f"Menu item '{food_name}' not found in the dataset"}), 404

        # Extract food data
        food_data = food_row.iloc[0].to_dict()

        # Combine BMI, Glucose Value, and food's nutritional data
        combined_data = {
            "BMI": bmi,
            "Glucose Value": glucose_value,
            "Calories": food_data.get("Calories", 0),
            "Sugars": food_data.get("Sugars", 0),
            "Fats": food_data.get("Fats", 0),
            "Carbohydrates": food_data.get("Carbohydrates", 0),
            "Fiber": food_data.get("Fiber", 0),
            "Sodium": food_data.get("Sodium", 0),
            "Proteins": food_data.get("Proteins", 0),
        }

        # Prepare DataFrame and scale
        df = pd.DataFrame([combined_data])
        for feature in menu_features:
            if feature not in df.columns:
                df[feature] = 0
        df = df[menu_features]
        scaled_data = menu_scaler.transform(df)

        # Prediction
        prediction = menu_model.predict(scaled_data)[0]
        recommendation = "Recommended" if prediction == 1 else "Not Recommended"
        return jsonify({"Food_Name": food_name, "recommendation": recommendation})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Individual food recommendation endpoint
@app.route('/predict-food', methods=['POST'])
def predict_food():
    try:
        input_data = request.json["data"]
        food_name = input_data.get("food")
        bmi = input_data.get("bmi")
        glucose_value = input_data.get("glucose")

        if not food_name:
            return jsonify({"error": "No food selected"}), 400  # Check if food name is provided

        # Lookup food in the individual food dataset
        food_row = food_dataset[food_dataset["Food_Name"].str.lower() == food_name.lower()]
        if food_row.empty:
            return jsonify({"error": f"Food '{food_name}' not found in the dataset"}), 404

        # Extract food data
        food_data = food_row.iloc[0].to_dict()

        # Combine BMI, Glucose Value, and food's nutritional data
        combined_data = {
            "BMI": bmi,
            "Glucose Value": glucose_value,
            "Calories": food_data.get("Calories", 0),
            "Sugars": food_data.get("Sugars", 0),
            "Fats": food_data.get("Fats", 0),
            "Carbohydrates": food_data.get("Carbohydrates", 0),
            "Fiber": food_data.get("Fiber", 0),
            "Sodium": food_data.get("Sodium", 0),
            "Proteins": food_data.get("Proteins", 0),
        }

        # Prepare DataFrame and scale
        df = pd.DataFrame([combined_data])
        for feature in food_features:
            if feature not in df.columns:
                df[feature] = 0
        df = df[food_features]
        scaled_data = food_scaler.transform(df)

        # Prediction
        prediction = food_model.predict(scaled_data)[0]
        recommendation = "Recommended" if prediction == 1 else "Not Recommended"
        return jsonify({"Food_Name": food_name, "recommendation": recommendation})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Get all restaurant names
@app.route('/get-restaurants', methods=['GET'])
def get_restaurants():
    try:
        restaurant_names = menu_dataset["Restaurant"].unique().tolist()
        return jsonify({"restaurants": restaurant_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Get food items by restaurant
@app.route('/get-foods/<restaurant_name>', methods=['GET'])
def get_foods_by_restaurant(restaurant_name):
    try:
        foods = menu_dataset[menu_dataset['Restaurant'].str.lower() == restaurant_name.lower()]['Food_Name'].unique()
        return jsonify({"foods": list(foods)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Get individual foods
@app.route('/get-individual-foods', methods=['GET'])
def get_individual_foods():
    try:
        food_names = food_dataset["Food_Name"].tolist()
        return jsonify({"food_items": food_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Render the front-end HTML
@app.route('/form')
def form():
    return render_template('form-inputs.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
