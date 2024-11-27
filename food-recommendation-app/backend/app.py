# Imports
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

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
        # Load and merge augmented data for menu items
        augmented_menu_data = pd.read_csv("../augmented-data/augmented_menu_data.csv")
        merged_menu_dataset = pd.concat([menu_dataset, augmented_menu_data], ignore_index=True)

        # Parse input data
        input_data = request.json  # Example: {"BMI": 25.0, "Glucose Value": 110, "Food_Name": "Big Mac"}
        food_name = input_data.get("Food_Name")
        bmi = input_data.get("BMI")
        glucose_value = input_data.get("Glucose Value", 0)  # Default glucose value to 0 if not provided

        # Lookup the food in the merged dataset
        food_row = merged_menu_dataset[merged_menu_dataset["Food_Name"] == food_name]
        if food_row.empty:
            return jsonify({"error": f"Menu item '{food_name}' not found in the dataset"}), 404

        # Extract menu item nutritional data
        food_data = food_row.iloc[0].to_dict()

        # Combine BMI, Glucose Value, and menu item's nutritional data
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

        # Create DataFrame and add missing features dynamically
        df = pd.DataFrame([combined_data])
        for feature in menu_features:
            if feature not in df.columns:
                df[feature] = 0

        # Ensure the feature order matches the training data
        df = df[menu_features]

        # Scale the input data
        scaled_data = menu_scaler.transform(df)

        # Predict recommendation
        prediction = menu_model.predict(scaled_data)[0]
        recommendation = "Recommended" if prediction == 1 else "Not Recommended"

        return jsonify({"Food_Name": food_name, "recommendation": recommendation})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Food recommendation endpoint
@app.route('/predict-food', methods=['POST'])
def predict_food():
    try:
        # Load and merge augmented data for food items
        augmented_food_data = pd.read_csv("../augmented-data/augmented_food_data.csv")
        merged_food_dataset = pd.concat([food_dataset, augmented_food_data], ignore_index=True)

        # Parse input data
        input_data = request.json  # Example: {"BMI": 25.0, "Glucose Value": 110, "Food_Name": "Apple"}
        food_name = input_data.get("Food_Name")
        bmi = input_data.get("BMI")
        glucose_value = input_data.get("Glucose Value", 0)  # Default glucose value to 0 if not provided

        # Lookup the food in the merged dataset
        food_row = merged_food_dataset[merged_food_dataset["Food_Name"].str.lower() == food_name.lower()]
        if food_row.empty:
            return jsonify({"error": f"Food '{food_name}' not found in the dataset"}), 404

        # Extract food nutritional data
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

        # Create DataFrame and add missing features dynamically
        df = pd.DataFrame([combined_data])
        for feature in food_features:
            if feature not in df.columns:
                df[feature] = 0

        # Ensure the feature order matches the training data
        df = df[food_features]

        # Scale the input data
        scaled_data = food_scaler.transform(df)

        # Predict recommendation
        prediction = food_model.predict(scaled_data)[0]
        recommendation = "Recommended" if prediction == 1 else "Not Recommended"

        return jsonify({"Food_Name": food_name, "recommendation": recommendation})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Endpoint to add new food items to the datasets
@app.route('/add-food', methods=['POST'])
def add_food():
    try:
        # Parse input JSON
        input_data = request.json
        new_food = input_data.get("data")  # New food data

        # Validate input
        if not new_food or "Food_Name" not in new_food:
            return jsonify({"error": "Invalid food data. 'Food_Name' is required"}), 400

        # Define the required columns for the food dataset
        required_columns = ['Food_Name', 'Calories', 'Sugars', 'Fats', 'Carbohydrates', 'Fiber', 'Proteins', 'Sodium']

        # Ensure that the new food data has all the required columns
        missing_columns = [col for col in required_columns if col not in new_food]
        if missing_columns:
            return jsonify({"error": f"Missing required columns: {', '.join(missing_columns)}"}), 400

        # Define file path for the food dataset
        file_path = "../augmented-data/augmented_food_data.csv"

        # Check if the CSV file exists and load it
        try:
            existing_data = pd.read_csv(file_path)
        except FileNotFoundError:
            existing_data = pd.DataFrame(columns=required_columns)

        # Check for duplicates in the Food_Name column
        if new_food["Food_Name"] in existing_data["Food_Name"].values:
            return jsonify({"error": f"Food '{new_food['Food_Name']}' already exists in the dataset"}), 400

        # Convert the new food data to a DataFrame, ensuring it is added as a new row
        new_entry = pd.DataFrame([new_food])

        # Append the new food data as a new row in the CSV
        new_entry.to_csv(file_path, mode='a', header=not existing_data.empty, index=False)

        return jsonify({"message": "New food added to augmented food dataset successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
