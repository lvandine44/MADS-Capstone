# # Imports
# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# # Initialize Flask app
# app = Flask(__name__)

# # Load pre-trained models
# menu_model = joblib.load('../../models/menu_xgb_model.pkl')
# food_model = joblib.load('../../models/food_xgb_model.pkl')

# # Load fitted scalers
# menu_scaler = joblib.load('../../models/scaler_menu.pkl')
# food_scaler = joblib.load('../../models/scaler_food.pkl')

# # Home route
# @app.route('/')
# def home():
#     return "Welcome to our Food Recommendation Application!"

# Menu recommendation endpoint
# @app.route('/predict-menu', methods=['POST'])
# def predict_menu():
#     try:
#         # Define the expected features (from training)
#         expected_features = [
#             'Patient_ID', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
#             'HeartDiseaseorAttack', 'PhysActivity', 'GenHlth', 'MentHlth', 'PhysHlth',
#             'Sex', 'Age', 'Glucose Value', 'Cluster', 'Calories', 'Carbohydrates',
#             'Sugars', 'Fats', 'Saturated_Fats', 'Cholesterol', 'Sodium', 'Fiber',
#             'Potassium', 'Proteins', 'General_Score', 'Hour', 'Day', 'Month', 'Weekday',
#             'GlucoseRank_Low', 'GlucoseRank_Norm'
#         ]

#         # Parse input data (expects JSON)
#         input_data = request.json  # Example: {"BMI": 25.0, "Sugars": 10, "Fats": 5, "Calories": 300}
#         df = pd.DataFrame([input_data])  # Convert input JSON to a DataFrame

#         # Add missing features with default values
#         for feature in expected_features:
#             if feature not in df.columns:
#                 df[feature] = 0  # Default value for missing features

#         # Ensure DataFrame has only the expected features, in the correct order
#         df = df[expected_features]

#         # Debugging info: Print DataFrame structure
#         print("Processed DataFrame Columns:", df.columns.tolist())
#         print("Processed DataFrame Shape:", df.shape)

#         # Scale the input data
#         scaled_data = menu_scaler.transform(df)

#         # Make prediction
#         prediction = menu_model.predict(scaled_data)[0]
#         recommendation = "Recommended" if prediction == 1 else "Not Recommended"

#         # Return prediction
#         return jsonify({"recommendation": recommendation})

#     except Exception as e:
#         print("Error:", str(e))  # Debugging log
#         return jsonify({"error": str(e)}), 400



# # Food recommendation endpoint
# # Hard-code feature order for food
# expected_food_features = [
#     'Patient_ID', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
#     'HeartDiseaseorAttack', 'PhysActivity', 'GenHlth', 'MentHlth', 'PhysHlth',
#     'Sex', 'Age', 'Glucose Value', 'Cluster', 'Calories', 'Carbohydrates',
#     'Sugars', 'Fats', 'Fiber', 'Proteins', 'General_Score', 'Hour', 'Day',
#     'Month', 'Weekday', 'GlucoseRank_Low', 'GlucoseRank_Norm'
# ]

# Food recommendation endpoint
# @app.route('/predict-food', methods=['POST'])
# def predict_food():
#     try:
#         # Parse input data (expects JSON)
#         input_data = request.json  # Example: {"BMI": 25.0, "Sugars": 10, "Fats": 5, "Calories": 300}
#         df = pd.DataFrame([input_data])  # Convert input JSON to a DataFrame

#         # Align input data to the feature order
#         df = df.reindex(columns=expected_food_features, fill_value=0)

#         # Debugging info
#         print("Processed DataFrame Columns (Food):", df.columns.tolist())
#         print("Processed DataFrame Shape (Food):", df.shape)

#         # Scale the input data
#         scaled_data = food_scaler.transform(df)

#         # Make prediction
#         prediction = food_model.predict(scaled_data)[0]
#         recommendation = "Recommended" if prediction == 1 else "Not Recommended"

#         # Return prediction
#         return jsonify({"recommendation": recommendation})

#     except Exception as e:
#         print("Error (Food):", str(e))  # Debugging log
#         return jsonify({"error": str(e)}), 400

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
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
        # Parse input data
        input_data = request.json  # Example: {"BMI": 25.0, "Glucose Value": 110, "Food_Name": "Big Mac"}
        food_name = input_data.get("Food_Name")
        bmi = input_data.get("BMI")
        glucose_value = input_data.get("Glucose Value", 0)  # Default glucose value to 0 if not provided

        # Lookup the food in the menu dataset
        food_row = menu_dataset[menu_dataset["Food_Name"] == food_name]
        
        if food_row.empty:
            return jsonify({"error": f"Menu item '{food_name}' not found in the dataset"}), 404

        # Extract menu item nutritional data
        food_data = food_row.iloc[0].to_dict()

        # Combine BMI, Glucose Value, Sodium, Fiber, and menu item's nutritional data
        combined_data = {
            "BMI": bmi,
            "Glucose Value": glucose_value,
            "Calories": food_data.get("Calories", 0),
            "Sugars": food_data.get("Sugars", 0),
            "Fats": food_data.get("Fats", 0),
            "Carbohydrates": food_data.get("Carbohydrates", 0),
            "Fiber": food_data.get("Fiber", 0),
            "Sodium": food_data.get("Sodium", 0),  # Added sodium
            "Proteins": food_data.get("Proteins", 0),
        }

        # Create DataFrame and add missing features dynamically
        df = pd.DataFrame([combined_data])

        # Add missing features with default values
        for feature in menu_features:
            if feature not in df.columns:
                df[feature] = 0

        # Ensure the feature order matches the training data
        df = df[menu_features]

        # Debugging: Print the processed DataFrame
        print("Processed DataFrame Columns:", df.columns.tolist())
        print("Processed DataFrame Shape:", df.shape)

        # Scale the input data
        scaled_data = menu_scaler.transform(df)

        # Predict recommendation
        prediction = menu_model.predict(scaled_data)[0]
        recommendation = "Recommended" if prediction == 1 else "Not Recommended"

        # Return result
        return jsonify({"Food_Name": food_name, "recommendation": recommendation})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Food recommendation endpoint
@app.route('/predict-food', methods=['POST'])
def predict_food():
    try:
        # Parse input data
        input_data = request.json  # Example: {"BMI": 25.0, "Glucose Value": 110, "Food_Name": "Apple"}
        food_name = input_data.get("Food_Name")
        bmi = input_data.get("BMI")
        glucose_value = input_data.get("Glucose Value", 0)  # Default glucose value to 0 if not provided

        # Lookup the food in the individual foods dataset
        food_row = food_dataset[food_dataset["Food_Name"].str.lower() == food_name.lower()]
        
        if food_row.empty:
            return jsonify({"error": f"Food '{food_name}' not found in the dataset"}), 404

        # Extract food nutritional data
        food_data = food_row.iloc[0].to_dict()

        # Combine BMI, Glucose Value, Sodium, Fiber, and food's nutritional data
        combined_data = {
            "BMI": bmi,
            "Glucose Value": glucose_value,
            "Calories": food_data.get("Calories", 0),
            "Sugars": food_data.get("Sugars", 0),
            "Fats": food_data.get("Fats", 0),
            "Carbohydrates": food_data.get("Carbohydrates", 0),
            "Fiber": food_data.get("Fiber", 0),
            "Sodium": food_data.get("Sodium", 0),  # Added sodium
            "Proteins": food_data.get("Proteins", 0),
        }

        # Create DataFrame and add missing features dynamically
        df = pd.DataFrame([combined_data])

        # Add missing features with default values
        for feature in food_features:
            if feature not in df.columns:
                df[feature] = 0

        # Ensure the feature order matches the training data
        df = df[food_features]

        # Debugging: Print the processed DataFrame
        print("Processed DataFrame Columns (Food):", df.columns.tolist())
        print("Processed DataFrame Shape (Food):", df.shape)

        # Scale the input data
        scaled_data = food_scaler.transform(df)

        # Predict recommendation
        prediction = food_model.predict(scaled_data)[0]
        recommendation = "Recommended" if prediction == 1 else "Not Recommended"

        # Return result
        return jsonify({"Food_Name": food_name, "recommendation": recommendation})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
