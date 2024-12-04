MS-MADS Capstone 
# Smart Meal Choices
## A Data Science Approach to Personalized Diabetes-Friendly Restaurant Meal Recommendations

#### Team 5: [Claire Bentzen](mailto:cbentzen@sandiego.edu), [Tara Dehdari](mailto:tdehdari@sandiego.edu), [Logan Van Dine](mailto:lvandine@sandiego.edu)

---

## Overview

In order to assist the millions of individuals who are diabetic or prediabetic, the team is proposing a predictive modeling recommendation system that implements aspects of classification modeling. This recommendation system will label certain meals, or ingredients, as safe depending on the measure of harmful nutritional elements in relation to one's glucose levels. With the nature of the recommendation system, accuracy must be optimized to deem the data science project a success. In order to properly optimize accuracy metrics such as precision, a comprehensive list of features will be developed, classification thresholds will be monitored, and hyperparameters will be tuned as needed.

---

## How to use:

### If Running Unique Data:
1. Obtain API from Dexcom, NutritionX, USDA.
2. Download desired datasets from [Kaggle Diabetic Data](https://www.kaggle.com/datasets/julnazz/diabetes-health-indicators-dataset).
3. Follow notebooks 3-4b.

### If Running with This Project:
1. Run notebooks 2-4b.  
   **Note**: Notebooks 2 and 3 contain data in `.csv` format.
---

## Project Structure

- **Data Scraping**: Extracting necessary data via downloads and API's
- **Data Cleaning**: Merging the two food data sources and the two patient data sources
- **Exploratory Data Analysis (EDA)**: Initial analysis of the the data, separately. Food and Patient data analyzed independently for outliers, missing data, distributions, and correlations
- **Data Preprocessing**: Create suitability scores based on diabetic guidelines
- **Regression Modeling**: Linear Regression, Random Forest Regression, XGBoost Regression, and Support Vector Regression to predict patientu suitability scores
- **Classification Modeling**: Logistic Regression, Random Forest Classification, XGBoost, Neural Network to classify binary target variable 'Recommendations'
---

## Key Metrics
- **Precision**: Optimize the measure the proportion of True Positive values to ensure that the recommendation model is suggesting proper food for diabetic patients
---

## Conclusion

The XGBoost classifier showed the optimal results by achieving great performance across both individual food and restaurant datasets. The high precision scores, 0.994 for menu data and 0.989 for individual food data, indicates that the model effectively minimizes false positives which is an important requirement for health-related recommendations. This study successfully developed a personalized restaurant meal recommendation system for individuals with diabetes based on specialized metrics and general guidelines, which aims to bridge the gap between personalized healthcare and data science.

---

## Application
### To run the application locally:
1. clone repository onto local computer
```bash
git clone https://github.com/lvandine44/MADS-Capstone
```
2. navigate to the backend directory
```bash
cd food-recommendation-app/backend
```
3. install the requirements
```bash
pip install -r requirements.txt
```
4. run the flask app 
```bash
python app.py
```
5. Open a browser and go to: [Food Recommendation App](http://127.0.0.1:5000/form)
