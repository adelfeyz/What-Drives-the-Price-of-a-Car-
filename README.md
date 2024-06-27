# Used Car Pricing Model

This repository contains a machine learning project aimed at predicting the prices of used cars based on various features such as the car's age, odometer reading, manufacturer, model, and more. The project leverages different regression models to provide accurate price predictions, helping a used car dealership understand the factors influencing car prices.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
The goal of this project is to understand what factors make a used car more or less expensive. By analyzing the provided dataset and building predictive models, we aim to offer clear recommendations to a used car dealership regarding what consumers value in a used car.

## Dataset
The dataset used for this project contains information on 426,000 used cars. The original dataset, sourced from Kaggle, includes features such as:
- `id`
- `region`
- `price`
- `year`
- `manufacturer`
- `model`
- `condition`
- `cylinders`
- `fuel`
- `odometer`
- `title_status`
- `transmission`
- `VIN`
- `drive`
- `size`
- `type`
- `paint_color`
- `state`

## Preprocessing
To prepare the data for modeling, the following steps were performed:
1. Handling missing values
2. Filtering outliers in the `price` column
3. Dropping less relevant columns (`size`, `cylinders`, `VIN`, `paint_color`, `condition`, `id`, `drive`, `type`)

## Feature Engineering
Several feature engineering techniques were applied:
1. Created a new feature `car_age` by subtracting the manufacturing year from the current year (2024).
2. Frequency encoding for the `model` column.
3. Standardization of numerical features.
4. Polynomial features of degree 3 for numerical columns.
5. One-hot encoding for categorical variables.

## Models
The following regression models were trained and evaluated:
- Linear Regression
- Ridge Regression
- Lasso Regression

## Evaluation
The models were evaluated using Mean Squared Error (MSE) and R² scores on both training and testing datasets to assess their performance and accuracy.

## Key Findings
#### Odometer and Car Age:

Odometer: Higher odometer readings are significantly associated with a decrease in the target value. This aligns with the common understanding that vehicles with higher mileage tend to depreciate in value.
Car Age: Similarly, older vehicles (higher car age) also show a negative impact on the target value, indicating depreciation over time.
#### Model-Encoded Variables:

Both model_encoded and model_Target_encoded show positive coefficients, suggesting that certain vehicle models or brands may hold higher value or perform better in terms of the target variable.
#### Interaction Terms:

Significant interaction terms such as odometer^2, odometer car_age, and car_age model_encoded highlight the complex relationships between these features, indicating non-linear effects and interactions that are critical to understanding vehicle valuation.
#### Regional and Manufacturer Effects:

The analysis revealed significant regional variations, with some regions (e.g., region_bellingham) positively impacting the target variable and others (e.g., region_farmington) having a negative impact. This suggests regional market conditions and preferences significantly influence vehicle value.
Manufacturer-specific effects were also observed, indicating that vehicles from certain manufacturers hold different values.

## Usage
To use the car pricing prediction model:
1. Ensure you have the necessary libraries installed: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`.
2. Prepare your input data in the same format as the training data.
3. Load the trained model and use the `predict_car_price` function to predict car prices.

## Files
- [UsedCarPrice.ipynb](https://github.com/adelfeyz/What-Drives-the-Price-of-a-Car-/blob/main/UsedCarPrice.ipynb): Jupyter notebook containing the complete analysis and model training.
- [lasso_diabetes_model.pkl](https://github.com/adelfeyz/What-Drives-the-Price-of-a-Car-/blob/main/lasso_diabetes_model.pkl): Pickle file of the trained Lasso regression model.
- [coefficients.csv](https://github.com/adelfeyz/What-Drives-the-Price-of-a-Car-/blob/main/coefficients.csv): CSV file containing the coefficients of the trained model.

Example usage:
```python
def predict_car_price(model, input_data):
    """
    Predict car price using the trained model.

    Parameters:
    - model: Trained model (Linear Regression, Ridge, or Lasso)
    - input_data: Dictionary containing the input features for the new data

    Returns:
    - predicted_values: Predicted car price values
    """
    # Convert input data to DataFrame
    input_data_df = pd.DataFrame(input_data)

    # Apply the same preprocessing as the training data
    input_data_transformed = preprocessor.transform(input_data_df)

    # Make predictions
    predicted_values = model.predict(input_data_transformed)

    return predicted_values

# Example input data
example_input_data = {
    'odometer': [50000],
    'car_age': [2024 - 2015],
    'model_encoded': [model_frequency['model_name']],  
    'model_Target_encoded': [model_target_mean_train['model_name']]
    'region': ['region_name'],  
    'manufacturer': ['manufacturer_name'],  
    'fuel': ['fuel_type'],  
    'title_status': ['title_status_type'],  
    'transmission': ['transmission_type'],  
    'state': ['state_name']  
}

# Select a trained model for prediction
selected_model = models['Lasso Regression']  

predicted_price = predict_car_price(selected_model, example_input_data)
print("Predicted car price:", predicted_price)



