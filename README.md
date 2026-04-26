# House Price Regression

## Overview

This project predicts house prices using machine learning regression models. It follows a complete ML workflow from data preprocessing to model training, evaluation, and deployment-ready `.pkl` implementation.

## Objective

The main objective of this project is to build a regression model that can predict house prices based on input features such as location, area, number of rooms, and other property-related attributes.

## Workflow

- Load and understand the dataset
- Check dataset structure and missing values
- Perform data cleaning
- Handle categorical and numerical features
- Apply encoding for categorical columns
- Split data into training and testing sets
- Train regression models
- Evaluate model performance
- Select the best-performing model
- Save the trained model using Pickle
- Load the saved `.pkl` model for future predictions

## Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## Best Model

The best-performing regression model is selected based on evaluation metrics such as:

- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- R² Score
