# UrbHost

# README: Step-by-Step Instructions for Running the Code

## **Required Libraries**
Before running the code, ensure the following libraries are installed on your system:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

**To install these libraries, run the following command in your terminal**
pip install pandas scikit-learn numpy matplotlib

## Step-by-Step Instructions

### 1. Data Cleaning
Start by preparing the raw dataset. The cleaning process involves:
Removing redundant or low-value features.
Handling missing values by applying transformations or imputations as required.
Creating new variables (e.g., binary features, categorical groupings) to enhance model performance.
Ensure to save the cleaned dataset for use in the subsequent steps.
### 2. Machine Learning Code
Once the data is cleaned:
Run all chunk blocks of the machine learning script in sequence. This includes:
Splitting the data into training and testing sets.
Preprocessing categorical and numerical features using one-hot encoding and standard scaling.
Training a linear regression model on the processed data.
Generating predictions and evaluating the model using metrics such as Mean Squared Error (MSE) and R^2
(Optional) Running a Lasso regression to explore feature selection and regularization.
Visualize and interpret the results. Ensure that all graphs and outputs are saved for further analysis.

## Key Outputs

### Model Evaluation Metrics:
Training and test set MSE and R^2 values.
Feature coefficients sorted by their impact on predictions.
### Visualizations:
Plots showing relationships between key variables (e.g., price vs. property type, cancellation policy).
### Insights:
Analysis of the most significant factors affecting Airbnb listing prices.

## Reproducing Results

Use the same raw dataset provided.
Follow the cleaning and machine learning steps outlined above.
Ensure all library dependencies are installed.
Verify the outputs align with the results detailed in the analysis.
