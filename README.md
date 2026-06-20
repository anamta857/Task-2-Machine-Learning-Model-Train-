# House Price Prediction - Machine Learning Regression Project

## Project Overview

This project implements a machine learning regression workflow using the California Housing dataset. The goal is to predict house prices based on various housing features and compare the performance of different regression models.

## Dataset

The project uses the California Housing dataset provided by Scikit-learn.

Features:

* MedInc (Median Income)
* HouseAge
* AveRooms
* AveBedrms
* Population
* AveOccup
* Latitude
* Longitude

Target:

* Price

## Data Preprocessing

The following preprocessing steps were performed:

* Data Cleaning
* Missing Value Handling
* Duplicate Removal
* Feature Selection
* Data Visualization

## Data Visualization

The following visualizations were generated:

* Correlation Heatmap
* Histograms
* Scatter Plot (Median Income vs House Price)

## Machine Learning Models

The following regression models were trained and evaluated:

### 1. Linear Regression

* R² Score: 0.5758
* MSE: 0.5559

### 2. Decision Tree Regressor

* R² Score: 0.6221
* MSE: 0.4952

## Model Comparison

After comparing both models:

**Best Model:** Decision Tree Regressor

Reason:

* Higher R² Score
* Lower Mean Squared Error (MSE)

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Project Structure

ML_Project/
│
├── dataset.csv
├── cleaned_dataset.csv
├── main.py
├── heatmap.png
├── histograms.png
├── scatter_plot.png
├── output_screenshot.png
└── README.md

## How to Run

1. Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

2. Run the project:

python main.py

## Author

BSIT Student: Anamta - Machine Learning Regression Project
