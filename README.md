# NGSIM-EDA
NGSIM Vehicle Trajectory Prediction & EDA
Intelligent Transportation Systems (ITS) Research Project

1. Project Overview
This project focuses on the analysis and modeling of vehicle trajectories using the NGSIM (Next Generation Simulation) dataset. The goal is to predict a vehicle's future longitudinal position (Local_Y) by understanding the relationship between its current kinematics and its surrounding traffic context, such as space headway and velocity.

2. Key Features
Robust Data Pipeline: Custom cleaning logic to handle common NGSIM data artifacts, including NUL byte stripping and comma-delimited numeric strings.

Exploratory Data Analysis (EDA): Visualizations of velocity distributions and the "Safety-Speed" relationship (Velocity vs. Space Headway).

Machine Learning Architecture: Implementation of a Random Forest Regressor with 500 estimators, optimized for stable and reproducible trajectory forecasting.

Scientific Reproducibility: Fixed random states and standardized evaluation metrics (Mean Absolute Error) to ensure consistent results across different environments.

3. Getting Started
Prerequisites
Python 3.8+

The NGSIM dataset file (CSV) in the root directory.

Installation
Create a Virtual Environment (Recommended):

Terminal

python -m venv ngsim_env

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

ngsim_env\Scripts\activate

Install Dependencies:
pip install -r requirements.txt


4. How to Run
Ensure your dataset is named ngsim_first_10000.csv (or update the DATA_PATH variable in the script) and run:

python ngsimtrafficanalysis.py

5. Model Evaluation
The model is evaluated using Mean Absolute Error (MAE), which represents the average physical distance (in feet) between the predicted and actual vehicle position.

Baseline MAE: ~1.8-4.5 feet (depending on sample size and model configuration).

Context: Given a standard vehicle length of ~15 feet, this error margin represents high-fidelity tracking within the vehicle's own physical footprint.


