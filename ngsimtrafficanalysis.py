import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class NGSIMTrafficAnalysis:
    """
    A unified tool to explore NGSIM data and build a simple
    predictive model for vehicle positioning based on a few parameters.
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

        # Initialize the Random Forest model
        self.model = RandomForestRegressor(n_estimators=500, random_state=42)

        # Alternatively, we could use Gradient Boosting, we'll try both
        #self.model = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.1,max_depth=7, n_iter_no_change=20, random_state=42)


    def load_and_prepare_data(self, sample_size=10000):
        """
        Handles the heavy lifting of loading and cleaning. 
        It strips out null bytes and fixes those pesky comma-strings.
        """
        try:
            print(f"Reading {self.file_path}...")
            
            # Open as raw text first to strip NUL bytes that crash standard loaders
            with open(self.file_path, 'r', encoding='latin1', errors='ignore') as f:
                raw_content = f.read().replace('\0', '')
            
            # Convert the cleaned string into a dataframe
            self.df = pd.read_csv(io.StringIO(raw_content), sep=None, engine='python').head(sample_size)

            # Columns like 'Space_Headway' can cause issues because of commas (e.g., "1,150")
            numeric_cols = ['Space_Headway', 'Time_Headway', 'v_Vel', 'v_Acc', 'Local_X', 'Local_Y']
            for col in numeric_cols:
                if col in self.df.columns and self.df[col].dtype == 'object':
                    self.df[col] = pd.to_numeric(self.df[col].str.replace(',', ''), errors='coerce')

            # Logic for Leader Vehicles: A 0 headway means nobody is in front.
            # We set this to 999 so the model knows the road is clear.
            if 'Space_Headway' in self.df.columns:
                self.df['Space_Headway'] = self.df['Space_Headway'].replace(0, 999).fillna(999)
                

            print(f"Successfully loaded and cleaned {len(self.df)} rows.")
            
        except Exception as e:
            print(f"Loading failed: {e}")
            self.df = None

    def perform_eda(self):
        """
        Basic exploratory data analysis with plots.
        We're looking for how speed interacts with safety margins.
        """
        if self.df is None: return

        print("\n--- Running Exploratory Data Analysis ---")
        
        # Plot 1: Velocity Distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df['v_Vel'], kde=True, color='teal')
        plt.title('How Fast are Vehicles Moving? (Velocity Distribution)')
        plt.xlabel('Velocity (ft/s)')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        plt.show()

        # Plot 2: The Safety-Speed Curve
        # This shows if drivers give more room when they go faster
        plt.figure(figsize=(10, 5))
        plt.scatter(self.df['Space_Headway'], self.df['v_Vel'], alpha=0.3, color='coral', s=10)
        plt.title('The Relationship Between Following Distance and Speed')
        plt.xlabel('Space Headway (ft)')
        plt.ylabel('Velocity (ft/s)')
        plt.xlim(0, 400) 
        plt.show()

        # Print basic stats to the console
        print("\nQuick Stats Summary:")
        print(self.df[['v_Vel', 'v_Acc', 'Space_Headway']].describe())

    def train_trajectory_model(self):
        """
        Trains the selected model to predict where a car will be (Local_Y)
        based on its movement and its relationship to the car in front.
        """
        if self.df is None: return

        print("\n--- Training Predictive Model ---")
        
        # Features that define the 'state' of the vehicle and its environment
        features = ['Local_X', 'v_Vel', 'v_Acc', 'Space_Headway', 'Lane_ID']
        target = 'Local_Y'
        
        # Drop rows with NaN if any slipped through cleaning
        data_clean = self.df[features + [target]].dropna()
        
        X = data_clean[features]
        y = data_clean[target]
        
        # We use a 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # Calculate MAE (Mean Absolute Error) to see how many feet off we are on average
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        print("-" * 30)
        print(f"FINAL RESULT: Mean Absolute Error is {mae:.4f} feet.")
        print("-" * 30)

if __name__ == "__main__":

    DATA_PATH = 'ngsim_first_10000.csv' #This file will be uploaded as well
    
    analysis = NGSIMTrafficAnalysis(DATA_PATH)
    
    # 1. Load a 10k sample (fast to process but enough for a clear pattern)
    analysis.load_and_prepare_data(sample_size=10000)
    
    # 2. Show the plots
    analysis.perform_eda()
    
    # 3. Train and get the MAE (Mean Absolute Error)
    analysis.train_trajectory_model()