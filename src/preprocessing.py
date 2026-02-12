"""
Data Preprocessing Module
Handles data loading, cleaning, and preprocessing for Smart Health Tracker analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class HealthDataPreprocessor:
    """
    Preprocessor for health tracker data
    """
    
    def __init__(self, filepath):
        """
        Initialize preprocessor with data file path
        
        Args:
            filepath (str): Path to the CSV data file
        """
        self.filepath = filepath
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load data from CSV file"""
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)
        print(f"Data loaded successfully! Shape: {self.df.shape}")
        return self.df
    
    def check_missing_values(self):
        """Check and display missing values"""
        print("\n" + "="*50)
        print("MISSING VALUES ANALYSIS")
        print("="*50)
        missing_percent = self.df.isnull().mean() * 100
        missing_df = pd.DataFrame({
            'Column': missing_percent.index,
            'Missing %': missing_percent.values
        }).sort_values('Missing %', ascending=False)
        
        print(missing_df)
        return missing_df
    
    def fill_missing_values(self):
        """Fill missing values with appropriate strategies"""
        print("\n" + "="*50)
        print("FILLING MISSING VALUES")
        print("="*50)
        
        # Numerical features - fill with median/mean
        numerical_cols = {
            'Age': 'median',
            'Daily_Steps': 'mean',
            'Resting_Heart_Rate': 'median',
            'Active_Heart_Rate': 'mean',
            'Hours_of_Sleep': 'mean',
            'Daily_Calorie_Intake': 'mean',
            'Sleep_Quality': 'mean'
        }
        
        for col, method in numerical_cols.items():
            if col in self.df.columns:
                if method == 'median':
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                print(f"✓ Filled {col} with {method}")
        
        # Categorical features - fill with mode
        categorical_cols = ['Gender', 'Stress_Level', 'Daily_Activity_Type', 'Mood']
        
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                print(f"✓ Filled {col} with mode")
        
        print("\nMissing values filled successfully!")
        return self.df
    
    def encode_categorical(self, columns=None):
        """
        Encode categorical variables
        
        Args:
            columns (list): List of columns to encode. If None, encodes common categorical columns
        """
        if columns is None:
            columns = ['Gender', 'Daily_Activity_Type', 'Mood']
        
        print("\n" + "="*50)
        print("ENCODING CATEGORICAL VARIABLES")
        print("="*50)
        
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"✓ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return self.df
    
    def create_derived_features(self):
        """Create useful derived features"""
        print("\n" + "="*50)
        print("CREATING DERIVED FEATURES")
        print("="*50)
        
        # Well Rested indicator (>= 7 hours of sleep)
        self.df['Well_Rested'] = (self.df['Hours_of_Sleep'] >= 7).astype(int)
        print("✓ Created 'Well_Rested' (1 if sleep >= 7 hours)")
        
        # High Sleep Quality indicator (>= 75 score)
        self.df['High_Sleep_Quality'] = (self.df['Sleep_Quality'] >= 75).astype(int)
        print("✓ Created 'High_Sleep_Quality' (1 if quality >= 75)")
        
        # Activity Level Score (normalized combination)
        self.df['Activity_Score'] = (
            (self.df['Daily_Steps'] / self.df['Daily_Steps'].max()) * 0.5 +
            (self.df['Active_Heart_Rate'] / self.df['Active_Heart_Rate'].max()) * 0.5
        )
        print("✓ Created 'Activity_Score' (normalized activity metric)")
        
        return self.df
    
    def get_clean_data(self):
        """
        Get fully preprocessed data
        
        Returns:
            pd.DataFrame: Cleaned and preprocessed dataframe
        """
        return self.df
    
    def get_summary_stats(self):
        """Display summary statistics"""
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(self.df.describe())
        
        print("\n" + "="*50)
        print("DATA TYPES")
        print("="*50)
        print(self.df.dtypes)
        
        print("\n" + "="*50)
        print("DATASET INFO")
        print("="*50)
        print(f"Total Rows: {len(self.df)}")
        print(f"Total Columns: {len(self.df.columns)}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def load_and_preprocess_data(filepath, verbose=True):
    """
    Convenience function to load and preprocess data in one step
    
    Args:
        filepath (str): Path to CSV file
        verbose (bool): Whether to print progress messages
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
        HealthDataPreprocessor: Preprocessor object for further use
    """
    preprocessor = HealthDataPreprocessor(filepath)
    
    # Load data
    preprocessor.load_data()
    
    if verbose:
        preprocessor.check_missing_values()
    
    # Fill missing values
    preprocessor.fill_missing_values()
    
    # Encode categorical variables
    preprocessor.encode_categorical()
    
    # Create derived features
    preprocessor.create_derived_features()
    
    if verbose:
        preprocessor.get_summary_stats()
    
    return preprocessor.get_clean_data(), preprocessor


if __name__ == "__main__":
    # Example usage
    filepath = "../data/smart_health_tracker_data.csv"
    df, preprocessor = load_and_preprocess_data(filepath)
    print("\n✅ Data preprocessing complete!")
    print(f"Final shape: {df.shape}")
