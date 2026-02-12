"""
Classification Models Module
Implements various classification algorithms for health prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve,
    classification_report
)
from pathlib import Path


class HealthClassifier:
    """
    Classification models for health predictions
    """
    
    def __init__(self, df, save_dir='../results'):
        """
        Initialize classifier
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            save_dir (str): Directory to save results
        """
        self.df = df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
        
    def prepare_data(self, features, target, test_size=0.2, scale=True):
        """
        Prepare data for training
        
        Args:
            features (list): List of feature column names
            target (str): Target column name
            test_size (float): Test set proportion
            scale (bool): Whether to scale features
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = self.df[features].values
        y = self.df[target].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features if requested
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.scaler = scaler
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train, class_weight='balanced'):
        """Train Logistic Regression model"""
        print("\n🔄 Training Logistic Regression...")
        model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        print("✓ Logistic Regression trained")
        return model
    
    def train_svm(self, X_train, y_train, kernel='rbf', class_weight='balanced'):
        """Train SVM model"""
        print(f"\n🔄 Training SVM ({kernel} kernel)...")
        model = SVC(kernel=kernel, class_weight=class_weight, probability=True, random_state=42)
        model.fit(X_train, y_train)
        model_name = f'SVM ({kernel})'
        self.models[model_name] = model
        print(f"✓ SVM ({kernel}) trained")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name (str): Name of the model
        
        Returns:
            dict: Dictionary of metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        }
        
        if y_prob is not None:
            metrics['ROC-AUC'] = roc_auc_score(y_test, y_prob)
        
        self.results[model_name] = metrics
        
        # Print results
        print(f"\n📊 {model_name} Results:")
        print("-" * 50)
        for metric, value in metrics.items():
            if metric != 'Model':
                print(f"{metric:15s}: {value:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, model, X_test, y_test, model_name, labels=None):
        """Plot confusion matrix"""
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        if labels is None:
            labels = ['Not Well Rested', 'Well Rested']
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
        plt.show()
    
    def plot_roc_curve(self, model, X_test, y_test, model_name):
        """Plot ROC curve"""
        if not hasattr(model, 'predict_proba'):
            print(f"⚠ {model_name} does not support probability predictions")
            return
        
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / f'roc_curve_{model_name.replace(" ", "_").lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curve to {save_path}")
        plt.show()
    
    def cross_validate(self, model, X, y, cv=5, scoring='f1'):
        """Perform cross-validation"""
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        print(f"\n📊 {cv}-Fold Cross-Validation ({scoring}):")
        print(f"Scores: {scores}")
        print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return scores
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.results:
            print("⚠ No models have been evaluated yet")
            return
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.drop('Model', axis=1, errors='ignore')
        print(results_df.to_string())
        
        # Save to CSV
        save_path = self.save_dir / 'model_comparison.csv'
        results_df.to_csv(save_path)
        print(f"\n✓ Saved comparison to {save_path}")
        
        # Plot comparison
        self._plot_model_comparison(results_df)
        
        return results_df
    
    def _plot_model_comparison(self, results_df):
        """Plot model comparison bar chart"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        results_df[available_metrics].plot(kind='bar', ax=ax, width=0.8, alpha=0.8)
        
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.save_dir / 'model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
        plt.show()


def train_sleep_quality_classifiers(df, save_dir='../results'):
    """
    Train and evaluate classifiers for sleep quality prediction
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        save_dir (str): Directory to save results
    
    Returns:
        HealthClassifier: Trained classifier object
    """
    print("\n" + "="*70)
    print("SLEEP QUALITY CLASSIFICATION")
    print("="*70)
    
    # Initialize classifier
    classifier = HealthClassifier(df, save_dir)
    
    # Prepare data
    features = ['Sleep_Quality', 'Stress_Level', 'Daily_Calorie_Intake', 
                'Active_Heart_Rate', 'Resting_Heart_Rate']
    target = 'Well_Rested'
    
    print(f"\nFeatures: {features}")
    print(f"Target: {target}")
    
    X_train, X_test, y_train, y_test = classifier.prepare_data(features, target)
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models
    models_to_train = [
        ('Logistic Regression', lambda: classifier.train_logistic_regression(X_train, y_train)),
        ('SVM (RBF)', lambda: classifier.train_svm(X_train, y_train, kernel='rbf')),
        ('SVM (Linear)', lambda: classifier.train_svm(X_train, y_train, kernel='linear')),
    ]
    
    for model_name, train_func in models_to_train:
        model = train_func()
        classifier.evaluate_model(model, X_test, y_test, model_name)
        classifier.plot_confusion_matrix(model, X_test, y_test, model_name)
        classifier.plot_roc_curve(model, X_test, y_test, model_name)
    
    # Compare models
    classifier.compare_models()
    
    print("\n✅ Classification complete!")
    return classifier


if __name__ == "__main__":
    from preprocessing import load_and_preprocess_data
    
    df, _ = load_and_preprocess_data("../data/smart_health_tracker_data.csv", verbose=False)
    classifier = train_sleep_quality_classifiers(df)
