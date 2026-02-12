"""
Deep Learning Models Module
PyTorch and TensorFlow/Keras implementations for health prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# TensorFlow/Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
)


class PyTorchPerceptron(nn.Module):
    """Simple Perceptron model in PyTorch"""
    
    def __init__(self, input_size):
        super(PyTorchPerceptron, self).__init__()
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class PyTorchDeepNN(nn.Module):
    """Deep Neural Network in PyTorch"""
    
    def __init__(self, input_size, hidden_sizes=[32, 16]):
        super(PyTorchDeepNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


class HealthDeepLearning:
    """
    Deep Learning models for health predictions
    """
    
    def __init__(self, df, save_dir='../results'):
        """
        Initialize deep learning trainer
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            save_dir (str): Directory to save results
        """
        self.df = df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
        
    def prepare_data(self, features, target, test_size=0.2):
        """Prepare and scale data"""
        X = self.df[features].values
        y = self.df[target].values
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.scaler = scaler
        return X_train, X_test, y_train, y_test
    
    def train_pytorch_perceptron(self, X_train, y_train, X_test, y_test, 
                                 epochs=100, lr=0.01, verbose=True):
        """
        Train PyTorch Perceptron
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            epochs (int): Number of training epochs
            lr (float): Learning rate
            verbose (bool): Print training progress
        
        Returns:
            model: Trained PyTorch model
        """
        print("\n🔄 Training PyTorch Perceptron...")
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        # Initialize model
        input_size = X_train.shape[1]
        model = PyTorchPerceptron(input_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{epochs}, Loss: {loss.item():.4f}")
        
        print("✓ PyTorch Perceptron trained")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X_test_t).numpy()
            y_pred = (y_pred_prob >= 0.5).astype(int)
        
        self.models['PyTorch Perceptron'] = model
        self._evaluate_and_save('PyTorch Perceptron', y_test, y_pred, y_pred_prob)
        
        return model
    
    def train_pytorch_deep_nn(self, X_train, y_train, X_test, y_test,
                              hidden_sizes=[32, 16], epochs=100, lr=0.01, verbose=True):
        """
        Train PyTorch Deep Neural Network
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            hidden_sizes (list): Hidden layer sizes
            epochs (int): Number of training epochs
            lr (float): Learning rate
            verbose (bool): Print training progress
        
        Returns:
            model: Trained PyTorch model
        """
        print(f"\n🔄 Training PyTorch Deep NN (architecture: {hidden_sizes})...")
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        # Initialize model
        input_size = X_train.shape[1]
        model = PyTorchDeepNN(input_size, hidden_sizes)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{epochs}, Loss: {loss.item():.4f}")
        
        print("✓ PyTorch Deep NN trained")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X_test_t).numpy()
            y_pred = (y_pred_prob >= 0.5).astype(int)
        
        self.models['PyTorch Deep NN'] = model
        self._evaluate_and_save('PyTorch Deep NN', y_test, y_pred, y_pred_prob)
        
        return model
    
    def train_keras_model(self, X_train, y_train, X_test, y_test,
                         hidden_sizes=[32, 16], epochs=100, lr=0.01, verbose=0):
        """
        Train Keras Sequential model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            hidden_sizes (list): Hidden layer sizes
            epochs (int): Number of training epochs
            lr (float): Learning rate
            verbose (int): Keras verbosity level
        
        Returns:
            model: Trained Keras model
        """
        print(f"\n🔄 Training Keras Deep NN (architecture: {hidden_sizes})...")
        
        # Build model
        input_size = X_train.shape[1]
        model = Sequential([
            Dense(hidden_sizes[0], input_shape=(input_size,), activation='relu'),
            Dense(hidden_sizes[1], activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=lr), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        # Train
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                          verbose=verbose, validation_split=0.1)
        
        print("✓ Keras Deep NN trained")
        
        # Evaluate
        y_pred_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        self.models['Keras Deep NN'] = model
        self._evaluate_and_save('Keras Deep NN', y_test, y_pred, y_pred_prob)
        
        return model
    
    def _evaluate_and_save(self, model_name, y_test, y_pred, y_prob):
        """Evaluate model and save results"""
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        
        self.results[model_name] = metrics
        
        print(f"\n📊 {model_name} Results:")
        print("-" * 50)
        for metric, value in metrics.items():
            if metric != 'Model':
                print(f"{metric:15s}: {value:.4f}")
    
    def plot_confusion_matrix(self, model_name, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                      display_labels=['Not Well Rested', 'Well Rested'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
        plt.show()
    
    def plot_roc_curve(self, model_name, y_test, y_prob):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
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
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.results:
            print("⚠ No models evaluated yet")
            return
        
        print("\n" + "="*70)
        print("DEEP LEARNING MODEL COMPARISON")
        print("="*70)
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.drop('Model', axis=1, errors='ignore')
        print(results_df.to_string())
        
        save_path = self.save_dir / 'deep_learning_comparison.csv'
        results_df.to_csv(save_path)
        print(f"\n✓ Saved to {save_path}")
        
        return results_df


def train_deep_learning_models(df, save_dir='../results'):
    """
    Train all deep learning models
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        save_dir (str): Directory to save results
    
    Returns:
        HealthDeepLearning: Trainer object
    """
    print("\n" + "="*70)
    print("DEEP LEARNING MODELS")
    print("="*70)
    
    trainer = HealthDeepLearning(df, save_dir)
    
    # Prepare data
    features = ['Sleep_Quality', 'Stress_Level', 'Daily_Calorie_Intake',
                'Active_Heart_Rate', 'Resting_Heart_Rate']
    target = 'Well_Rested'
    
    X_train, X_test, y_train, y_test = trainer.prepare_data(features, target)
    
    # Train models
    trainer.train_pytorch_perceptron(X_train, y_train, X_test, y_test)
    trainer.train_pytorch_deep_nn(X_train, y_train, X_test, y_test)
    trainer.train_keras_model(X_train, y_train, X_test, y_test)
    
    # Compare
    trainer.compare_models()
    
    print("\n✅ Deep learning training complete!")
    return trainer


if __name__ == "__main__":
    from preprocessing import load_and_preprocess_data
    
    df, _ = load_and_preprocess_data("../data/smart_health_tracker_data.csv", verbose=False)
    trainer = train_deep_learning_models(df)
