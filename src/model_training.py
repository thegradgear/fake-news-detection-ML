import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import time

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'svm': LinearSVC(random_state=42)
        }
        self.best_model = None
        self.best_score = 0
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and select the best one"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            training_time = time.time() - start_time
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'training_time': training_time,
                'predictions': y_pred
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, Time: {training_time:.2f}s")
            
            # Track best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for the best performing models"""
        print("\nPerforming hyperparameter tuning...")
        
        # Logistic Regression tuning
        lr_params = {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }
        
        lr_grid = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            lr_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        lr_grid.fit(X_train, y_train)
        
        # Random Forest tuning
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            rf_params,
            cv=3,  # Reduced CV for faster training
            scoring='accuracy',
            n_jobs=-1
        )
        
        rf_grid.fit(X_train, y_train)
        
        return {
            'logistic_regression': lr_grid.best_estimator_,
            'random_forest': rf_grid.best_estimator_
        }
    
    def save_model(self, model, filename='best_model.pkl'):
        """Save the trained model"""
        joblib.dump(model, f'models/{filename}')
        print(f"Model saved as {filename}")

if __name__ == "__main__":
    # Load processed data
    X_train, X_test, y_train, y_test = joblib.load('models/processed_data.pkl')
    
    trainer = ModelTrainer()
    
    # Train initial models
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Hyperparameter tuning for best models
    tuned_models = trainer.hyperparameter_tuning(X_train, y_train)
    
    # Evaluate tuned models
    print("\nEvaluating tuned models:")
    best_tuned_model = None
    best_tuned_score = 0
    
    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Tuned {name} - Accuracy: {accuracy:.4f}")
        
        if accuracy > best_tuned_score:
            best_tuned_score = accuracy
            best_tuned_model = model
    
    # Save the best model
    trainer.save_model(best_tuned_model, 'best_model.pkl')
    print(f"\nBest model accuracy: {best_tuned_score:.4f}")