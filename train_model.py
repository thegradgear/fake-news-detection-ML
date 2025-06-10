#!/usr/bin/env python3
"""
Complete training pipeline for fake news detection model
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

def main():
    print("=== Fake News Detection Model Training ===\n")
    
    # Check if data files exist
    if not os.path.exists('data/fake.csv') or not os.path.exists('data/real.csv'):
        print("Error: Data files not found!")
        print("Please download the dataset from Kaggle and place fake.csv and real.csv in the 'data' folder.")
        return
    
    # Step 1: Data Preprocessing
    print("Step 1: Data Preprocessing")
    print("-" * 30)
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_prepare_data()
    X, y = preprocessor.prepare_features(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Step 2: Model Training
    print("\nStep 2: Model Training")
    print("-" * 30)
    trainer = ModelTrainer()
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Step 3: Hyperparameter Tuning
    print("\nStep 3: Hyperparameter Tuning")
    print("-" * 30)
    tuned_models = trainer.hyperparameter_tuning(X_train, y_train)
    
    # Select best model
    best_model = None
    best_score = 0
    
    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
    
    # Save best model
    trainer.save_model(best_model, 'best_model.pkl')
    
    # Step 4: Model Evaluation
    print("\nStep 4: Model Evaluation")
    print("-" * 30)
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(X_test, y_test)
    
    print(f"\n=== Training Complete ===")
    print(f"Best Model Accuracy: {best_score:.4f}")
    print(f"Model saved to: models/best_model.pkl")
    
    # Test prediction
    print(f"\n=== Testing Prediction ===")
    test_text = "Breaking news: Local mayor announces new infrastructure project to improve city roads and public transportation."
    result = evaluator.predict_single(test_text)
    print(f"Sample text: {test_text[:80]}...")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()