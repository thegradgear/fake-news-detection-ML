import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, roc_curve, auc)
import joblib

class ModelEvaluator:
    def __init__(self, model_path='models/best_model.pkl'):
        self.model = joblib.load(model_path)
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Print results
        print("Model Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # ROC Curve
        self.plot_roc_curve(y_test, y_prob)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_prob):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('models/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single(self, text, vectorizer_path='models/tfidf_vectorizer.pkl'):
        """Predict single news article"""
        from src.data_preprocessing import DataPreprocessor
        
        # Load vectorizer
        vectorizer = joblib.load(vectorizer_path)
        
        # Preprocess text
        preprocessor = DataPreprocessor()
        clean_text = preprocessor.clean_text(text)
        
        # Vectorize
        text_vector = vectorizer.transform([clean_text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        probability = self.model.predict_proba(text_vector)[0]
        
        return {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': max(probability),
            'probabilities': {
                'fake': probability[0],
                'real': probability[1]
            }
        }

if __name__ == "__main__":
    # Load test data
    X_train, X_test, y_train, y_test = joblib.load('models/processed_data.pkl')
    
    # Evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(X_test, y_test)
    
    # Test single prediction
    test_text = """
    Breaking: Scientists discover new planet that could support life. 
    The planet, located 100 light years away, shows signs of water and oxygen.
    """
    
    result = evaluator.predict_single(test_text)
    print(f"\nSample prediction:")
    print(f"Text: {test_text[:100]}...")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")