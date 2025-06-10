import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import os

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class DataPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.8
        )
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords and stem
        words = [word for word in text.split() if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def load_and_prepare_data(self, fake_path='data/fake.csv', real_path='data/real.csv'):
        """Load and prepare the dataset"""
        print("Loading data...")
        
        # Load datasets
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
        
        # Add labels
        fake_df['label'] = 0  # Fake news
        real_df['label'] = 1  # Real news
        
        # Combine datasets
        df = pd.concat([fake_df, real_df], ignore_index=True)
        
        print(f"Total samples: {len(df)}")
        print(f"Fake news: {len(fake_df)}")
        print(f"Real news: {len(real_df)}")
        
        # Combine title and text for better features
        df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        
        # Clean the content
        print("Cleaning text data...")
        df['cleaned_content'] = df['content'].apply(self.clean_text)
        
        # Remove empty content
        df = df[df['cleaned_content'].str.len() > 10]
        
        return df
    
    def prepare_features(self, df, fit_vectorizer=True):
        """Prepare features for model training"""
        print("Preparing features...")
        
        if fit_vectorizer:
            X = self.vectorizer.fit_transform(df['cleaned_content'])
            # Save vectorizer
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.vectorizer, 'models/tfidf_vectorizer.pkl')
        else:
            X = self.vectorizer.transform(df['cleaned_content'])
        
        y = df['label'].values
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_prepare_data()
    X, y = preprocessor.prepare_features(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Save processed data
    joblib.dump((X_train, X_test, y_train, y_test), 'models/processed_data.pkl')
    print("Data preprocessing completed!")