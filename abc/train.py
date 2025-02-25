import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
data = pd.read_csv("sentiment_data.csv")

# Split (very small dataset, so be careful!)
X = data['text']
y = data['sentiment']

# Feature extraction (convert text to numbers)
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train (basic model)
model = LogisticRegression()
model.fit(X_vec, y)

# Save the model and vectorizer
joblib.dump((model, vectorizer), 'sentiment_model.pkl')

print("Model trained and saved!")
