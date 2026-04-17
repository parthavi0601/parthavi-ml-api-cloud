# Step 1: Install dependencies (run once in terminal)
# pip install pandas scikit-learn joblib

# Step 2: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Step 3: Load dataset
# Make sure the CSV file is in the same folder OR give full path
df = pd.read_csv("Train.csv")

print("Dataset loaded successfully!")
print(df.head())

# Step 4: Prepare data
texts = df["text"]
labels = df["label"]  # already 'positive' or 'negative'

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Step 6: Create model pipeline
model = make_pipeline(
    TfidfVectorizer(
        stop_words="english",
        max_features=10000
    ),
    LogisticRegression(max_iter=200)
)

# Step 7: Train model
print("\nTraining model...")
model.fit(X_train, y_train)

# Step 8: Evaluate model
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Step 9: Save model
joblib.dump(model, "sentiment_model.joblib")
print("\nModel saved as sentiment_model.joblib")

# Step 10: Test predictions
sample_texts = [
    "This movie was absolutely amazing!",
    "Worst film I have ever seen.",
    "It was okay, not great but not bad either."
]

predictions = model.predict(sample_texts)

print("\nSample Predictions:")
for text, pred in zip(sample_texts, predictions):
    print(f"\nText: {text}")
    print(f"Predicted Sentiment: {pred}")