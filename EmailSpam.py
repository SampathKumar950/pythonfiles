import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Assuming a CSV file with 'text' and 'label' columns
data = pd.read_csv("path/to/your/dataset.csv")
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text into words
    words = text.split()
    # Remove stop words (optional)
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    # Rejoin words into a string
    text = " ".join(words)
    return text

data["text"] = data["text"].apply(preprocess_text)
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)
predictions = classifier.predict(X_test_counts)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
new_message = "Free money! Click here!"
new_message_counts = vectorizer.transform([new_message])
prediction = classifier.predict(new_message_counts)[0]
print(prediction)  
# Output: 'spam'
