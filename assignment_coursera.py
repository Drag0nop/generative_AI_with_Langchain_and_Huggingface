import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------
# Example dataset (Twitter-like)
# -------------------------
tweets = [
    "I love NLP",
    "This movie is terrible",
    "I am so happy",
    "I hate this weather",
    "What a wonderful day",
    "This is the worst thing ever",
    "I really enjoyed this movie",
    "This film was awful and boring",
    "Amazing experience, I feel great",
    "I am sad and disappointed"
]
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 1 = positive, 0 = negative

# -------------------------
# 1. TF-IDF Vectorization
# -------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(tweets)

# -------------------------
# 2. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# -------------------------
# 3. Train Logistic Regression
# -------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# -------------------------
# 4. Evaluation
# -------------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# -------------------------
# 5. Confusion Matrix Visualization
# -------------------------
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------
# 6. Visualize Top Features (words)
# -------------------------
feature_names = np.array(vectorizer.get_feature_names_out())
coeffs = clf.coef_[0]

# Top positive words
top_pos_idx = np.argsort(coeffs)[-5:]
top_neg_idx = np.argsort(coeffs)[:5]

plt.figure(figsize=(10, 5))

# Positive words
plt.barh(feature_names[top_pos_idx], coeffs[top_pos_idx], color="green")
# Negative words
plt.barh(feature_names[top_neg_idx], coeffs[top_neg_idx], color="red")

plt.title("Top Predictive Words (Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.show()

# -------------------------
# 7. Test on a custom tweet
# -------------------------
test_tweet = ["I am happy with this movie"]
X_new = vectorizer.transform(test_tweet)
print("Custom Tweet Prediction:", clf.predict(X_new)[0])  # 1 = positive, 0 = negative
