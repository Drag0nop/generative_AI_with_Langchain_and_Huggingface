import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------
# Example dataset
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

# -------------------------
# Streamlit UI
# -------------------------
st.title("Twitter Sentiment Analysis (Logistic Regression)")
st.write(f"Model Accuracy on Test Set: **{acc:.2f}**")

# ---- Confusion Matrix ----
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ---- Feature Importance ----
st.subheader("Top Predictive Words")
feature_names = np.array(vectorizer.get_feature_names_out())
coeffs = clf.coef_[0]

top_pos_idx = np.argsort(coeffs)[-5:]
top_neg_idx = np.argsort(coeffs)[:5]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feature_names[top_pos_idx], coeffs[top_pos_idx], color="green", label="Positive")
ax.barh(feature_names[top_neg_idx], coeffs[top_neg_idx], color="red", label="Negative")
ax.set_title("Top Positive and Negative Words")
ax.set_xlabel("Coefficient Value")
ax.legend()
st.pyplot(fig)

# ---- User Input ----
st.subheader("Test Your Own Tweet")
user_tweet = st.text_area("Enter a tweet here:")
if st.button("Predict Sentiment"):
    if user_tweet.strip():
        X_new = vectorizer.transform([user_tweet])
        pred = clf.predict(X_new)[0]
        sentiment = "Positive 😀" if pred == 1 else "Negative 😡"
        st.write(f"Prediction: **{sentiment}**")
    else:
        st.write("Please enter a tweet to analyze.")
