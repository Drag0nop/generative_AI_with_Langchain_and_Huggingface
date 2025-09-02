from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents (using the same as before)
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create a TfidfVectorizer object
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents to a TF-IDF representation
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Get the feature names (words/terms)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Print the TF-IDF matrix and feature names
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("\nTF-IDF Feature Names:")
print(tfidf_feature_names)