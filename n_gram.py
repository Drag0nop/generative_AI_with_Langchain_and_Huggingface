# Create the Bag OF Words model with ngram
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create a CountVectorizer object with n-gram range (1, 2) for unigrams and bigrams
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))

# Fit and transform the documents
ngram_matrix = ngram_vectorizer.fit_transform(documents)

# Get the feature names (n-grams)
ngram_feature_names = ngram_vectorizer.get_feature_names_out()

# Print the N-gram matrix and feature names
print("N-gram Matrix:")
print(ngram_matrix.toarray())
print("\nN-gram Feature Names:")
print(ngram_feature_names)