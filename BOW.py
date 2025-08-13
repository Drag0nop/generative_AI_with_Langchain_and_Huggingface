from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit and transform the documents to a Bag-of-Words representation
bow_matrix = vectorizer.fit_transform(documents)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Print the Bag-of-Words matrix and feature names
print("Bag-of-Words Matrix:")
print(bow_matrix.toarray())
print("\nFeature Names:")
print(feature_names)