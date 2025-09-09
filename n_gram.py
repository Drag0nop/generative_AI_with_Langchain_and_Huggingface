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
n = CountVectorizer(ngram_range=(1, 2))

# Fit and transform the documents
ngram_matrix = n.fit_transform(documents)

# Get the feature names (n-grams)
n_feature = n.get_feature_names_out()

# Print the N-gram matrix and feature names
print("N-gram Matrix:")
print(ngram_matrix.toarray())
print("\nN-gram Feature Names:")
print(n_feature)
import pandas as pd

# Create a pandas DataFrame
ngram_df_preprocessed = pd.DataFrame(ngram_matrix_preprocessed.toarray(), columns=ngram_feature_names_preprocessed)

# Display the DataFrame
display(ngram_df_preprocessed)