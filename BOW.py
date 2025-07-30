import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load dataset
messages = pd.read_csv('file.csv', sep='\t', names=["label", "message"])
print(messages.head())

# Initialize stemmer and stopword list
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

corpus = []

for i in range(len(messages)):
    message = messages['message'].iloc[i]
    
    # Decode if bytes
    if isinstance(message, bytes):
        try:
            message = message.decode('utf-8')
        except:
            message = message.decode('latin1')

    # Ensure it's a string
    message = str(message)

    # Clean text
    review = re.sub('[^a-zA-Z]', ' ', message)     # Remove non-letters
    review = review.lower()                        # Lowercase
    review = review.split()                        # Tokenize

    # Stemming and stopword removal
    review = [ps.stem(word) for word in review if word not in stop_words]

    # Join words back into a string
    review = ' '.join(review)

    # Add to corpus
    corpus.append(review)
print(corpus) 