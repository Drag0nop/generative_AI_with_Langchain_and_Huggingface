"""
-Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or 
to the roots of words known as a lemma.
-Stemming is important in natural language understanding (NLU) and natural language processing (NLP).
-It is a common technique in natural language processing (NLP) to reduce words to their base or root form.
-It helps in reducing the complexity of text data and improving the performance of NLP models."""

from nltk.stem import RegexpStemmer

reg_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)

print(reg_stemmer.stem('eating'))  # Output: 'eat'
print(reg_stemmer.stem('ingeating'))  # Output: 'ingeat'

# Porter Stemmer
from nltk.stem import PorterStemmer