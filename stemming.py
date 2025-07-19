"""
-Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or 
to the roots of words known as a lemma.
-Stemming is important in natural language understanding (NLU) and natural language processing (NLP).
-It is a common technique in natural language processing (NLP) to reduce words to their base or root form.
-It helps in reducing the complexity of text data and improving the performance of NLP models."""

# used for Classification Problems
# used in text preprocessing

# Porter Stemmer
from nltk.stem import PorterStemmer

words = ['running', 'ran', 'easily', 'fairly', 'happily', 'happiness', 'happier']
stemming = PorterStemmer()
for word in words:
    print(word+" ------> " + stemming.stem(word))

from nltk.stem import RegexpStemmer

reg_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)
print(reg_stemmer.stem('eating'))  # Output: 'eat'
print(reg_stemmer.stem('ingeating'))  # Output: 'ingeat'

reg_stemmer = RegexpStemmer('ing|s$|e$|able$', min=4)
print(reg_stemmer.stem('ingeating'))  # Output: 'eat'


# Snowball Stemmer (Better than Porter Stemmer)
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
for word in words:
    print(word + " ------> " + snowball_stemmer.stem(word))

print("using PorterStemmer : ",stemming.stem('fairly'))  # Output: 'fairli'
print("using SnowballStemmer : ",snowball_stemmer.stem('fairly'))  # Output: 'fair'
