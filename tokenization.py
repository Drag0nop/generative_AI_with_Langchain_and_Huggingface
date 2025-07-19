import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize

# Download necessary NLTK resources
nltk.download('punkt_tab')

corpus = """Hello, world! This is a test sentence.
Another sentence follows. Isn't it great? Yes, it is!"""

print(sent_tokenize(corpus)) #convert paragraphs to sentences
print(word_tokenize(corpus)) #convert sentences to words
print(wordpunct_tokenize(corpus)) #convert sentences to words with punctuation