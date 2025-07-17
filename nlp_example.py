import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')

# # sentences -> paragraphs

corpus = """Hello, world! This is a test sentence.
Another sentence follows. Isn't it great? Yes, it is!"""

print(sent_tokenize(corpus))