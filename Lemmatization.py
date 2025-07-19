"""
-> Lemmatization is a technique that reduces words to their base or dictionary form, known as the lemma.
-> This technique is like stemming but it considers the context and converts the word to its meaningful base form.
-> It uses a dictionary to find the base form of a word, which is known as its lemma.
-> Unlike stemming, which may produce non-words, lemmatization ensures that the output is a valid word.
-> Lemmatization is more accurate than stemming because it takes into account the part of speech of the word.
-> For example, "better" becomes "good" and "running" becomes "run".
"""
import nltk

# Q&A, chatbots, and text summarization
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
'''
POS- Noun -n
     Verb -v
     Adjective -a
     Adverb -r
'''
print(lemmatizer.lemmatize("going"))  # returns 'going'
print(lemmatizer.lemmatize("going", pos="v"))  # returns 'go'

words = ['running', 'ran', 'easily', 'fairly', 'happily', 'happiness', 'happier']
for word in words:
    print(word + " ------> " + lemmatizer.lemmatize(word, pos='v'))  # Using verb POS
    print(word + " ------> " + lemmatizer.lemmatize(word, pos='n'))  # Using noun POS
    print(word + " ------> " + lemmatizer.lemmatize(word, pos='a'))  # Using adjective POS
    print(word + " ------> " + lemmatizer.lemmatize(word, pos='r'))  # Using adverb POS

