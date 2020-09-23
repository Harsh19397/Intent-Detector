#data preprocessing
#importing libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import re
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#Loading the dataset
def load_dataset(filename):
  df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
  print(df.head())
  intent = df["Intent"]
  unique_intent = list(set(intent))
  sentences = list(df["Sentence"])

  return (intent, unique_intent, sentences)

#Cleaning the phrases with all kind of punctuation marks
def cleaning(sentences):
  words = []
  stemmer = LancasterStemmer()
  for s in sentences:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
    w = word_tokenize(clean)
    #stemming
    [stemmer.stem(x) for x in s.split()]
    words.append([i.lower() for i in w])

  return words

#Tokenizing the phrases and filtering out the special characters
def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token

#Getting the length of the longest word from the dataset
def max_length(words):
  return(len(max(words, key = len)))

#Encoding the tokens generated earlier
def encoding_doc(token, words):
  return(token.texts_to_sequences(words))

#Padding the encoded token with the length of the longest token
def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

#One hot encoding the intents
def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))

#Splitting the dataset into train and test data
def dataset_split(X, y, test_size=0.2):
    return train_test_split(X, y, shuffle = True, test_size = 0.2)

