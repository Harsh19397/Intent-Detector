import numpy as np
import re
from keras.models import load_model
import data_preprocessing_utilities as dpu
import logging
import train_NN
import os


logging.basicConfig(level=logging.DEBUG)

#Loading the dataset
#Change the path for your dataset
intent, unique_intent, sentences = dpu.load_dataset('C:\Machine Learning\Projects\Intent_detection\Dataset.csv')
logging.debug(sentences[:5])



#Cleaning the words
cleaned_words = dpu.cleaning(sentences)
logging.debug("The length of the cleaned words is: "+str(len(cleaned_words)))
logging.debug(cleaned_words[:2])

#Creating the Tokens
#calculating vocab_size and maximum length of the cleaned words
word_tokenizer = dpu.create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = dpu.max_length(cleaned_words)

logging.debug("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))

#Encoding text to sequences
encoded_doc = dpu.encoding_doc(word_tokenizer, cleaned_words)

#Padding the sequences encoded above
padded_doc = dpu.padding_doc(encoded_doc, max_length)
logging.debug(padded_doc[:5])
logging.debug("Shape of padded docs = ",padded_doc.shape)

#Tokenizer with filter changed
output_tokenizer = dpu.create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
output_tokenizer.word_index
encoded_output = dpu.encoding_doc(output_tokenizer, intent)
encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)

logging.debug(encoded_output.shape)

#One Hot Encoding
output_one_hot = dpu.one_hot(encoded_output)
logging.debug(output_one_hot.shape)

#Splitting the dataset into training and test dataset
train_X, val_X, train_Y, val_Y = dpu.train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)

logging.debug("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
logging.debug("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))

#Running the RNN
filename = 'IntentDetectormodel.h5'

if os.path.isfile('IntentDetectormodel.h5') == False:
    train_NN.create_model(vocab_size, max_length, train_X, train_Y, val_X, val_Y, filename, len(unique_intent))

#Loading the model
model = load_model('IntentDetectormodel.h5')

def predictions(text):
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = dpu.word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  test_ls = word_tokenizer.texts_to_sequences(test_word)
  print(test_word)
  #Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))

  test_ls = np.array(test_ls).reshape(1, len(test_ls))

  x = dpu.padding_doc(test_ls, max_length)
  model = load_model('IntentDetectormodel.h5')
  pred = model.predict_proba(x)


  return pred


def get_final_output(pred, classes):
  predictions = pred[0]

  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
  return classes[np.argmax(predictions)]

#Testing
def get_intent(text):
    #text = "Please search about Jarvis on google"
    pred = predictions(text)
    print("Intent Detected: "+ get_final_output(pred, unique_intent))
    return get_final_output(pred, unique_intent)









