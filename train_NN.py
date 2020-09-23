from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint

def create_model(vocab_size, max_length, train_X, train_Y, val_X, val_Y, filename, output_categories):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  model.add(Bidirectional(LSTM(128)))
  #model.add(LSTM(128))
  model.add(Dense(64, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(output_categories, activation = "softmax"))
  model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
  model.summary()
  checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])

