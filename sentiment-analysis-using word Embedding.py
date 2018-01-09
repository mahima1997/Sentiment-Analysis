from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import pandas as pd
import numpy as np

df = pd.read_excel('training.xlsx',header=None)
texts=df[1]
labels=df[0]
MAX_NB_WORDS=1500
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# pad documents to a max length of 25 words
MAX_SEQUENCE_LENGTH=25
#pad_sequences is for padding the sequences to mak all he sequence lengths=25
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

#the output from the Embedding layer will be 25 vectors of 20 dimensions each, one vector
#for each word in the text data instance. We flatten this to a one 32-element vector to pass on to the Dense output layer.
#The Embedding has a vocabulary of 1500 and an input length of 25. We have chosen a small embedding space of 8 dimensions.

EMBEDDING_DIM=20
embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH)

#In general, using pre-trained embeddings is relevant for natural processing tasks were little training data is available
#(functionally the embeddings act as an injection of outside information which might prove useful for the model).
#glove.6B.100d.txt is the data dump of pretrained embeddings that if used for the embedding layer give greater accuracy

# split the data into a training set and a validation set
VALIDATION_SPLIT=0.2
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]

# define the model
model = Sequential()
model.add(Embedding(len(word_index) + 1,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(x_train,y_train, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(x_test,y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))

