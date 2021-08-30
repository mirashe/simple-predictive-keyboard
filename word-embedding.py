import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

# Loading the dataset
path = '1661.txt'
text = open(path, encoding="utf8").read().lower().replace('\n', ' ')

# import nltk
# sentences = nltk.tokenize.sent_tokenize(text)

# print('corpus length:', len(text))

# Split the entire dataset into each word
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

# Create a word - position dictionary
unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

# Preparing word lists
WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])

# print(prev_words[0])
# print(next_words[0])

from keras.preprocessing.text import Tokenizer as PreprocessingTokenizer
from sklearn.model_selection import train_test_split
prev_words_train, prev_words_test, next_words_train, next_words_test = train_test_split(prev_words, next_words, test_size=0.25, random_state=1000)
preprocessingTokenizer = PreprocessingTokenizer(num_words=5000, oov_token=0)
preprocessingTokenizer.fit_on_texts(prev_words)
prev_words_embedding_train = preprocessingTokenizer.texts_to_sequences(prev_words_train)
prev_words_embedding_test = preprocessingTokenizer.texts_to_sequences(prev_words_test)
next_words_embedding_train = preprocessingTokenizer.texts_to_sequences([next_words_train])[0]
next_words_embedding_test = preprocessingTokenizer.texts_to_sequences([next_words_test])[0]
vocab_size = len(preprocessingTokenizer.word_index)



# Generating feature vectors (by using one-hot encoding)
# X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
next_words_one_hot_train = np.zeros((len(next_words_train), len(unique_words)), dtype=bool)
next_words_one_hot_test = np.zeros((len(next_words_test), len(unique_words)), dtype=bool)

for i, each_word in enumerate(next_words_one_hot_train):
    next_words_one_hot_train[i, unique_word_index[next_words_train[i]]] = 1
for i, each_word in enumerate(next_words_one_hot_test):
    next_words_one_hot_test[i, unique_word_index[next_words_test[i]]] = 1

# print(X[0][0])

# Building the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                           output_dim=50,
                           # weights=[embedding_matrix],  # If we want to use pretrained data
                           input_length=WORD_LENGTH
                           # , trainable=True
                           ))
model.add(LSTM(128))
model.add(Dense(len(unique_words)))
# model.add(Activation('softmax'))
model.add(Dense(1, activation='sigmoid'))

# Training
optimizer = RMSprop(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(  prev_words_embedding_train
                    , next_words_embedding_train
                    , validation_data=(prev_words_embedding_test, next_words_embedding_test)
                    , batch_size=128
                    , epochs=4
                    , verbose=False
                    ).history

# Saving the trained model
model.save('word-embedding-model.h5')
pickle.dump(history, open("word-embedding-history2.p", "wb"))
# model = load_model('word-one-hot-model.h5')
# history = pickle.load(open("word-one-hot-history2.p", "rb"))


def plot_history():
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


plot_history()


# Prediction
def prepare_input(itext):
    x = np.zeros((1, WORD_LENGTH, len(unique_words)))
    for t, word in enumerate(itext.split()):
        print(word)
        x[0, t, unique_word_index[word]] = 1
    return x


# prepare_input("It is not a lack".lower())

# Best possibles
def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_completions(text, n=3):
    if text == "":
        return "0"
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] for idx in next_indices]


q = "Your life will never be the same again"
print("correct sentence: ", q)
seq = " ".join(tokenizer.tokenize(q.lower())[0:5])
print("Sequence: ",seq)
print("next possible words: ", predict_completions(seq, 5))
