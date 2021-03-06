import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq
import seaborn as sns
from pylab import rcParams
from colorama import Fore
import codeReader

should_save = not True
should_load = True
input_file_path = 'uniface-code-samples-01.txt'
model_files_title = 'trained_with_all_tests'

np.random.seed(42)
tf.random.set_seed(42)

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 5

# operations_texts = text.split('\r\noperation')
# for operation_index, operation_text in enumerate(operations_texts):
#     if operation_index != 0:
#          operations_texts[operation_index] = 'operation' + operation_text

# print('trimmed text: ', text)
tests_path = '..\\..\\..\\uniface\\qa\\urt\\tests'
code_directories = ['libprc', 'libinc', 'ent', 'cpt', 'aps']

operations_texts = codeReader.read_xml_directories(tests_path, code_directories)


# chars = sorted(list(set(text)))

char_set = set()
for opt in operations_texts:
    char_set = set.union(char_set, set(opt))

chars = sorted(list(char_set))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f'unique chars: {len(chars)}')

SENTENCES_MINIMUM_LENGTH = 1
step = 300
sentences = []
next_chars = []

SENTENCES_EQUALIZED_LENGTH = 40

for operation_index, operation_text in enumerate(operations_texts):
    for i in range(0, len(operation_text) - SENTENCES_MINIMUM_LENGTH, step):
        sentences.append(operation_text[max(0, i + SENTENCES_MINIMUM_LENGTH - SENTENCES_EQUALIZED_LENGTH):
                                        i + SENTENCES_MINIMUM_LENGTH])
        next_chars.append(operation_text[i + SENTENCES_MINIMUM_LENGTH])
print(f'num training examples: {len(sentences)}')

X = np.zeros((len(sentences), SENTENCES_EQUALIZED_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(reversed(sentence[-SENTENCES_EQUALIZED_LENGTH:])):
        X[i, SENTENCES_EQUALIZED_LENGTH-t-1, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

if should_load:
    model = load_model(model_files_title + '-model.h5')
    history = pickle.load(open(model_files_title + "-history.p", "rb"))
else:
    model = Sequential()
    model.add(LSTM(128, input_shape=(SENTENCES_EQUALIZED_LENGTH, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=20, shuffle=True).history

    if should_save:
        model.save(model_files_title + '-model.h5')
        pickle.dump(history, open(model_files_title + '-history.p', "wb"))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


def predict_completions(sentence):
    sentence_matrix = np.zeros((1, SENTENCES_EQUALIZED_LENGTH, len(chars)), dtype=np.bool)
    for t, char in enumerate(reversed(sentence[-SENTENCES_EQUALIZED_LENGTH:])):
        if char in char_indices:
            sentence_matrix[0, SENTENCES_EQUALIZED_LENGTH-t-1, char_indices[char]] = 1

    for prediction_index in range(200):
        new_prediction = model.predict(sentence_matrix)[0]
        if max(new_prediction) < 0.10:
            break
        number_of_guesses = 1
        best_predictions_indices = heapq.nlargest(number_of_guesses, range(new_prediction.size), new_prediction.take)
        best_predictions = np.array(chars)[best_predictions_indices]
        if best_predictions[0] != '\r':
            print(best_predictions[0], end="")

        new_char_matrix = np.zeros((1, 1, len(chars)), dtype=np.bool)
        new_char_matrix[0, 0, char_indices[best_predictions[0]]] = 1

        sentence_matrix = np.concatenate((sentence_matrix[0:1, 1:SENTENCES_EQUALIZED_LENGTH, :], new_char_matrix), axis=1)


while True:
    print(f"\r\n\r\n{Fore.YELLOW}Input:")
    input_sample = input()  # operations_texts[3][0:40]
    if not input_sample:
        break
    print(f"{Fore.YELLOW}Prediction: \r\n{Fore.WHITE}", input_sample, end="")
    predict_completions(input_sample)
