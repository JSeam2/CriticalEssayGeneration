import numpy as np

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.utils import np_utils

import random
import sys
import io
import os

# load text file
current_dir = os.path.dirname(__file__)
with io.open(os.path.join(current_dir, "Data", "PhilosophyOfMind.txt"),
             encoding='utf-8') as f:
    text = f.read().lower()

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars: ', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redudant sequences of max length char ??
maxlen = 100
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print('nb sequences: ', len(sentences))

print('Vectorization...')

x = np.zeros((len(sentences), maxlen, len(chars)), dtype= np.bool)
y = np.zeros((len(sentences), len(chars)), dtype= np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build model using single lstm
print("Build model...")
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr= 0.01)
model.compile(loss='categorical_crossentropy', optimizer = optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index form a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) /  temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/ np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. prints generated text
    print()
    print('----- Generating text after Epoch: {}'.format(epoch))

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity: ', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))

            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

        print()


print_callback = LambdaCallback(on_epoch_end = on_epoch_end)

model.fit(x, y,
          batch_size = 1,
          epochs = 60,
          callbacks= [print_callback])

# serialize model
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model")

## Using saved model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
#
## evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
#                     metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

