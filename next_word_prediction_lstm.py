# https://mc.ai/build-a-simple-predictive-keyboard-using-python-and-keras/

import nltk
from keras.models import Sequential
from keras.layers import LSTM,Dense
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from nltk.tokenize import RegexpTokenizer

# model = Sequential()
# model.add(LSTM(300,return_sequences=True,input_shape=(1,)))


os.chdir(r"F:\python codes\interview_codes\rnn")
filepath = "data.txt"
file = open(filepath,encoding="utf8")

text = file.read()

text = text.lower()
text = text.replace("œ","")
text = text.replace("â","")
text = text.replace("å","")

#print(len(text))
# 594202

words = text.split(' ')
#print(len(words))

# Now, we want to split the entire dataset into each word
# in order without the presence of special characters.
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

import string
table = str.maketrans('', '', string.punctuation)
words = [w.translate(table) for w in words]

#print(words[0:100])

unique_words = np.unique(words)

#print(len(unique_words))

dict = {}

value = 0
for word in unique_words:
    dict[word] = value
    value = value + 1

print(dict)


# We define a WORD_LENGTH which means that the number of previous words
# that determines the next word.
# Also, we create an empty list called prev_words
# to store a set of five previous words
# and its corresponding next word in the next_words list.
# We fill these lists by looping over a range of 5 less than the length of words.

WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
 prev_words.append(words[i:i + WORD_LENGTH])
 next_words.append(words[i + WORD_LENGTH])
# print(prev_words[0:50])
# print(next_words[0:50])

#print(len(prev_words))

# ['to', 'sherlock', 'holmes', 'she', 'is']
# always

# Now, its time to generate feature vectors.
# For generating feature vector we use one-hot encoding.

# Explanation: one-hot encoding
# Here, we create two numpy array X(for storing the features)
# and Y(for storing the corresponding label(here, next word)).
#  We iterate X and Y if the word is present then the corresponding position is made 1.

X = np.zeros((len(prev_words),WORD_LENGTH,len(unique_words)),dtype=int)
y = np.zeros((len(next_words),len(unique_words)),dtype=int)
print(X,X.ndim)
print(y,y.ndim)

for i in range(0,len(prev_words)):
    for j in range(0, WORD_LENGTH):
        X[i][j][dict[prev_words[i][j]]] = 1

print('X',X)

for i in range(0,len(next_words)):
    for j in range(0, len(unique_words)):
        y[i][dict[next_words[i]]] = 1


print('y',y)

def createModel():
    # Building the model
    # We use a single-layer LSTM model with 128 neurons,
    #  a fully connected layer, and a softmax function for activation.

    model = Sequential()
    # X has shape (len(prev_words),WORD_LENGTH,len(unique_words))
    # so, sample shape(shape[0] will not go here
    # By default, return_sequences=False.
    # If we want to add more LSTM layers,
    # then the last LSTM layer must add return_sequences=True
    model.add(LSTM(128, input_shape=(WORD_LENGTH, len(unique_words))))

    # output class number, here it will be the unique words number
    model.add(Dense(len(unique_words), activation='softmax'))

    from keras.optimizers import RMSprop
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True)

    # save this model
    model.save('keras_next_word_model_omi.h5')
    return model

def load_saved_model():
    from keras.models import load_model
    model = load_model('keras_next_word_model_omi.h5')
    return model


#model = createModel()
model = load_saved_model()
# now input a 5 words text and show the next possible words

input_text = input("input text of 5 words")
input_text = input_text.lower()

input_text = nltk.word_tokenize(input_text)
print(input_text)

input_text_list = [input_text]

X_test = np.zeros((len(input_text_list),WORD_LENGTH,len(unique_words)),dtype=int)
print(X,X.ndim)


for i in range(0,len(input_text_list)):
    for j in range(0, WORD_LENGTH):
        X_test[i][j][dict[input_text_list[i][j]]] = 1

print('X_test',X_test)

# X_test [[[0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]]]

# predict next words

def predictWords(X_test,num_words=3):

    y_pred = model.predict(X_test, verbose=0)[0]


    indices = np.argsort(y_pred)
    indices = indices[::-1]

    print('Predicted words: ')
    for i in range(0,len(y_pred)):
        print(unique_words[indices[i]],'---probability: ',y_pred[indices[i]])



predictWords(X_test)

