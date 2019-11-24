import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb 

# num_words use only word that are most relevant and used much
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=880000)

#print(test_data[0])

word_index = data.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


# switch index => key -> value
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# same size for all data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# No error form missing keys
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#print(decode_review(test_data[0]))

# Test data length
#print(len(test_data[0]), len(test_data[1]))

# start model 
model = keras.Sequential()
# to vector similar words together (good, great)
model.add(keras.layers.Embedding(880000, 16))
# shirnk down numbers => less data
model.add(keras.layers.GlobalAveragePooling1D())
# classify
model.add(keras.layers.Dense(16, activation="relu"))
# between 0 adn 1 -> positive, negative
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

# binary_crossentropy how much its away from 0 or 1
model.compile(optimiser="adam", loss="binary_crossentropy", metrics=["accuracy"])

# split test data into test_data and validation data
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# batch_size how many reviews get loadet at once
model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# test model on unseen data
results = model.evaluate(test_data, test_labels)

print(results)

model.save("text.h5")



'''
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))'''


