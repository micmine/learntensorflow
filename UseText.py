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

def review_encode(text):
    encode = [1]

    for word in text:
        # if word is known
        if word in word_index:
            encode.append(word_index[word])
        else:
            encode.append(2)
    return encode

model = keras.models.load_model("text.h5")

with open("rewiew.txt", encoding="utf-8") as f:
    for line in f.readlines():
        # remove "."  for instance "good." is not a word only good 
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        # limit to 250
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        # predict
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict)
