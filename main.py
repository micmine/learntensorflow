import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Use existing dataset
data = keras.datasets.fashion_mnist

# Split dataset for testing and training
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Raw data
#print(train_images[7])

# less data
train_images = train_images/255.0
test_images = test_images/255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # input layer
    keras.layers.Dense(128, activation='relu'), # hidden layer to detect patterns
    keras.layers.Dense(10, activation='softmax') # output layer get to a answer
])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", matrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5) # epoch how many times the model sees a images

#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print("Tested acc:", test_acc)

prediction = model.predict(test_images)

print(prediction[0])
# get largest one
print(np.argmax(prediction[0]))
# get Name
print(class_names[np.argmax(prediction[0])])

for i in range(6):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actoal: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()