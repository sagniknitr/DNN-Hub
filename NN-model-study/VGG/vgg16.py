
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize



def amateur_vgg16() : 
    from tensorflow.keras import datasets, layers, models

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0


    model = models.Sequential()
    model.add(layers.Conv2D(input_shape=(32,32,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))


    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096,activation="relu"))
    model.add(layers.Dense(units=4096,activation="relu"))
    model.add(layers.Dense(units=10, activation="softmax"))
    model.summary()



    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


    history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
    
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_acc)  




if __name__ == '__main__':
    amateur_vgg16()
