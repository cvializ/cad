import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import os
os.environ['PYTHONINSPECT'] = 'TRUE'

image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    shuffle=False,
    color_mode="grayscale",
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    shuffle=False,
    color_mode="grayscale",
    image_size=image_size,
    batch_size=batch_size,
)

data_augmentation = keras.Sequential(
    [
        layers.Resizing(28, 28),
        layers.Rescaling(1./255),
        # layers.Reshape((784,), input_shape=((28, 28, 1))),
    ]
)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), data_augmentation(x)))
val_ds = val_ds.map(lambda x, y: (data_augmentation(x), data_augmentation(x)))

input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

###

autoencoder.fit(train_ds,
                epochs=1,
                batch_size=32,
                shuffle=True,
                validation_data=val_ds)

###

decoded_imgs = autoencoder.predict(val_ds)

###

plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
    for i in range(0, 8, 2):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow((images[i].numpy() * 255).reshape(28, 28).astype("uint8"))
        plt.axis("off")
        ax = plt.subplot(2, 4, i + 1 + 1)
        plt.imshow((decoded_imgs[i] * 255).reshape(28, 28).astype("uint8"))
        plt.axis("off")
