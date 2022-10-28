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
        layers.Reshape((784,), input_shape=((28, 28, 1))),
    ]
)

# label_augmentation = keras.Sequential(
#     [

#     ]
# )

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), data_augmentation(x)))
val_ds = val_ds.map(lambda x, y: (data_augmentation(x), data_augmentation(x)))

# plt.figure(figsize=(10, 10))
# for images, labels in augmented_train_ds.take(1):
#     for i in range(0, 8, 2):
#         ax = plt.subplot(2, 4, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.axis("off")
#         ax = plt.subplot(2, 4, i + 1 + 1)
#         plt.imshow(labels[i].numpy().astype("uint8"))
#         plt.axis("off")

# plt.savefig('wow.png')
# This is the size of our encoded representations
encoding_dim = 36  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# This is our input image
input_img = keras.Input(shape=(784,))
# input_img = keras.Input(shape=(180, 180, 1))
# resized = layers.Resizing(28, 28)(input_img)
# rescaled = layers.Rescaling(1./255)(resized)
# reshaped = layers.Reshape((784,), input_shape=((28, 28, 1)))(rescaled)
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(784, activation='sigmoid')(encoded)
# reconstructed = layers.Reshape((28, 28, 1), input_shape=((784,)))(decoded)
# autoencoder_rescaled = layers.Rescaling(255)(reconstructed)
# maximized = layers.Resizing(180, 180)(autoencoder_rescaled)
# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

###

# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)

###

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

###

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

###

autoencoder.fit(train_ds,
                epochs=50,
                batch_size=32,
                shuffle=True,
                validation_data=val_ds)

###

encoded_imgs = encoder.predict(val_ds)
decoded_imgs = decoder.predict(encoded_imgs)

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
