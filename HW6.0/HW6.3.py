import keras
from keras import layers
import matplotlib.pyplot as plt

from keras.datasets import mnist,cifar10, cifar100
import numpy as np

from keras import losses
from keras.callbacks import CSVLogger

import pandas as pd

#USER PARAM
INJECT_NOISE    =   False
EPOCHS          =   35
NKEEP           =   2500        #DOWNSIZE DATASET
BATCH_SIZE      =   128
DATA            =   "CIFAR"

#GET DATA
if(DATA=="MNIST"):
    (x_train, _), (x_test, _) = mnist.load_data()
    N_channels=1; PIX=28

if(DATA=="CIFAR"):
    (x_train, _), (x_test, _) = cifar10.load_data()
    N_channels=3; PIX=32
    EPOCHS=100 #OVERWRITE

#NORMALIZE AND RESHAPE
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#DOWNSIZE TO RUN FASTER AND DEBUG
print("BEFORE",x_train.shape)
x_train=x_train[0:NKEEP]
x_test=x_test[0:NKEEP]
print("AFTER",x_train.shape)

#ADD NOISE IF DENOISING
if(INJECT_NOISE):
    EPOCHS=2*EPOCHS
    #GENERATE NOISE
    noise_factor = 0.5
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_train=x_train+noise
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    x_test=x_test+noise

    #CLIP ANY PIXELS OUTSIDE 0-1 RANGE
    x_train = np.clip(x_train, 0., 1.)
    x_test = np.clip(x_test, 0., 1.)

#BUILD CNN-AE MODEL


if(DATA=="MNIST"):
    input_img = keras.Input(shape=(PIX, PIX, N_channels))

    # #ENCODER
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    # # AT THIS POINT THE REPRESENTATION IS (4, 4, 8) I.E. 128-DIMENSIONAL
 
    # #DECODER
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)


if(DATA=="CIFAR"):
    input_img = keras.Input(shape=(PIX, PIX, N_channels))

    #ENCODER
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    #DECODER
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)



#COMPILE
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy');
autoencoder.summary()

log = CSVLogger('HW6.3_log.txt', append=True, separator=';') 

#TRAIN
history = autoencoder.fit(x_train, x_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks = log
                )

#HISTORY PLOT
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()
plt.savefig('HW6.3_History.png') 
plt.show()
plt.close()

#MAKE PREDICTIONS FOR TEST DATA
decoded_imgs = autoencoder.predict(x_test)

# threshold for anomaly scores
threshold = 4*autoencoder.evaluate(x_test,x_test, batch_size = x_test.shape[0])
threshold

# =============================================================================
# Anomaly Detection
# =============================================================================

(x_train, _), (x_test, _) = cifar100.load_data()
N_channels=3; PIX=32

X = x_train.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.

#PLOT ORIGINAL AND RECONSTRUCTED 
X1=autoencoder.predict(X)

X = X.reshape(50000, 32*32*3); 
X1 = X1.reshape(50000 ,32*32*3); 

reconstruction_errors = losses.binary_crossentropy(X, X1)

anomaly = pd.Series(reconstruction_errors) > threshold

anom_percentage = len(X[anomaly])/len(X)

with open('HW6.3_log.txt', 'a') as f:
    f.write('Anomaly percentage = ')
    f.write(str(anom_percentage))
    
#RESHAPE
X=X.reshape(50000,32,32,3); #print(X[0])
X1=X1.reshape(50000,32,32,3); #print(X[0])    
    
#COMPARE ORIGINAL 
f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(X[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(X[I2])
ax[3].imshow(X1[I2])
plt.savefig('HW6.3_Reconstruct.png') 
plt.show()
plt.close()

# # #VISUALIZE THE RESULTS
# # n = 10
# # plt.figure(figsize=(20, 4))
# # for i in range(1, n + 1):
# #     # Display original
# #     ax = plt.subplot(2, n, i)
# #     plt.imshow(x_test[i].reshape(PIX, PIX,N_channels))
# #     plt.gray()
# #     ax.get_xaxis().set_visible(False)
# #     ax.get_yaxis().set_visible(False)

# #     # Display reconstruction
# #     ax = plt.subplot(2, n, i + n)
# #     plt.imshow(decoded_imgs[i].reshape(PIX, PIX,N_channels))
# #     plt.gray()
# #     ax.get_xaxis().set_visible(False)
# #     ax.get_yaxis().set_visible(False)
# # plt.show()




