from keras import layers 
from keras import models
import numpy as np
import warnings
from tensorflow.keras.utils import to_categorical
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import random
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

NKEEP = 60000
batch_size = int(0.1 * NKEEP)
epochs = 30
dataset = 'fashion'
num_of_viz = 10
loss_function = 'binary_crossentropy'
optimizer = 'adam'
met = 'accuracy'
model = 'ANN'
num_of_class = 10
valid_percent = 0.2
augmentation = False
smooth = True

# =============================================================================
# 
# =============================================================================

# NKEEP = 60000
# batch_size = int(0.1 * NKEEP)
# epochs = 5
# dataset = 'mnist'
# num_of_viz = 10
# loss_function = 'binary_crossentropy'
# optimizer = 'rmsprop'
# met = 'accuracy'
# model = 'ANN'
# num_of_class = 10
# valid_percent = 0.2
# augmentation = False

# =============================================================================
# BUILD MODEL SEQUENTIALLY (LINEAR STACK)
# =============================================================================

# A flag to change the number of channels for different datasets

if dataset == 'cifar':
    num_of_ch = 3
    num_of_input = 32
else:
    num_of_ch = 1
    num_of_input = 28

# A flag to build the nnets models depending on the type desired

if model == 'CNN':
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(num_of_input, num_of_input, num_of_ch)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_of_class, activation='softmax'))

    model.summary()

if model == 'ANN':
    
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=[num_of_input,num_of_input,num_of_ch]))
    model.add(layers.Dense(512, activation='relu'))
    model.add(Dense(num_of_class, activation='softmax'))
    
    model.summary()

# =============================================================================
# GET DATA AND REFORMAT
# =============================================================================

# Read in the data

if dataset == 'mnist':

    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
elif dataset == 'fashion':
    
    from keras.datasets import fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
elif dataset == 'cifar':
    
    from keras.datasets import cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
train_images = train_images.reshape((train_images.shape[0], num_of_input, num_of_input, num_of_ch))
test_images = test_images.reshape((test_images.shape[0], num_of_input, num_of_input, num_of_ch))

# Split the trainset into train and valid

if valid_percent != 0:
    valid_ind = int(train_images.shape[0] * valid_percent)
    
    valid_images = train_images[0:valid_ind,:,:]
    train_images = train_images[valid_ind:int(train_images.shape[0]),:,:]
    
    valid_labels = train_labels[0:valid_ind]
    train_labels = train_labels[valid_ind:int(train_labels.shape[0])]
    
# Normalize the data

train_images = train_images.astype('float32') / 255 
test_images = test_images.astype('float32') / 255
valid_images = valid_images.astype('float32') / 255  


print("batch_size",batch_size)
rand_indices = np.random.permutation(train_images.shape[0])
train_images=train_images[rand_indices[0:NKEEP],:,:]
train_labels=train_labels[rand_indices[0:NKEEP]]

# CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX

tmp=train_labels[0]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
valid_labels = to_categorical(valid_labels)
print(tmp, '-->', train_labels[0])
print("train_labels shape:", train_labels.shape)


# Visulize Random Train Data

for i in range(num_of_viz):
    plt.imshow(train_images[random.randint(0, train_images.shape[0])])
    plt.show()
    
    
        
# =============================================================================
# COMPILE AND TRAIN MODEL
# =============================================================================

# Compile the model based on the parameteres given

model.compile(loss = loss_function,
              optimizer = optimizer,
              metrics = [met])

# Configurate augmentation

if augmentation == True:
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
      
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(
        train_images,
        train_labels,
        batch_size = batch_size)
    
    validation_generator = test_datagen.flow(
        valid_images,
        valid_labels,
        batch_size = batch_size)
    
    history = model.fit_generator(
        train_generator, steps_per_epoch=10,
        epochs=10,
        validation_data = validation_generator, validation_steps=1, verbose = 0)

# Fit model without augmentation
    
else:    
    history = model.fit(train_images, train_labels, validation_data = (valid_images, valid_labels), epochs=epochs, batch_size=batch_size)
    
model.save('keras.h5')


# =============================================================================
# EVALUATE ON TEST DATA
# =============================================================================

train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

if smooth == True:
    def smooth_curve(points, factor=0.8):
      smoothed_points = []
      for point in points:
        if smoothed_points:
          previous = smoothed_points[-1]
          smoothed_points.append(previous * factor + point * (1 - factor))
        else:
          smoothed_points.append(point)
      return smoothed_points

    plt.plot(epochs,
             smooth_curve(acc), 'bo', label='Smoothed training acc')
    plt.plot(epochs,
             smooth_curve(val_acc), 'b', label='Smoothed validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs,
             smooth_curve(loss), 'bo', label='Smoothed training loss')
    plt.plot(epochs,
             smooth_curve(val_loss), 'b', label='Smoothed validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

# =============================================================================
# VIZ Layer
# =============================================================================

def viz_layer_CNN(num_of_image, num_of_cha):
    img_tensor = np.expand_dims(train_images[num_of_image,:,:], axis=0)
    img_tensor /= 255.
    plt.imshow(tf.squeeze(img_tensor))
    plt.show()
    layer_outputs = [layer.output for layer in model.layers[:len(model.layers)]] 
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    first_layer_activation = activations[0]
    plt.matshow(first_layer_activation[0, :, :, num_of_cha], cmap='viridis')
    plt.show()
    
def viz_layer_ANN(num_of_image, num_of_cha):
    img_tensor = np.expand_dims(train_images[num_of_image,:,:], axis=0)
    img_tensor /= 255.
    plt.imshow(tf.squeeze(img_tensor))
    plt.show()
    layer_outputs = [layer.output for layer in model.layers[:len(model.layers)]] 
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    first_layer_activation = activations[0]
    plt.matshow(first_layer_activation, cmap='viridis')
    plt.show()

if model == 'CNN':
    for i in range(num_of_viz):
        viz_layer_CNN(random.randint(0, train_images.shape[0]), random.randint(0, len(model.layers)))

if model == 'ANN':
    for i in range(num_of_viz):
        viz_layer_ANN(random.randint(0, train_images.shape[0]), random.randint(0, len(model.layers)))

    
    
   

    
    
    