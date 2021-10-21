# !kaggle competitions download -c dogs-vs-cats

import os, shutil
from keras import layers
from keras import models
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import scipy
from keras import optimizers
from IPython.display import display
import PIL
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from keras.preprocessing import image

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

train_size = 1000 
valid_size = 500 
test_size = 500
validation_steps=50
loss_function = 'binary_crossentropy'
model_type = 'VGG16'
optimizer = 'rmsprop'
met = 'acc'
rescale_rate = 1/255
target_size = 150
batch_size = 20
class_mod = 'binary'
steps_per_epoch=100
num_of_epochs=30
validation_steps=50
dropout_rate = 0
smooth = True
dense_layer = True
augmentation = False

# =============================================================================
# Accessing Data
# =============================================================================

# Download data

import requests, zipfile, io
r = requests.get('http://zhou.georgetown.domains/Data.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('Data_Origin')

original_dataset_dir = 'Data_Origin/Train'

base_dir = 'Data'

# Split data into train/valid/test

try:
    shutil.rmtree(base_dir)
except:
    pass
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(train_size)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['cat.{}.jpg'.format(i) for i in range(train_size, train_size + test_size)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['cat.{}.jpg'.format(i) for i in range(train_size + test_size, train_size + test_size + valid_size)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(train_size)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(train_size, train_size + test_size)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(train_size + test_size, train_size + test_size + valid_size)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)    

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory, target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode='binary')
    i=0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
        return features, labels

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

    
# =============================================================================
# BUILD MODEL SEQUENTIALLY (LINEAR STACK)
# =============================================================================

# Build models

if model_type == 'Linear Stack':
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.summary()
    
if model_type == 'VGG16':
    datagen = ImageDataGenerator(
          rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')
    
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(target_size, target_size, 3))

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    datagen = ImageDataGenerator(rescale=rescale_rate)
    batch_size = batch_size
        
    train_features, train_labels = extract_features(train_dir, train_size)
    validation_features, validation_labels = extract_features(validation_dir, valid_size)
    test_features, test_labels = extract_features(test_dir, test_size)

    train_features = np.reshape(train_features, (train_size, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (valid_size, 4 * 4 * 512))
    test_features = np.reshape(test_features, (test_size, 4 * 4 * 512))
    
    model = models.Sequential()
    model.add(conv_base)
    model.summary()

if dense_layer:

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()


# =============================================================================
# CONFIGURING TRAINING MODEL
# =============================================================================

model.compile(loss= loss_function,
              optimizer=optimizer,
              metrics=[met])

# =============================================================================
# PREPROCESSING
# =============================================================================

train_datagen = ImageDataGenerator(rescale=rescale_rate)
test_datagen = ImageDataGenerator(rescale=rescale_rate)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (target_size, target_size),
    batch_size = batch_size,
    class_mode = class_mod)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (target_size, target_size),
    batch_size = batch_size,
    class_mode = class_mod)

# =============================================================================
# FITTING MODEL
# =============================================================================

if augmentation == True:
    datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    
    history = model.fit_generator(
          train_generator,
          steps_per_epoch=steps_per_epoch,
          epochs=num_of_epochs,
          validation_data=validation_generator,
          validation_steps=validation_steps)
else:
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=num_of_epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)

model.save('cats_and_dogs_small_1.h5')

# =============================================================================
# VISUALIZATION
# =============================================================================

acc = history.history['acc']
val_acc = history.history['val_acc']
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
# DATA AUGMENTATION CONFIGURATION
# =============================================================================

# datagen = ImageDataGenerator(
#       rotation_range=40,
#       width_shift_range=0.2,
#       height_shift_range=0.2,
#       shear_range=0.2,
#       zoom_range=0.2,
#       horizontal_flip=True,
#       fill_mode='nearest')

# =============================================================================
# DISPLAYING RANDOM TRAINING IMAGES
# =============================================================================

# fnames = [os.path.join(train_cats_dir, fname) for
#      fname in os.listdir(train_cats_dir)]

# img_path = fnames[3]
# img = image.load_img(img_path, target_size=(150, 150))
# x = image.img_to_array(img)
# x = x.reshape((1,) + x.shape)

# i=0
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i % 4 == 0:
#         break
    
# plt.show()

# =============================================================================
# CONVNET WITH DROPOUT/AUGMENTATION (CODE WORKS BUT IT TAKES HOURS TO RUN)
# =============================================================================

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu',
#                         input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',
#               optimizer='RMSprop',
#               metrics=['acc'])

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')

# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=100,
#       epochs=100,
#       validation_data=validation_generator,
#       validation_steps=50)

# model.save('cats_and_dogs_small_2.h5')

# =============================================================================
# VGG16
# =============================================================================

# conv_base = VGG16(weights='imagenet',
#                   include_top=False,
#                   input_shape=(150, 150, 3))

# base_dir = 'Sample'
# train_dir = os.path.join(base_dir, 'train')
# validation_dir = os.path.join(base_dir, 'validation')
# test_dir = os.path.join(base_dir, 'test')
# datagen = ImageDataGenerator(rescale=1./255)
# batch_size = 20

# def extract_features(directory, sample_count):
#     features = np.zeros(shape=(sample_count, 4, 4, 512))
#     labels = np.zeros(shape=(sample_count))
#     generator = datagen.flow_from_directory(
#         directory, target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')
#     i=0
#     for inputs_batch, labels_batch in generator:
#         features_batch = conv_base.predict(inputs_batch)
#         features[i * batch_size : (i + 1) * batch_size] = features_batch
#         labels[i * batch_size : (i + 1) * batch_size] = labels_batch
#         i += 1
#         if i * batch_size >= sample_count:
#             break
#         return features, labels
    
# train_features, train_labels = extract_features(train_dir, 2000)
# validation_features, validation_labels = extract_features(validation_dir, 1000)
# test_features, test_labels = extract_features(test_dir, 1000)

# train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
# validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
# test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

# =============================================================================
# DENSELY CONNECTED CLASSIFIER
# =============================================================================

# model = models.Sequential()
# model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='RMSprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])
# history = model.fit(train_features, train_labels,
#                     epochs=30,
#                     batch_size=20,
#                     validation_data=(validation_features, validation_labels))


# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# =============================================================================
# ADDING DENSELY CONNECTED CLASSIFIER
# =============================================================================

# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.summary()

# =============================================================================
# END WITH FROZEN END
# =============================================================================

# train_datagen = ImageDataGenerator(
#       rescale=1./255,
#       rotation_range=40,
#       width_shift_range=0.2,
#       height_shift_range=0.2,
#       shear_range=0.2,
#       zoom_range=0.2,
#       horizontal_flip=True,
#       fill_mode='nearest')

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150, 150),
#     batch_size=20,
#     class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=20,
#         class_mode='binary')

# model.compile(loss='binary_crossentropy',
#               optimizer='RMSprop',
#               metrics=['acc'])

# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=100,
#       epochs=30,
#       validation_data=validation_generator,
#       validation_steps=50)

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# =============================================================================
# FREEZING ALL LAYERS UP
# =============================================================================

# conv_base.trainable = True
# set_trainable = False

# for layer in conv_base.layers:
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False



# test_generator = test_datagen.flow_from_directory(
#         test_dir,
#         target_size=(150, 150),
#         batch_size=20,
#         class_mode='binary')
# test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
# print('test acc:', test_acc)

