

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob

from google.colab import drive

drive.mount('/content/gdrive')

IMAGE_SIZE = [224, 224]
train_data_dir = '/content/gdrive/My Drive/Cotton leafs dataset/train'
validation_data_dir = '/content/gdrive/My Drive/Cotton leafs dataset/test'
batch_size=32

train_datagen=ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=True,
    zoom_range = 0.2,
    shear_range=0.2,
    rotation_range=20,
    brightness_range=[0.2,1.0],
)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory( train_data_dir, target_size=(224, 224), 
    color_mode="rgb" , batch_size=batch_size,
    class_mode="categorical"
)

valid_generator = test_datagen.flow_from_directory( validation_data_dir, target_size=(224, 224), 
    color_mode="rgb" , batch_size=batch_size,
    class_mode="categorical"
)

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

folders = glob('/content/gdrive/My Drive/Cotton leafs dataset/train/*')

x=GlobalAveragePooling2D()(inception.output)

prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=inception.input, outputs=prediction)

model.summary()

for layer in inception.layers:
    layer.trainable = False

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('cotton_leaf_disease_model.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
callbacks_list=[checkpoint]

from tensorflow import keras
opt = keras.optimizers.Adam(lr=0.001)
model.compile(
  loss='categorical_crossentropy',
  optimizer = opt,
  metrics=['accuracy']
)

r = model.fit(
  train_generator,
  validation_data=valid_generator,
  epochs=20,
  steps_per_epoch=len(train_generator),
  validation_steps=len(valid_generator),
  callbacks=callbacks_list
)
