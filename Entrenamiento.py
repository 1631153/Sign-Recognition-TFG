import tensorflow as tf

from tensorflow.keras.layers                import *
from tensorflow.keras.preprocessing.image   import ImageDataGenerator
from tensorflow.keras.optimizers            import SGD
from tensorflow.keras                       import regularizers
from tensorflow.keras.callbacks             import ModelCheckpoint
from tensorflow.keras.callbacks             import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image   import ImageDataGenerator

bs = 64 #bach size
k = 2

# Generador de imágenes de entrenamiento.
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=(0.3),
        zoom_range=(0.3),
        width_shift_range=(0.2),
        height_shift_range=(0.2),
        validation_split = 0.2,
        brightness_range=(0.05,0.85),
        horizontal_flip=False)

# Carga de imágenes al generador de entrenamiento desde directorio.
train_generator = train_datagen.flow_from_directory(
        'DataSet',
        class_mode='categorical',
        shuffle=True,
        target_size=(28*k, 28*k),
        color_mode = 'rgb', 
        subset = 'training',
        batch_size=bs)

valid_generator = train_datagen.flow_from_directory(
        'DataSet',
        class_mode='categorical',
        shuffle=True,
        target_size=(28*k, 28*k),
        color_mode = 'rgb', 
        subset = 'validation',
        batch_size=bs)
 
model = tf.keras.applications.VGG19()
num_classes = 24
epochs = 5

# VGG19
VGG19_model = tf.keras.applications.VGG19(input_shape=(28*k,28*k,3), include_top=False, weights='imagenet')
for layer in VGG19_model.layers[:6]:
  layer.trainable = False

model = tf.keras.Sequential()
model.add(VGG19_model) # Añadimos el modelo preentrenado como una capa.
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01), activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation = 'softmax'))

## EJECUCION DEL MODELO
model.compile(loss="categorical_crossentropy", optimizer= SGD(learning_rate=0.01), metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='model', verbose=1, save_best_only=True, monitor = 'val_acc', mode = 'max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)
history= model.fit(train_generator, validation_data = valid_generator, callbacks=[reduce_lr, checkpointer], epochs=epochs)

model.save("hand_model.h5")
model.save_weights("hand_model.weights.h5")