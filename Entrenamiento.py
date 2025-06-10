import tensorflow as tf
from tensorflow import keras

# Directorios de las imágenes de entrenamiento y validación (Editado sin probar. Usaba un absolute path)
train_dir = '\\Fotos\\Entrenamiento'
val_dir = '\\Fotos\\Validacion'

# Parámetros del modelo
img_size = (200, 200)
batch_size = 32
epochs = 50

# Generadores de imágenes con aumento de datos
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Carga de datos
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Definición del modelo
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compilación del modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# Guardar el modelo
model.save("hand_model.h5")
model.save_weights("hand_model.weights.h5")

accuracy = model.evaluate(val_generator)[1]
print("Accuracy:", accuracy)