import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Ruta a los keypoints
KEYPOINTS_DIR = 'extracted_keypoints_holistic'

# Cargar los datos
samples = []
for file in os.listdir(KEYPOINTS_DIR):
    if file.endswith('.npy'):
        data = np.load(os.path.join(KEYPOINTS_DIR, file))
        if data.shape == (30, 258):
            samples.append(data)

X = np.array(samples, dtype=np.float32)
X_train, X_test = train_test_split(X, test_size=0.2)

timesteps = 30
features = 258

# Modelo autoencoder
inputs = Input(shape=(timesteps, features))
encoded = LSTM(128, activation='relu')(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(features))(decoded)  # Aquí arreglamos el error

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test),
                epochs=40, batch_size=8,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

# Guardar modelo y encoder
from tensorflow.keras.models import Model

encoder = Model(inputs, encoded)
autoencoder.save('gesture_autoencoder.h5')
encoder.save('gesture_encoder.h5')

print("✅ Autoencoder entrenado correctamente.")
