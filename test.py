# Ressourcenschonende Flugzeug/Auto Klassifikation: Hyperparameter-Suche (32×32)
import os
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# TensorFlow-Log-Level reduzieren
def suppress_tf_warnings():
    import logging, os
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
suppress_tf_warnings()

# 1. Datensatz laden & filtern
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
mask_train = np.isin(y_train, [0,1]); mask_test = np.isin(y_test, [0,1])
x_train, y_train = x_train[mask_train], y_train[mask_train]
x_test,  y_test  = x_test[mask_test],  y_test[mask_test]

# 2. Auf 32×32 runterskalieren & normalisieren
IMG_SIZE = 32
x_train = tf.image.resize(x_train.astype('float32')/255.0, (IMG_SIZE, IMG_SIZE)).numpy()
x_test  = tf.image.resize(x_test.astype('float32')/255.0,  (IMG_SIZE, IMG_SIZE)).numpy()

y_train_o = to_categorical(y_train, 2)
y_test_o  = to_categorical(y_test, 2)

# 3. Hyperparameter-Raum
dropout_rates = [0.2, 0.3, 0.5]
learning_rates = [1e-4, 1e-3]
batch_sizes    = [16, 32]
num_filters    = [16, 32]
kernel_sizes   = [3]
epochs         = 10

# 4. Model-Bauer

def build_model(dropout_rate, learning_rate, filters, kernel_size):
    model = Sequential([
        Conv2D(filters, (kernel_size, kernel_size), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(filters*2, (kernel_size, kernel_size), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Flatten(),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(2, activation='softmax')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 5. EarlyStopping-Callback
early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

# 6. Grid Search
total_configs = list(itertools.product(dropout_rates, learning_rates, batch_sizes, num_filters, kernel_sizes))
results = []
for dp, lr, bs, nf, ks in total_configs:
    print(f"Training mit dropout={dp}, lr={lr}, batch_size={bs}, filters={nf}, kernel={ks}")
    model = build_model(dp, lr, nf, ks)
    history = model.fit(
        x_train, y_train_o,
        validation_data=(x_test, y_test_o),
        epochs=epochs,
        batch_size=bs,
        callbacks=[early_stopping],
        verbose=0
    )
    loss, acc = model.evaluate(x_test, y_test_o, verbose=0)
    results.append({
        'dropout': dp,
        'learning_rate': lr,
        'batch_size': bs,
        'filters': nf,
        'kernel_size': ks,
        'accuracy': acc
    })
    print(f"-> Val Accuracy: {acc:.4f}\n")

# 7. Bestes Ergebnis finden
best = max(results, key=lambda x: x['accuracy'])
print("Beste Hyperparameter-Kombination:", best)

# 8. (Optional) Modell mit besten Parametern speichern
# best_model = build_model(best['dropout'], best['learning_rate'], best['filters'], best['kernel_size'])
# best_model.fit(x_train, y_train_o, epochs=epochs, batch_size=best['batch_size'], callbacks=[early_stopping])
# best_model.save('best_cnn_model.h5')

# Die Klassifikation eigener Bilder erfolgt später, sobald Bilder vorliegen.
