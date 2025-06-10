import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# === 1. Threads & TF konfigurieren ===
num_threads = os.cpu_count()
# TF parallelism
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(1)
# BLAS libs
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)

# === 2. Daten laden und vorverarbeiten ===
def load_data():
    (x_t, y_t), (x_v, y_v) = keras.datasets.cifar10.load_data()
    y_t, y_v = y_t.flatten(), y_v.flatten()
    mask_t = np.isin(y_t, [0,1]); mask_v = np.isin(y_v, [0,1])
    x_t, y_t = x_t[mask_t], y_t[mask_t]
    x_v, y_v = x_v[mask_v], y_v[mask_v]
    x_t = tf.image.resize(x_t/255.0, (32,32)).numpy()
    x_v = tf.image.resize(x_v/255.0, (32,32)).numpy()
    return x_t, to_categorical(y_t, 2), x_v, to_categorical(y_v, 2)

x_train, y_train_o, x_test, y_test_o = load_data()

# === 3. Hilfsfunktionen ===
def clone_optimizer(opt):
    cfg = opt.get_config()
    return type(opt).from_config(cfg)

def run_experiment(params):
    opt_cfg, points, epochs, bs, act, loss_fn = params
    # frischen Optimizer
    opt = clone_optimizer(opt_cfg)
    # Modell
    m = Sequential([
        Flatten(input_shape=(32,32,3)),
        Dense(points, activation=act),
        Dense(2, activation='softmax')
    ])
    m.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'],
              run_eagerly=False)             # <<-- Graph-Modus
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    m.fit(x_train, y_train_o,
          validation_data=(x_test, y_test_o),
          epochs=epochs,
          batch_size=bs,
          callbacks=[early],
          verbose=0)
    _, acc = m.evaluate(x_test, y_test_o, verbose=0)
    return {'activation': act,
            'loss':       loss_fn,
            'batch_size': bs,
            'points':     points,
            'val_accuracy': acc}

# === 4. Grid‐Search konfigurieren ===
sgds = [
    keras.optimizers.SGD(0.01, weight_decay=1e-6, momentum=0.8),
    keras.optimizers.SGD(0.01, weight_decay=1e-6, momentum=0.9),
    # … weitere …
]
acts   = ['relu','tanh']
losses = ['categorical_crossentropy','mean_squared_error']
point, batch_size, epochs = 128, 8, 5

tasks = [
    (opt, point, epochs, batch_size, act, loss_fn)
    for act in acts
    for loss_fn in losses
    for opt in sgds
]

# === 5. Parallel mit Threads ausführen ===
results = []
with ThreadPoolExecutor(max_workers=num_threads) as exe:
    futures = [exe.submit(run_experiment, t) for t in tasks]
    for fut in as_completed(futures):
        try:
            r = fut.result()
            results.append(r)
            print(f"Done: {r['activation']}/{r['loss']} → {r['val_accuracy']:.4f}")
        except Exception as e:
            print("Task error:", e)

# === 6. Bestes Ergebnis finden ===
if results:
    best = max(results, key=lambda x: x['val_accuracy'])
    print("\nBest hyperparameters:", best)
else:
    print("Keine Ergebnisse — irgendwas lief schief.")
