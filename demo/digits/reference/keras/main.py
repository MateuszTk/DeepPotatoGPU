import struct
import numpy as np
import keras
import time
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

TRAIN_IMAGES_FILE = "../../data/train-images.idx3-ubyte"
TRAIN_LABELS_FILE = "../../data/train-labels.idx1-ubyte"
TEST_IMAGES_FILE = "../../data/t10k-images.idx3-ubyte"
TEST_LABELS_FILE = "../../data/t10k-labels.idx1-ubyte"
BATCH_SIZE = 30
TEST_INTERVAL = 6000

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols, 1).astype(np.float32) / 255.0
    return data

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

x_train = load_images(TRAIN_IMAGES_FILE)
y_train = to_categorical(load_labels(TRAIN_LABELS_FILE), 10)
x_test = load_images(TEST_IMAGES_FILE)
y_test = to_categorical(load_labels(TEST_LABELS_FILE), 10)

model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(10, activation='sigmoid')
])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=1.0), loss='mse', metrics=['accuracy'])

timer_start = time.time()
samples_processed = 0
for start_idx in range(0, len(x_train), BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE
    batch_x = x_train[start_idx:end_idx]
    batch_y = y_train[start_idx:end_idx]

    model.train_on_batch(batch_x, batch_y)
    samples_processed += len(batch_x)

    if samples_processed % TEST_INTERVAL == 0:
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"After {samples_processed} samples: Test loss={loss:.4f}, Test acc={acc:.4f}")

timer_stop = time.time()

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Test loss={loss:.4f}, Test acc={acc:.4f}", f"Speed={samples_processed / (timer_stop - timer_start):.2f} samples/sec")
