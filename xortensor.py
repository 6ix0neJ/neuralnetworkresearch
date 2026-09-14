import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

# Training data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)

# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu", input_shape=(2,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Configure training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
traininterval = int(input("How many epochs?: "))

class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        print(f"\nEpoch {epoch + 1} took {epoch_time * 1000:.2f} ms")

def epoch_time_graph(epoch_times):
    epochs = list(range(1, len(epoch_times) + 1))

    epoch_times_ms = [t * 1000 for t in epoch_times]

    plt.figure(figsize=(10, 5))
    plt.bar(epochs, epoch_times_ms, width=0.4, color='magenta')

    plt.title("Epoch Times")
    plt.xlabel("Epoch #")
    plt.ylabel("Completion Time (milliseconds)")
    plt.xticks(epochs)

    plt.show()

# Train

timer = EpochTimer()
model.fit(
    X,
    y,
    epochs=traininterval,
    verbose=0,
    callbacks=[timer]
    )
"""
#print(model.history)
#train_loss = model.history.history['loss']
#val_loss = model.history.history['val_loss']

# plotting

#plt.plot(model.history.history['loss'])
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Training Loss')
#plt.show()

#plt.plot(model.history.history['accuracy'])
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.title('Training Accuracy')
"""

# Test

epochtimegraphyn = input("Do you want to see the epoch time graph? (y/n): ")
if epochtimegraphyn.lower() == "y":
    print(timer.epoch_times)
    epoch_time_graph(timer.epoch_times)

predictions = model.predict(X)

for inputs, prediction in zip(X, predictions):
    print(inputs, "->", prediction[0])
