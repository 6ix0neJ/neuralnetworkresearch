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
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        print(f"\nEpoch {epoch + 1} took {epoch_time:.2f} seconds")
# Train
model.fit(X, y, epochs=traininterval, verbose=0, callbacks=[EpochTimer()])


#print(model.history)
#train_loss = model.history.history['loss']
#val_loss = model.history.history['val_loss']

# plotting

#plt.plot(model.history.history['loss'])
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Training Loss')
#plt.show()

plt.plot(model.history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
# Test



predictions = model.predict(X)
training = False
for inputs, prediction in zip(X, predictions):
    print(inputs, "->", prediction[0])