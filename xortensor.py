import tensorflow as tf

# Create the neural network
xor = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

training_data = [
    ([0,0], 0),
    ([0,1], 1),
    ([1,0], 1),
    ([1,1], 0)
]

# Prepare the training data
X_train = tf.constant([data[0] for data in training_data], dtype=tf.float32)
y_train = tf.constant([data[1] for data in training_data], dtype=tf.float32)

xor.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
epochselect = int(input("How many epochs?: "))
xor.fit(X_train, y_train, epochs=epochselect)

while xor.fit(X_train, y_train, epochs=epochselect):
    print(xor.epochs)