# Code By Jibril Richardson

import random
import math
import os
import matplotlib.pyplot as plt
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("Training Logs/MASTERLOGS", exist_ok=True)

traininginterval = int(input("Enter Training Interval: "))
graphtraining = str(input("Graph Training Progress? (y/n): ")).lower()
traindisp = str(input("Display Training Progress? (y/n): ")).lower()
savelogs = input("Save training logs? (y/n): ").lower()

print("Training...")

logfile = open(f"Training Logs/MASTERLOGS/training_{timestamp}.txt", "w")

if savelogs == "y": logfile.write("=== Neural Network Training Log ===\n")

# sigmoid activation
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# derivative for learning
def sigmoid_derivative(x):
    return x * (1 - x)

# training data
training_data = [
    ([0,0], 0),
    ([0,1], 1),
    ([1,0], 1),
    ([1,1], 0)
]

# random weights
w1 = random.random()
w2 = random.random()
w3 = random.random()
w4 = random.random()
w5 = random.random()
w6 = random.random()

# biases
b1 = random.random()
b2 = random.random()
b3 = random.random()

errors = []

learning_rate = 0.5

if graphtraining == "y":
    plt.ion()
    fig, ax = plt.subplots()

# training loop
for epoch in range(traininginterval):

    total_error = 0

    for inputs, target in training_data:

        x1, x2 = inputs

        # hidden layer
        h1 = sigmoid(x1*w1 + x2*w2 + b1)
        h2 = sigmoid(x1*w3 + x2*w4 + b2)

        # output layer
        output = sigmoid(h1*w5 + h2*w6 + b3)

        # error
        error = target - output

        total_error += abs(error)

        # output adjustments
        d_output = error * sigmoid_derivative(output)

        # backprop hidden layer
        d_h1 = d_output * w5 * sigmoid_derivative(h1)
        d_h2 = d_output * w6 * sigmoid_derivative(h2)

        # update output weights
        w5 += h1 * d_output * learning_rate
        w6 += h2 * d_output * learning_rate

        # update hidden weights
        w1 += x1 * d_h1 * learning_rate
        w2 += x2 * d_h1 * learning_rate

        w3 += x1 * d_h2 * learning_rate
        w4 += x2 * d_h2 * learning_rate

    errors.append(total_error)

        # update graph every 100 epochs
    if graphtraining == "y":
        if epoch % 100 == 0:
            ax.clear()

            ax.plot(errors)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Total Error")
            ax.set_title("Neural Network Learning")


            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

    if epoch % 1000 == 0:
        accuracy = (1 - (total_error / 4)) * 100
        if logfile:
            logfile.write(
                f"Epoch: {epoch} | "
                f"Error: {total_error:.4f} | "
                f"Accuracy: {accuracy:.2f}%\n"
            )

    if traindisp == "y":

        if epoch % 1000 == 0:
            accuracy = (1 - (total_error / 4)) * 100
            errors.append(accuracy)
            print("Epoch: ", epoch, "Total Error: ", total_error, "Accuracy: ", accuracy, "%")

        if epoch % 5000 == 0:
            print("Weight 1: ", w1, "Weight 2: ", w2, "Weight 3: ", w3, "Weight 4: ", w4)

    elif traindisp == "n":

        if epoch % 10000 == 0:
            print("Still training... Epoch: ", epoch)


if graphtraining == "y":
    if plt.ioff():
        plt.show()

if logfile:
    logfile.write("\n=== Final Weights ===\n")

    logfile.write(f"w1={w1}\n")
    logfile.write(f"w2={w2}\n")
    logfile.write(f"w3={w3}\n")
    logfile.write(f"w4={w4}\n")
    logfile.write(f"w5={w5}\n")
    logfile.write(f"w6={w6}\n")


if logfile:
    logfile.write("\n=== Final Model Outputs ===\n")
# test
for inputs, target in training_data:
    x1, x2 = inputs

    h1 = sigmoid(x1 * w1 + x2 * w2 + b1)
    h2 = sigmoid(x1 * w3 + x2 * w4 + b2)

    output = sigmoid(h1 * w5 + h2 * w6 + b3)

    print(inputs, "->", round(output, 3))

    if logfile:
        logfile.write(
            f"Input: {inputs} | "
            f"Expected: {target} | "
            f"Output: {round(output, 3)}\n"
        )

logfile.close()

print("TRAINING COMPLETE. Logs saved to: ", f"Training Logs/MASTERLOGS/training_{timestamp}.txt")