import numpy as np
import tensorflow as tf
import argparse
import datetime

from cnn import CNN
from net_manager import NetManager
from reinforce import Reinforce

# TensorFlow 2.x does not use the deprecated input_data API. Use the updated MNIST dataset loading method.
from tensorflow.keras.datasets import mnist


def parse_args():
    desc = "TensorFlow implementation of 'Neural Architecture Search with Reinforcement Learning'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--max_layers", default=2)

    args = parser.parse_args()
    args.max_layers = int(args.max_layers)
    return args



def policy_network(state, max_layers):
    # Define the policy network using the Keras Functional API
    inputs = tf.keras.Input(shape=(4 * max_layers,))
    x = tf.expand_dims(inputs, -1)  # Add channel dimension

    # Define NASCell equivalent (or use a different Keras layer if NASCell is not available)
    nas_cell = tf.keras.layers.LSTM(4 * max_layers, return_sequences=True)(x)
    bias = tf.Variable(tf.constant([0.05] * 4 * max_layers, dtype=tf.float32))
    x = tf.keras.layers.Add()([nas_cell, bias])

    # Output layer
    outputs = tf.keras.layers.Dense(4 * max_layers)(x)
    model = tf.keras.Model(inputs, outputs)

    return model(inputs)


def train(mnist_data):
    global args

    # TensorFlow 2.x uses eager execution by default, so no need to create a session
    # Define global step variable and learning rate decay
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1, decay_steps=500, decay_rate=0.96, staircase=True
    )

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    reinforce = Reinforce(optimizer, policy_network, args.max_layers, global_step)

    (x_train, y_train), (x_test, y_test) = mnist_data

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    net_manager = NetManager(
        num_input=784,
        num_classes=10,
        learning_rate=0.001,
        mnist=(x_train, y_train, x_test, y_test),
        batch_size=100,
    )

    MAX_EPISODES = 2500
    step = 0
    state = np.array([[10.0, 128.0, 1.0, 1.0] * args.max_layers], dtype=np.float32)
    pre_acc = 0.0
    total_rewards = 0

    for i_episode in range(MAX_EPISODES):
        action = reinforce.get_action(state)
        print("ca:", action)
        if all(ai > 0 for ai in action[0][0]):
            reward, pre_acc = net_manager.get_reward(action, step, pre_acc)
            print("=====>", reward, pre_acc)
        else:
            reward = -1.0
        total_rewards += reward

        # In our sample action is equal state
        state = action[0]
        reinforce.storeRollout(state, reward)

        step += 1
        ls = reinforce.train_step(1)
        log_str = f"current time: {datetime.datetime.now().time()} episode: {i_episode} loss: {ls} last_state: {state} last_reward: {reward}\n"
        with open("lg3.txt", "a+") as log:
            log.write(log_str)
        print(log_str)


def main():
    global args
    args = parse_args()

    # Load MNIST dataset using TensorFlow 2.x method
    mnist_data = mnist.load_data()
    train(mnist_data)


if __name__ == "__main__":
    main()
