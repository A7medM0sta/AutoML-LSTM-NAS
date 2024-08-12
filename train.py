import numpy as np
import tensorflow as tf
import argparse
import datetime

from cnn import CNN
from net_manager import NetManager
from reinforce import Reinforce

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Add
from tensorflow.keras.models import Model


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
    # Define the input layer
    inputs = Input(shape=(4 * max_layers,))

    # Reshape input to add channel dimension
    x = tf.expand_dims(inputs, -1)  # Add channel dimension

    # Define LSTM layer
    x = LSTM(4 * max_layers, return_sequences=True)(x)

    # Add bias as a constant
    bias = tf.Variable(tf.constant([0.05] * 4 * max_layers, dtype=tf.float32))
    x = Add()([x, bias])

    # Output layer
    outputs = Dense(4 * max_layers)(x)

    # Create and return the model
    model = Model(inputs, outputs)
    return model


def train(mnist):
    global args
    sess = tf.compat.v1.Session()
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.compat.v1.train.exponential_decay(
        starter_learning_rate, global_step, 500, 0.96, staircase=True
    )

    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)

    # Pass the policy_network function
    reinforce = Reinforce(optimizer, policy_network, args.max_layers, global_step)
    net_manager = NetManager(
        num_input=784, num_classes=10, learning_rate=0.001, mnist=mnist, batch_size=100
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
        reinforce.store_rollout(state, reward)

        step += 1
        ls = reinforce.train_step(1)
        log_str = (
            "current time:  "
            + str(datetime.datetime.now().time())
            + " episode:  "
            + str(i_episode)
            + " loss:  "
            + str(ls)
            + " last_state:  "
            + str(state)
            + " last_reward:  "
            + str(reward)
            + "\n"
        )
        log = open("lg3.txt", "a+")
        log.write(log_str)
        log.close()
        print(log_str)


def main():
    global args
    args = parse_args()

    # Load MNIST dataset using TensorFlow 2.x method
    mnist_data = mnist.load_data()
    train(mnist_data)


if __name__ == "__main__":
    main()
