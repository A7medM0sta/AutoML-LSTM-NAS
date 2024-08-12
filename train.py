import numpy as np
import tensorflow as tf
import argparse
import datetime

from cnn import CNN
from net_manager import NetManager
from reinforce import Reinforce

from tensorflow.keras.datasets import mnist


def parse_args():
    desc = "TensorFlow implementation of 'Neural Architecture Search with Reinforcement Learning'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--max_layers", default=2)

    args = parser.parse_args()
    args.max_layers = int(args.max_layers)
    return args


def policy_network(state, max_layers):
    with tf.name_scope("policy_network"):
        nas_cell = tf.compat.v1.nn.rnn_cell.NASCell(
            4 * max_layers
        )  # Adjust for TensorFlow 2.x
        outputs, state = tf.compat.v1.nn.dynamic_rnn(
            nas_cell, tf.expand_dims(state, -1), dtype=tf.float32
        )
        bias = tf.Variable([0.05] * 4 * max_layers)
        outputs = tf.nn.bias_add(outputs, bias)
        return outputs[:, -1:, :]


def train(mnist_data):
    global args

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1, decay_steps=500, decay_rate=0.96, staircase=True
    )
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    reinforce = Reinforce(optimizer, policy_network, args.max_layers, global_step)
    net_manager = NetManager(
        num_input=784,
        num_classes=10,
        learning_rate=0.001,
        mnist=mnist_data,
        batch_size=100,
    )

    MAX_EPISODES = 2500
    step = 0
    state = np.array([[10.0, 128.0, 1.0, 1.0] * args.max_layers], dtype=np.float32)
    pre_acc = 0.0
    total_rewards = 0
    for i_episode in range(MAX_EPISODES):
        action = reinforce.get_action(state)
        if all(ai > 0 for ai in action[0][0]):
            reward, pre_acc = net_manager.get_reward(action, step, pre_acc)
        else:
            reward = -1.0
        total_rewards += reward

        state = action[0]
        reinforce.storeRollout(state, reward)

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
        with open("lg3.txt", "a+") as log:
            log.write(log_str)
        print(log_str)


def main():
    global args
    args = parse_args()

    mnist_data = mnist.load_data()
    train(mnist_data)


if __name__ == "__main__":
    main()
