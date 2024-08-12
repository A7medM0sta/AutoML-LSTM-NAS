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
    nas_cell = tf.keras.layers.GRUCell(4 * max_layers)
    outputs, state = nas_cell(tf.expand_dims(state, -1))
    bias = tf.Variable([0.05] * 4 * max_layers)
    outputs = tf.nn.bias_add(outputs, bias)
    return outputs[:, -1:, :]


def train(mnist):
    global args
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        0.99, global_step, 500, 0.96, staircase=True
    )

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    reinforce = Reinforce(policy_network, args.max_layers)
    net_manager = NetManager(
        num_input=784, num_classes=10, learning_rate=0.001, mnist=mnist, bathc_size=100
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
        reinforce.store_rollout(state, reward)

        step += 1
        ls = reinforce.train_step(state, reward, optimizer)
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

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train((x_train, y_train, x_test, y_test))


if __name__ == "__main__":
    main()
