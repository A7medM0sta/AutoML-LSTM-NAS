import tensorflow as tf
import random
import numpy as np


class Reinforce(tf.keras.Model):
    def __init__(
        self,
        policy_network,
        max_layers,
        division_rate=100.0,
        reg_param=0.001,
        discount_factor=0.99,
        exploration=0.3,
    ):
        super(Reinforce, self).__init__()
        self.policy_network = policy_network
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor = discount_factor
        self.max_layers = max_layers
        self.exploration = exploration

        self.reward_buffer = []
        self.state_buffer = []

    def get_action(self, state):
        if random.random() < self.exploration:
            return np.array([[random.sample(range(1, 35), 4 * self.max_layers)]])
        else:
            return tf.cast(
                tf.scalar_mul(self.division_rate, self.policy_network(state)), tf.int32
            )

    def train_step(self, states, rewards, optimizer):
        with tf.GradientTape() as tape:
            action_scores = self.policy_network(states)
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=action_scores, labels=states
            )
            pg_loss = tf.reduce_mean(cross_entropy_loss)
            reg_loss = tf.reduce_sum(
                [
                    tf.reduce_sum(tf.square(x))
                    for x in self.policy_network.trainable_variables
                ]
            )
            loss = pg_loss + self.reg_param * reg_loss

        grads = tape.gradient(loss, self.policy_network.trainable_variables)
