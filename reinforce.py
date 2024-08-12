import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Add
from tensorflow.keras.models import Model
class Reinforce:
    def __init__(
        self,
        optimizer,
        policy_network_fn,
        max_layers,
        global_step,
        division_rate=100.0,
        reg_param=0.001,
        discount_factor=0.99,
        exploration=0.3,
    ):
        self.optimizer = optimizer
        self.policy_network_fn = policy_network_fn
        self.max_layers = max_layers
        self.global_step = global_step
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor = discount_factor
        self.exploration = exploration

        self.reward_buffer = []
        self.state_buffer = []

        self.create_variables()

    def create_variables(self):
        self.states = Input(shape=(self.max_layers * 4,))

        # Create the policy network
        self.policy_network = self.policy_network_fn(self.states, self.max_layers)
        self.policy_outputs = self.policy_network(self.states)

        self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
        self.predicted_action = tf.cast(
            self.action_scores * self.division_rate, tf.int32, name="predicted_action"
        )

        # Define loss and training operations
        self.discounted_rewards = Input(shape=(None,))

        self.cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.pg_loss = tf.reduce_mean(
            self.cross_entropy_loss(self.states, self.policy_outputs)
        )
        self.reg_loss = tf.reduce_sum(
            [
                tf.reduce_sum(tf.square(x))
                for x in self.policy_network.trainable_variables
            ]
        )
        self.loss = self.pg_loss + self.reg_param * self.reg_loss

        with tf.name_scope("train_policy_network"):
            self.train_op = self.optimizer.minimize(
                self.loss, var_list=self.policy_network.trainable_variables
            )

    def get_action(self, state):
        if tf.random.uniform([]) < self.exploration:
            return tf.random.uniform(
                [1, 4 * self.max_layers], minval=1, maxval=35, dtype=tf.int32
            )
        else:
            return self.policy_network(self.states)

    def store_rollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)

    def train_step(self, steps_count):
        states = (
            tf.convert_to_tensor(self.state_buffer[-steps_count:]) / self.division_rate
        )
        rewards = self.reward_buffer[-steps_count:]
        with tf.GradientTape() as tape:
            logits = self.policy_network(states)
            loss = self.loss_function(logits, states)
        grads = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.policy_network.trainable_variables)
        )
        return loss
