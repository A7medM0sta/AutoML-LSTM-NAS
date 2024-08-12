import tensorflow as tf
from cnn import CNN


class NetManager:
    def __init__(
        self,
        num_input,
        num_classes,
        learning_rate,
        mnist,
        max_step_per_action=5500 * 3,
        batch_size=100,
        dropout_rate=0.85,
    ):

        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.mnist = mnist

        self.max_step_per_action = max_step_per_action
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

    def get_reward(self, action, step, pre_acc):
        action = [action[0][0][x : x + 4] for x in range(0, len(action[0][0]), 4)]
        cnn_drop_rate = [c[3] for c in action]
        model = CNN(self.num_input, self.num_classes, action)
        loss_op = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        for step in range(self.max_step_per_action):
            batch_x, batch_y = self.mnist.train.next_batch(self.batch_size)
            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)
                loss_value = loss_op(batch_y, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                # Calculate batch loss and accuracy
                logits = model(batch_x, training=False)
                loss_value = loss_op(batch_y, logits)
                acc = tf.keras.metrics.Accuracy()(batch_y, tf.argmax(logits, axis=1))
                print(
                    "Step "
                    + str(step)
                    + ", Minibatch Loss= "
                    + "{:.4f}".format(loss_value.numpy())
                    + ", Current accuracy= "
                    + "{:.3f}".format(acc.numpy())
                )
        batch_x, batch_y = self.mnist.test.next_batch(10000)
        logits = model(batch_x, training=False)
        loss_value = loss_op(batch_y, logits)
        acc = tf.keras.metrics.Accuracy()(batch_y, tf.argmax(logits, axis=1))
        print("!!!!!!acc:", acc.numpy(), pre_acc)
        if acc.numpy() - pre_acc <= 0.01:
            return acc.numpy(), acc.numpy()
        else:
            return 0.01, acc.numpy()
