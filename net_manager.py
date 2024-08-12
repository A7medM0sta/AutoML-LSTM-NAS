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
        batch_size=100,  # Fixed typo from 'bathc_size' to 'batch_size'
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

        # Create the model
        model = CNN(self.num_input, self.num_classes, action)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Define a training step function
        @tf.function
        def train_step(batch_x, batch_y):
            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)
                loss = tf.reduce_mean(model.loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        # Prepare dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.mnist.train.images, self.mnist.train.labels)
        )
        train_dataset = train_dataset.shuffle(10000).batch(self.batch_size)

        for epoch in range(self.max_step_per_action):
            for batch_x, batch_y in train_dataset:
                loss = train_step(batch_x, batch_y)

                if epoch % 100 == 0:
                    # Calculate batch loss and accuracy
                    logits = model(batch_x, training=False)
                    loss_value = tf.reduce_mean(model.loss)
                    acc_metric = tf.keras.metrics.CategoricalAccuracy()
                    acc_metric.update_state(batch_y, logits)
                    acc = acc_metric.result().numpy()
                    print(
                        "Epoch "
                        + str(epoch)
                        + ", Minibatch Loss= "
                        + "{:.4f}".format(loss_value)
                        + ", Current accuracy= "
                        + "{:.3f}".format(acc)
                    )

        # Evaluate the model
        test_images, test_labels = self.mnist.test.images, self.mnist.test.labels
        logits = model(test_images, training=False)
        loss_value = tf.reduce_mean(model.loss)
        acc_metric = tf.keras.metrics.CategoricalAccuracy()
        acc_metric.update_state(test_labels, logits)
        acc = acc_metric.result().numpy()

        print("!!!!!!acc:", acc, pre_acc)
        if acc - pre_acc <= 0.01:
            return acc, acc
        else:
            return 0.01, acc
