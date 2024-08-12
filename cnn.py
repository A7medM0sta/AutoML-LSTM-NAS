import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self, num_input, num_classes, cnn_config):
        super(CNN, self).__init__()
        cnn = [c[0] for c in cnn_config]
        cnn_num_filters = [c[1] for c in cnn_config]
        max_pool_ksize = [c[2] for c in cnn_config]

        # Define the layers
        self.conv_layers = []
        self.pool_layers = []
        for idd, filter_size in enumerate(cnn):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    filters=cnn_num_filters[idd],
                    kernel_size=int(filter_size),
                    strides=1,
                    padding="same",
                    activation="relu",
                    kernel_initializer="he_normal",
                    name=f"conv_out_{idd}",
                )
            )
            self.pool_layers.append(
                tf.keras.layers.MaxPooling1D(
                    pool_size=int(max_pool_ksize[idd]),
                    strides=1,
                    padding="same",
                    name=f"max_pool_{idd}",
                )
            )

        self.dropout_layers = [tf.keras.layers.Dropout(rate) for rate in cnn_config]
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_classes, activation=None)

    def call(self, inputs, training=False):
        x = tf.expand_dims(inputs, -1)  # Add channel dimension
        for conv, pool, drop in zip(
            self.conv_layers, self.pool_layers, self.dropout_layers
        ):
            x = conv(x)
            x = pool(x)
            x = drop(x, training=training)
        x = self.flatten(x)
        logits = self.dense(x)
        return logits

    def compute_loss(self, logits, labels):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)

    def compute_accuracy(self, logits, labels):
        predictions = tf.argmax(logits, axis=1)
        return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
