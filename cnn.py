import tensorflow as tf
from tensorflow.keras import layers


class CNN(tf.keras.Model):
    def __init__(self, num_input, num_classes, cnn_config):
        super(CNN, self).__init__()
        cnn = [c[0] for c in cnn_config]
        cnn_num_filters = [c[1] for c in cnn_config]
        max_pool_ksize = [c[2] for c in cnn_config]

        self.cnn_layers = []
        for idd, filter_size in enumerate(cnn):
            self.cnn_layers.append(
                layers.Conv1D(
                    filters=cnn_num_filters[idd],
                    kernel_size=int(filter_size),
                    strides=1,
                    padding="SAME",
                    activation=tf.nn.relu,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    name="conv_out_" + str(idd),
                )
            )
            self.cnn_layers.append(
                layers.MaxPooling1D(
                    pool_size=int(max_pool_ksize[idd]),
                    strides=1,
                    padding="SAME",
                    name="max_pool_" + str(idd),
                )
            )
            self.cnn_layers.append(layers.Dropout(rate=0.5, name="dropout_" + str(idd)))

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        X = tf.expand_dims(inputs, -1)
        for layer in self.cnn_layers:
            X = layer(X, training=training)
        X = self.flatten(X)
        logits = self.dense(X)
        return logits
