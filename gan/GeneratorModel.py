import tensorflow.keras
import tensorflow.keras.layers as layers
import config


class GeneratorModel(tensorflow.keras.Model):
    DROPOUT_GEN = 0.3

    def __init__(self, image_shape, input_dim=config.GAN_INPUT_DIM_SIZE):
        self.input_dim = input_dim
        super(GeneratorModel, self).__init__()
        # How many nodes there are in the flattened image.
        flatten_nodes = image_shape[0] * image_shape[1] * image_shape[2]
        # How many nodes in a 7*7 image
        n_nodes_7_7 = 7 * 7 * image_shape[2]
        self.layer1 = layers.Dense(n_nodes_7_7, input_dim=input_dim)
        self.relu = layers.Activation("relu")
        # Reshape the flattened nodes into a 3 dimension image.
        self.reshape_7_7_layer = layers.Reshape((7, 7, image_shape[2]), input_shape=(
            n_nodes_7_7,))
        # Transposed convolutions
        self.conv2trans_layer = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2),
                                                       padding="same")
        self.conv2trans_layer2 = layers.Conv2DTranspose(32, (5, 5), strides=(2, 2),
                                                        padding="same")
        self.flatten = layers.Flatten()
        self.layer2 = layers.Dense(flatten_nodes)
        self.dropout = layers.Dropout(self.DROPOUT_GEN)

        self.batchnorm0 = layers.BatchNormalization()
        self.batchnorm1 = layers.BatchNormalization()
        self.batchnorm2 = layers.BatchNormalization()
        self.batchnorm3 = layers.BatchNormalization()
        self.batchnorm4 = layers.BatchNormalization()

        # Output images in the original shape.
        self.output_layer = layers.Reshape(image_shape, input_shape=(flatten_nodes,))

    def call(self, x):
        if config.GAN_BATCHNORM_GEN:
            x = self.batchnorm0(x)
        x = self.layer1(x)
        if config.GAN_BATCHNORM_GEN:
            x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.reshape_7_7_layer(x)
        x = self.conv2trans_layer(x)
        if config.GAN_BATCHNORM_GEN:
            x = self.batchnorm2(x)
        x = self.conv2trans_layer2(x)
        if config.GAN_BATCHNORM_GEN:
            x = self.batchnorm3(x)
        x = self.flatten(x)
        x = self.layer2(x)
        if config.GAN_BATCHNORM_GEN:
            x = self.batchnorm4(x) # TODO: Try without this.
        x = self.relu(x)
        x = self.dropout(x)
        return self.output_layer(x)
