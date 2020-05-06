import config
import tensorflow.keras
import tensorflow.keras.layers as layers


class DiscriminatorModel(tensorflow.keras.Model):
    # Use probability of 0.3 to dropout.
    DROPOUT_DISC = 0.3

    def __init__(self, image_shape):
        super(DiscriminatorModel, self).__init__()
        # 3 convolution layers and then a sigmoid of 1 node out.
        self.conv1 = layers.Conv2D(
            filters=28, kernel_size=(5, 5), input_shape=image_shape, strides=(2, 2), padding="same"
        )
        self.conv2 = layers.Conv2D(56, (5, 5), strides=(2, 2), padding="same")
        self.conv3 = layers.Conv2D(112, (5, 5), strides=(2, 2), padding="same")
        # Recommended in the paper to use leakyrelu
        self.leakyrelu = layers.LeakyReLU(alpha=0.3)
        self.dropout = layers.Dropout(self.DROPOUT_DISC)
        self.flatten = layers.Flatten()
        self.batchnorm0 = layers.BatchNormalization()
        self.batchnorm1 = layers.BatchNormalization()
        self.batchnorm2 = layers.BatchNormalization()
        self.batchnorm3 = layers.BatchNormalization()
        self.output_layer = layers.Dense(1, activation="sigmoid")
        self.compilation()

    def compilation(self):
        optimizer = tensorflow.keras.optimizers.Adam(lr=0.00008)
        self.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    def call(self, x):
        if config.GAN_BATCHNORM_DISC:
            x = self.batchnorm0(x)
        x = self.conv1(x)
        if config.GAN_BATCHNORM_DISC:
            x = self.batchnorm1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if config.GAN_BATCHNORM_DISC:
            x = self.batchnorm2(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        if config.GAN_BATCHNORM_DISC: # TODO: Try without this.
            x = self.batchnorm3(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.flatten(x)
        return self.output_layer(x)
