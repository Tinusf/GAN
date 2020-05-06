import tensorflow as tf
import os
import matplotlib.pyplot as plt
import draw
import tensorflow.keras
import numpy as np
import config
from tensorflow.keras import backend as K


class GANModel(tensorflow.keras.Model):
    def __init__(self, generator, discriminator, classifier=None, num_classes=10):
        super(GANModel, self).__init__()
        # This flag is if the GAN should use the classifier when calculating loss.
        self.use_classifier = config.GAN_USE_CLASSIFIER
        self.num_classes = num_classes
        self.generator = generator
        self.discriminator = discriminator
        self.prev_perf_array = None
        optimizer = tensorflow.keras.optimizers.Adam(lr=0.0004)

        if self.use_classifier:
            # If we are using the classifier then we need to compile it eagerly and use a custom
            # loss function.
            self.classifier = classifier
            self.compile(optimizer=optimizer, loss=self.custom_loss_func, metrics=["accuracy"],
                         experimental_run_tf_function=False, run_eagerly=True)
        else:
            self.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    def call(self, inputs):
        generated_images = self.generator.call(inputs)
        if self.use_classifier:
            # If we are using the classifier we need to save the generated images for later.
            self.generated_images = generated_images
        return self.discriminator.call(generated_images)

    @staticmethod
    def softargmax(x, beta=1e10):
        """ In order to keep gradients we have to use soft argmax instead of the real argmax.
        Cannot do this K.argmax(predicted_classes, axis=1) without losing gradients
        Code taken from:
        https://stackoverflow.com/questions/46926809/getting-around-tf-argmax-which-is-not-differentiable
        """
        x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
        return tf.reduce_sum(tf.nn.softmax(x * beta) * x_range, axis=-1)

    def get_perfect_array(self, size):
        """
        This function gets a perfectly balanced array of size `size`.
        Example if size is 20 and the number of classes is 10 then the perfectly balanced array is
        [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
        """
        classes = self.num_classes
        n_each = int(size) // classes
        if self.prev_perf_array is not None:
            perf_bal_list = self.prev_perf_array
        else:
            perf_bal_list = []
            extras = int(size) % classes
            for i in range(classes - 1, -1, -1):
                cur_n_each = n_each
                if extras > 0:
                    cur_n_each +=1
                    extras -= 1
                perf_bal_list.extend([i]* cur_n_each)
            self.prev_perf_array = perf_bal_list

        perfect_balanced = tf.cast(np.array(perf_bal_list), dtype=tf.float64)
        return perfect_balanced

    def custom_loss_func(self, y_true, y_pred):
        # This custom loss function is only used when the classifier is being used.
        # First calculate the regular crossentropy loss.
        reconstruction_loss = tensorflow.keras.losses.binary_crossentropy(y_true, y_pred)
        # Predict the generated images using the classifier
        predicted_classes = self.classifier(self.generated_images, training=False)
        # Get the batch size
        batch_size = K.shape(y_true)[0]
        # Get the most likely classes
        predicted_classes = self.softargmax(predicted_classes)
        # Get the predicted classes in sorted order.
        predicted_classes, _ = tf.math.top_k(predicted_classes, k=batch_size)
        # Create a perfectly balanced tensor with all the classes.
        perfect_balanced = tf.constant(self.get_perfect_array(batch_size))
        # Run Mean squared error between the perfectly balanced tensor and the predicted classes
        # one. This will punish predictions that are very imbalanced.
        entropy_loss = tensorflow.keras.losses.MSE(predicted_classes, perfect_balanced)
        entropy_loss = tf.cast(entropy_loss, tf.float32)
        # Combine the reconstruction loss with the entropy loss. 
        return K.mean(entropy_loss) + K.mean(reconstruction_loss)

    def train(self, data_generator):
        # This method trains a GAN using a data_generator.
        x_train, _ = data_generator.get_full_data_set(training=True)
        x_train = x_train.astype(np.float64)

        batch_size = config.GAN_BATCH_SIZE
        # Keep track of the generator losses and the discriminator losses in order to plot it
        # later.
        generator_losses = []
        discriminator_losses = []
        minibatches_total = 0

        # labels for generator is a 1 for every fake image. Because you hope the
        # discriminator thinks its a real image, even though it isnt.
        gen_labels = np.ones(batch_size)

        if not os.path.exists(config.GAN_FOLDER_GRAPHS):
            os.mkdir(config.GAN_FOLDER_GRAPHS)

        for epoch in range(config.GAN_EPOCHS):
            print(f"Starting epoch {epoch}")
            minibatches = len(x_train) // batch_size
            latents = np.random.randn(batch_size, self.generator.input_dim)
            for i in range(minibatches):
                print(f"Starting batch {i} / {minibatches}")
                minibatches_total += 1
                # Generate random latents

                # Create fake images using the generator
                fake_images = self.generator.predict(latents, batch_size=batch_size)
                real_images = np.array(x_train[batch_size * i: batch_size * (i + 1)])
                # every 50th minibatch it plots the fake images.
                if i % 25 == 0:
                    # Plot the generated pictures.
                    filename = f"{config.GAN_FOLDER_GRAPHS}/reconstruction{epoch}-{i}.png"
                    # Plot, but save to file instead of showing.
                    draw.draw_images(np.array(fake_images[0:16]), save_to_file=True,
                                     filename=filename)
                # The labels for the discriminators are 0 for the fakes and 1 for the reals.
                disc_labels = np.array([0] * batch_size + [1] * batch_size)
                # Concatenate the fake images and the real images.
                all_images = np.concatenate([fake_images, real_images], axis=0)

                # Make the discriminator trainable
                self.discriminator.trainable = True
                self.discriminator.compilation()

                # Train the discriminator
                history = self.discriminator.fit(all_images, disc_labels, verbose=0)
                disc_loss = sum(history.history["loss"]) / len(history.history["loss"])
                print("Discrimination loss", disc_loss)
                discriminator_losses.append(disc_loss)

                # Train the generator

                # Make the discriminator untrainable
                self.discriminator.trainable = False
                self.discriminator.compilation()

                # Train the generator.
                history = self.fit(latents, gen_labels, verbose=0)
                gen_loss = sum(history.history["loss"]) / len(history.history["loss"])
                print("Generator loss", gen_loss)
                generator_losses.append(gen_loss)

        xes = list(range(0, minibatches_total))
        fig = plt.figure()
        # Plot the generator loss and the discriminator loss.
        plt.plot(xes, generator_losses, "b", label="Generator loss")
        plt.plot(xes, discriminator_losses, "r", label="Discriminator loss")
        plt.legend()
        plt.savefig(f"{config.GAN_FOLDER_GRAPHS}/gan_loss_last.png")
        plt.close(fig)
