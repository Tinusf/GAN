import os
import draw
from stacked_mnist import StackedMNISTData
import numpy as np
import config
import util
import basic_classifier_model
from gan.GANModel import GANModel
from gan.DiscriminatorModel import DiscriminatorModel
from gan.GeneratorModel import GeneratorModel
import tensorflow as tf

# This is a wierd fix for my GPU to get it to always work.
gpu = tf.config.experimental.list_physical_devices('GPU')
if gpu:
    print("Setting memory growth on GPU 0")
    tf.config.experimental.set_memory_growth(gpu[0], True)


def main():
    datamode = config.GAN_DATAMODE
    data_generator = StackedMNISTData(mode=datamode, default_batch_size=2048)

    x_test, y_test = data_generator.get_full_data_set(training=False)
    x_test = x_test.astype(np.float64)

    image_shape = x_test[0].shape

    # Create a generator model and a discriminator model.
    generator_model = GeneratorModel(image_shape)
    discriminator_model = DiscriminatorModel(image_shape)

    classifier = None

    # Colors have 1000 classes, whilst regular mnist only has 10.
    if datamode.name.startswith("COLOR"):
        num_classes = 1000
    else:
        num_classes = 10

    if config.GAN_USE_CLASSIFIER:
        # Get a regular classifier model.
        classifier = basic_classifier_model.get_model(datamode, data_generator,
                                                      num_classes=num_classes,
                                                      input_shape=image_shape)

    # GAN model.
    gan_model = GANModel(generator_model, discriminator_model, classifier, num_classes)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1))
    model.compile("adam", loss=GANModel.custom_loss_func)

    # In order to save different models with the classifier and without.
    folder = f"./models/gan_{config.GAN_BATCH_SIZE}_{int(config.GAN_USE_CLASSIFIER)}" \
             f"_{config.GAN_INPUT_DIM_SIZE}_{datamode.name}_{int(config.GAN_BATCHNORM_GEN)}/"

    if not os.path.exists(folder):
        os.mkdir(folder)

    gan_file_name = folder + "model.tf"
    if config.LOAD_GAN:
        latents = np.random.randn(1, config.GAN_INPUT_DIM_SIZE)
        # Again this trick in order to load weights.
        gan_model.fit(np.array(latents), np.array([1]), epochs=1, verbose=0)
        gan_model.load_weights(gan_file_name)
    else:
        gan_model.train(data_generator)
        gan_model.save_weights(gan_file_name)

    net = util.get_verification_model(datamode, data_generator)

    batch_size = 36

    # Generate 3 images and plot them.
    for _ in range(3):
        # Generate images
        latents = np.random.randn(batch_size, config.GAN_INPUT_DIM_SIZE)
        generated_images = generator_model.predict(latents, batch_size=batch_size)
        # Draw the generated images.
        draw.draw_images(generated_images, size=6)

    # This is for checking mode collapse
    cov = net.check_class_coverage(data=generated_images, tolerance=.8)
    pred, _ = net.check_predictability(data=generated_images)
    print(f"GAN - Generated images - Coverage: {100 * cov:.2f}%")
    print(f"GAN - Generated images - Predictability: {100 * pred:.2f}%")
    print("---------------------------------------------")


if __name__ == '__main__':
    main()
