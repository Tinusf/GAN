from stacked_mnist import DataMode

# Should the GAN be loaded from file
LOAD_GAN = False
# Which dataset to use for the GAN
GAN_DATAMODE = DataMode.MONO_BINARY_COMPLETE
# Should the GAN use batch normalization on the generator.
GAN_BATCHNORM_GEN = True
# Should the GAN use batch normalization on the disciminator.
GAN_BATCHNORM_DISC = False
# The dim size for the generator.
GAN_INPUT_DIM_SIZE = 128
# How many epochs
GAN_EPOCHS = 25
# Batch size
GAN_BATCH_SIZE = 512
# In order to reduce mode collapse you can use a classifier which adds loss when the generator
# creates images that predicts to the same class.
GAN_USE_CLASSIFIER = False
# Which folder to save all the figures in, both graph of loss and reconstructions.
GAN_FOLDER_GRAPHS = f"gan_graphs_{GAN_BATCH_SIZE}_{int(GAN_USE_CLASSIFIER)}" \
                    f"_{GAN_INPUT_DIM_SIZE}_{int(GAN_BATCHNORM_GEN)}_{GAN_DATAMODE.name}"

#
# Util
#
# If the verification net should be loaded from file or retrained.
LOAD_VERIFICATION_NET_MODEL = True
