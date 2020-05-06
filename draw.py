import numpy as np
import matplotlib.pyplot as plt


def predict_and_draw(model, unseen_data, labels, mult_255=False):
    """ Predicts the unseen data and plots the images first the unseens and then the
    reconstruction images. """
    predicted = model.predict(unseen_data)
    draw_images(unseen_data, labels, mult_255=mult_255)
    draw_images(predicted, labels, mult_255=mult_255)


def draw_images(images, labels=None, mult_255=False, save_to_file=False, filename=None, size=4):
    """ Draws the images into a nice grid. """
    fig = plt.figure(figsize=(8, 8))
    for i in range(images.shape[0]):
        fig.add_subplot(size, size, i + 1)

        # on the numpy array index starts with 0 like normal.
        if mult_255:
            image = np.array(images[i] * 255)
        else:
            image = np.array(images[i])

        if image.shape[-1] == 1:
            # Greyscale, then we need to remove the last dimension.
            image = image.reshape(*image.shape[:-1])
        plt.imshow(image)
        if labels is not None:
            label = labels[i]
            plt.title(label)
    if save_to_file:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)


