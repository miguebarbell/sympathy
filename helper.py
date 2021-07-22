import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


class Image2ArrayPP:
    """process an image to channel last ordering (by default) and returns a 3D Numpy array.
    dataformat: str() can be either "channels_first" or "channels_last"
    """

    def __init__(self, dataformat=None):
        self.data_format = dataformat

    def pp(self, image):
        """
        Process an image to channel last ordering (by default) and returns a 3D Numpy array.
        Parameters
        ----------
        image: PIL image

        Returns
        -------
        Numpy array
        """
        return img_to_array(image, data_format=self.data_format)


class ImagePP:
    """resize an image ignoring the aspect ratio"""

    def __init__(self, width=32, height=32, inter=cv2.INTER_AREA, gray=False):
        self.width = width
        self.height = height
        self.inter = inter
        self.gray = gray

    def pp(self, image, aspect_ratio=False):
        """
        return a resized image to a specific pixel length. can return the resized image with the same aspect ratio (True)
        if the ratio is diffferent, it will take the longest side of the new, and resize the new image respect to it.
        Parameters
        ----------
        image: PIL image

        Returns
        -------
        numpy array
        """
        if aspect_ratio:
            if self.height > self.width:
                # se debe recortar el ancho al mismo ratio de la imagen
                sobrante = (image.shape[1] - ((image.shape[0] * self.width) / self.height)) / 2
                # round down and convert to integer
                sobrante = int(round(sobrante, 0) if round(sobrante, 0) < sobrante else round(sobrante, 0) - 1)
                image = image[0:image.shape[0], sobrante:image.shape[1] - sobrante]
            elif self.height < self.width:
                # se debe recortar el ancho al mismo ratio de la imagen
                sobrante = (image.shape[0] - ((image.shape[1] * self.height) / self.width)) / 2
                # round down and convert to integer
                sobrante = int(round(sobrante, 0) if round(sobrante, 0) < sobrante else round(sobrante, 0) - 1)
                image = image[sobrante:image.shape[0] - sobrante, 0:image.shape[1]]
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


class DataLoader:
    """
    load dataset of images and extract the labels by path (every image must be in the folder named with the label
    """

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths, verbose=-1):
        """
        load the images, preprocess it and store it in np.array. load the label in np.array.
        its integrated with preprocessing, you can make a list of preprocessing for the loading
        Parameters
        ----------
        image_paths: str

        Returns
        -------
        tuple, with two np.array ([data], [labels])
        """
        data = []
        labels = []

        for (i, image_path) in enumerate(image_paths):
            # load the image and extract the label
            # /path/to/dataset/{label}/{image}.jpg
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.pp(image)
            data.append(image)
            labels.append(label)
            # print an update every verbose qty
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] processed {i + 1} / {len(image_paths)}")
        return (np.array(data), np.array(labels))


def save_plot(H, path):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(path)


def check_imbalanced_classes(labels, threshold=0.9):
    print('[INFO] checking for imbalances in dataset')
    imbalance = 1
    max_qty = 0
    # find the class with more occurrence
    for label in np.unicode(labels):
        if max_qty < np.sum(labels == label):
            max_qty = np.sum(labels == label)
    # max_qty2 = [sum(label == labels) for label in np.unicode(labels) if max_qty < sum(label == labels)]
    # compare every class with the max of occurrence
    for label in np.unicode(labels):
        if np.sum(labels == label)/max_qty < imbalance:
            imbalance = np.sum(labels == label) / max_qty
    if imbalance < threshold:
        print('[INFO] imbalances found')
        return imbalance
    print('[INFO] imbalances passed the threshold')
    return False


def convert_and_trim_bb(image, rect):
    """

    Parameters
    ----------
    image
    rect

    Returns
    -------

    """
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return (startX, startY, w, h)
