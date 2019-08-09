# System interfaces
import os
import multiprocessing

# Data manipulation tools
import numpy as np
import idx2numpy

# Visualization tools
import cv2


class Helper:
    """
    Just an encapsulation of independent helper functions
    """

    def load_data_from_MNIST(self, path_to_MNIST='../Data/'):
        print("Loading data from MNIST")
        images = idx2numpy.convert_from_file(
            path_to_MNIST + 'train-images.idx3-ubyte')
        labels = idx2numpy.convert_from_file(
            path_to_MNIST + 'train-labels.idx1-ubyte')
        data = {}
        for i in range(len(labels)):
            label, image = labels[i], images[i]
            if data.get(label) is None:
                data[label] = []
            data[label].append(image.flatten())
        symbols = list(data.keys())
        symbols.sort()
        return data, symbols

    def load_data(self, path, symbols):
        """
        Loads images from a text file of 0's and 1's into an images array
        of (image, symbol) tuples.

        @param path: Where the images are stored.
        @param symbols: Which symbols to load into an images array.
                        Default is None, which loads all the images and
                            symbols from a images folder.

        @return Returns a list of a symbol string and its images.
        """
        print("Loading data")
        if not symbols:
            symbols = os.listdir(path)
        args = [(path, symbol) for symbol in symbols]
        with multiprocessing.Pool(processes=10) as pool:
            results = pool.starmap(self.load_data_helper, args)
        data = {}
        for i in range(len(symbols)):
            data[symbols[i]] = results[i]
        return data, symbols

    def load_data_helper(self, path, symbol):
        """
        Helper function for loading data using multithreading!

        @param symbol: Symbol to load from dataset
        @return Returns an array of the symbol and the corresponding images
        """
        images = []
        image_names = os.listdir(os.path.join(path, symbol))
        for image_name in image_names:
            image = np.loadtxt(os.path.join(path, symbol, image_name))
            images.append(image.flatten())
        return images

    def value_to_pixel(self, value):
        """
        Converts a value in the weights array to its corresponding pixel
        """
        if abs(value) < 0.1:
            return np.array([255, 255, 255])
        elif value < 0:
            return np.array([255, 0, 0])
        else:
            return np.array([0, 0, 255])

    def weights_to_image(self, weights, DIM):
        """
        Converts a flat weights array into a DIM image to visualize.
        """
        image_vals = np.reshape(weights, DIM)
        min_val, max_val = np.amin(weights), np.amax(weights)
        if -min_val > max_val:
            image_vals /= -min_val
        else:
            image_vals /= max_val
        image = []
        for row in image_vals:
            image_row = []
            for col in row:
                image_row.append(self.value_to_pixel(col))
            image.append(image_row)
        return np.array(image)

    def shuffle(self, data, ratio=0.9):
        """
        Shuffles data into randomly selected training and testing data.

        @param data: data to sort into training/testing sets.
        @param ratio: Ratio between training and testing data.
                        Default is 0.9, which means 90% of data will be
                            training and 10% will be testing.

        @return Returns a tuple of arrays of tuples of training
            data/labels and testing data/labels, respectively.
        """
        length = len(data)
        testing_length = int(length*(1-ratio))
        randIndices = np.random.choice(
            length, testing_length, replace=False)
        training, testing = [], []
        for index in range(length):
            if index in randIndices:
                testing.append(data[index])
            else:
                training.append(data[index])
        return training, testing

    def test_accuracy(self, weights, testing_set_i, testing_set_j):
        """
        Tests the current accuracy of the weights on some testing data.
        """
        acc = 0
        for index in range(min(len(testing_set_i), len(testing_set_j))):
            if np.dot(testing_set_i[index], weights) > 0:
                acc += 1
            if np.dot(testing_set_j[index], weights) < 0:
                acc += 1
        return acc/2/min(len(testing_set_i), len(testing_set_j))

    def train_weights(self, weights, training_set_i, training_set_j):
        """
        Train weights off of training sets
        """
        for index in range(min(len(training_set_i), len(training_set_j))):
            if np.dot(training_set_i[index], weights) < 0:
                weights += training_set_i[index]
            if np.dot(training_set_j[index], weights) > 0:
                weights -= training_set_j[index]

    def convert_to_important(self, weights):
        index_sorted = np.argsort([-abs(val) for val in weights])

        def converter(num_important):
            important = np.zeros(weights.shape)
            indices = index_sorted[:num_important]
            for index in indices:
                important[index] = weights[index]
            return important
        return converter

    def get_test_pixel_indices(self, data, num):
        randIndices = np.random.choice(
            len(data[0])-1, num//2, replace=False)
        randIndices = randIndices.astype(int)
        randIndices += 1
        return randIndices

    def image_to_data(self, image_path, view):
        """
        Converts an input image of any size and resizes it to (28, 28).
        """
        assert image_path[-3:] == 'jpg', "Image must be a jpeg file"
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[0], img.shape[1]
        if h > w:
            padding = np.zeros((h, (h - w) // 2))
            padding += 255
            img = np.hstack((padding, img, padding))
        elif w > h:
            padding = np.zeros(((w - h) // 2, w))
            padding += 255
            img = np.vstack((padding, img, padding))
        img = cv2.resize(img, (28, 28))
        img = img.astype(np.uint8)
        show_image = np.where(img > 127, 255, 0)
        show_image = show_image.astype(np.uint8)
        cv2.imwrite(image_path[:len(image_path)-4] +
                    '_scaled.jpg', show_image)
        if view:
            cv2.imshow('img', show_image)
            cv2.waitKey()
        img = np.where(img > 127, 0, 1)
        img = img.flatten()
        return img
