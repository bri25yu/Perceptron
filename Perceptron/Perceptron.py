# System interfaces
import os
import multiprocessing

# Data manipulation tools
import numpy as np
import pandas as pd
import idx2numpy

# Visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


class Perceptron:

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
                if data.get(label) == None:
                    data[label] = []
                data[label].append(image.flatten())
            symbols = list(data.keys())
            symbols.sort()
            return data, symbols

        def load_data(self, path, symbols):
            """
            Loads images from a text file of 0's and 1's into an images array of (image, symbol) tuples. 

            @param path: Where the images are stored. 
            @param symbols: Which symbols to load into an images array. 
                            Default is None, which loads all the images and symbols from a images folder. 

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

            @param data: data to sort into training/testing sets
            @param ratio: Ratio between training and testing data. 
                            Default is 0.9, which means 90% of data will be training and 10% will be testing

            @return Returns a tuple of arrays of tuples of training data/labels and testing data/labels, respectively
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
            Converts an input image of any size and resizes it to (28, 28) properly
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

    def __init__(self, path_to_data='images/', symbols=None, DATA_DIM=(28, 28)):
        self.helper = self.Helper()
        self.DIM = DATA_DIM
        if path_to_data == None:
            self.path = None
        else:
            self.path = path_to_data
            if path_to_data == 'MNIST':
                self.data, self.symbols = self.helper.load_data_from_MNIST()
            else:
                self.data, self.symbols = self.helper.load_data(
                    self.path, symbols)

    def train(self, ITERATIONS=10, RATIO=0.9):
        """
        Works for 2-dimensional images. 
        USE_MULTIPROCESSING is currently not working!
        """
        self.weights = {}
        self.accuracy, self.pair_accuracy = {}, []
        args = []
        for i in range(len(self.symbols)):
            print('Training symbol pairs including: {}'.format(
                self.symbols[i]))
            for j in range(i+1, len(self.symbols)):
                # Initialize the new training set.
                symbol_i, symbol_j = self.symbols[i], self.symbols[j]
                if not self.weights.get(symbol_i):
                    self.weights[symbol_i] = {}
                self.weights[symbol_i][symbol_j] = np.random.normal(
                    0, 1, size=self.DIM[0] * self.DIM[1])
                if not self.accuracy.get(symbol_i):
                    self.accuracy[symbol_i] = {}
                self.accuracy[symbol_i][symbol_j] = []

                for _ in range(ITERATIONS):
                    training_set_i, testing_set_i = self.helper.shuffle(
                        self.data[symbol_i], RATIO)
                    training_set_j, testing_set_j = self.helper.shuffle(
                        self.data[symbol_j], RATIO)
                    acc = self.helper.test_accuracy(
                        self.weights[symbol_i][symbol_j],
                        testing_set_i,
                        testing_set_j
                    )
                    self.accuracy[symbol_i][symbol_j].append(acc)
                    self.helper.train_weights(
                        self.weights[symbol_i][symbol_j],
                        training_set_i,
                        training_set_j
                    )
                self.pair_accuracy.append(
                    [symbol_i, symbol_j, self.accuracy[symbol_i][symbol_j][-1]])

    def plot_accuracies(self, symbol_i, symbol_j, view=True, path=''):
        print("Plotting accuracies for {0} vs {1}".format(symbol_i, symbol_j))
        plt.plot(list(range(
            len(self.accuracy[symbol_i][symbol_j]))), self.accuracy[symbol_i][symbol_j])
        plt.savefig("iterations_vs_acc_{0}_{1}.jpg".format(symbol_i, symbol_j))
        if view:
            plt.show()
            plt.pause(3)
        plt.close()

    def visualize_weights(self, symbol_i, symbol_j, view=True, path=''):
        """
        Plots the "importance" of each pixel in a comparison between two symbols. 
        """
        print("Visualizing weights")
        image = self.helper.weights_to_image(
            self.weights[symbol_i][symbol_j], self.DIM)
        image = image.astype(np.uint8)
        cv2.imwrite('weights_{0}_{1}.png'.format(symbol_i, symbol_j), image)
        if view:
            cv2.imshow('Weights for classifying {0} vs {1}'.format(
                symbol_i, symbol_j), image)
            cv2.waitKey()

    def test_pixel_values(
        self,
        symbol_i,
        symbol_j,
        TEST_SIZE=1000,
        view=True,
        path=''
    ):
        """
        Generates a plot of exactly how many pixels are impactful in the recognition between symbols. 
        """
        print("Testing pixel values")
        N = self.DIM[0] * self.DIM[1]
        NUM_IMPORTANT = [N - i*10 for i in range(1, N//10)]
        accuracies = []
        converter = self.helper.convert_to_important(
            self.weights[symbol_i][symbol_j])
        for num in NUM_IMPORTANT:
            acc = 0
            important = converter(num)
            randIndices = self.helper.get_test_pixel_indices(
                self.data[symbol_i], TEST_SIZE)
            for index in randIndices:
                if np.dot(self.data[symbol_i][index], important) > 0:
                    acc += 1
                if np.dot(self.data[symbol_j][index], important) < 0:
                    acc += 1
            accuracies.append(acc/TEST_SIZE)
        plt.plot(NUM_IMPORTANT, accuracies, 'ro')
        plt.savefig('features_vs_acc_{0}_{1}.jpg'.format(symbol_i, symbol_j))
        if view:
            plt.show()
            plt.pause(3)
        plt.close()

    def plot_pairwise_accuracies(self, view=True, path=''):
        """
        Generates a heatmap of all the pairwise symbol accuracies. 
        """
        print("Plotting pairwise accuracies")
        xLabels = [pair[0] for pair in self.pair_accuracy]
        yLabels = [pair[1] for pair in self.pair_accuracy]
        vals = [pair[2] for pair in self.pair_accuracy]
        df = pd.DataFrame({
            'xLabels': xLabels,
            'yLabels': yLabels,
            'Values': vals
        })
        df = df.pivot(index='xLabels', columns='yLabels', values='Values')
        ax = sns.heatmap(df)
        plt.savefig(path + 'pairwise_accuracies.jpg')
        if view:
            plt.show()
            plt.pause(3)
        plt.close()

    def predict(self, image_path, view=True):
        img = self.helper.image_to_data(image_path, view)
        predictions = []
        for symbol_i in self.weights.keys():
            for symbol_j in self.weights[symbol_i].keys():
                val = np.dot(self.weights[symbol_i][symbol_j], img)
                if val > 0:
                    predictions.append(symbol_i)
                else:
                    predictions.append(symbol_j)
        aggregate_predictions = {}
        for symbol in self.symbols:
            aggregate_predictions[symbol] = 0
        for prediction in predictions:
            aggregate_predictions[prediction] += 1
        print("Aggregate predictions: \n{}".format(aggregate_predictions))
        print("Overall prediction: {}".format(
            max(aggregate_predictions.keys(), key=lambda key: aggregate_predictions[key])))

    def load_weights(self, path_to_weights='', DIM=(28, 28)):
        if DIM is not None:
            self.DIM = DIM
        self.weights, self.symbols = {}, set()
        with open(path_to_weights + 'weights.txt', 'r') as file:
            lines = file.read()
            lines = lines.split('\n')
            for i in range(len(lines) // 2):
                symbol_i, symbol_j = lines[2 * i].split(':::')
                self.symbols.add(symbol_i)
                self.symbols.add(symbol_j)
                if self.weights.get(symbol_i) is None:
                    self.weights[symbol_i] = {}
                new_weight = lines[2 * i + 1].split(' ')[:DIM[0]*DIM[1]]
                self.weights[symbol_i][symbol_j] = np.array(
                    new_weight, np.float64)
        self.symbols = list(self.symbols)
        self.symbols.sort()

    def save_weights(self, path_to_weights=''):
        print("Saving weights")
        with open(path_to_weights + 'weights.txt', 'w') as file:
            to_write = ''
            for i in range(len(self.symbols)):
                for j in range(i+1, len(self.symbols)):
                    symbol_i, symbol_j = self.symbols[i], self.symbols[j]
                    to_write += str(symbol_i) + r':::' + str(symbol_j) + '\n'
                    for val in self.weights[symbol_i][symbol_j]:
                        to_write += str(val) + ' '
                    to_write += '\n'
            file.writelines(to_write)


def main():
    p = Perceptron(path_to_data='MNIST')
    p.predict('../Data/zero.jpg', view=False)


if __name__ == '__main__':
    main()
