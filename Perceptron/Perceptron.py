# Data manipulation tools
import numpy as np
import pandas as pd

# Visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Custom imports
import Helper
import WeightsHelper
import VisualizationHelper


class Perceptron:

    def __init__(
        self,
        path_to_data='images/',
        symbols=None,
        DATA_DIM=(28, 28)
    ):
        self.DIM = DATA_DIM
        if path_to_data is None:
            self.path = None
        else:
            self.path = path_to_data
            if path_to_data == 'MNIST':
                self.data, self.symbols = Helper.load_data_from_MNIST()
            else:
                self.data, self.symbols = Helper.load_data(
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
                    training_set_i, testing_set_i = Helper.shuffle(
                        self.data[symbol_i], RATIO)
                    training_set_j, testing_set_j = Helper.shuffle(
                        self.data[symbol_j], RATIO)
                    acc = Helper.test_accuracy(
                        self.weights[symbol_i][symbol_j],
                        testing_set_i,
                        testing_set_j
                    )
                    self.accuracy[symbol_i][symbol_j].append(acc)
                    Helper.train_weights(
                        self.weights[symbol_i][symbol_j],
                        training_set_i,
                        training_set_j
                    )
                self.pair_accuracy.append([
                    symbol_i,
                    symbol_j,
                    self.accuracy[symbol_i][symbol_j][-1]])

    def plot_accuracies(self, symbol_i, symbol_j, view=True, path=''):
        print("Plotting accuracies for {0} vs {1}".format(symbol_i, symbol_j))
        plt.plot(list(range(len(
            self.accuracy[symbol_i][symbol_j]))),
            self.accuracy[symbol_i][symbol_j])
        plt.savefig("iterations_vs_acc_{0}_{1}.jpg".format(symbol_i, symbol_j))
        if view:
            plt.show()
            plt.pause(3)
        plt.close()

    def visualize_weights(
        self,
        symbol_i,
        symbol_j,
        view=True,
        path='../Output/'
    ):
        """
        Plots the "importance" of each pixel between two symbols.
        """
        print("Visualizing weights")
        image = Helper.weights_to_image(
            self.weights[symbol_i][symbol_j], self.DIM)
        image = image.astype(np.uint8)
        cv2.imwrite(
            path + 'weights_{0}_{1}.png'.format(symbol_i, symbol_j),
            image)

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
        path='../Output/'
    ):
        """
        Generates a plot of exactly how many pixels are impactful in the
            recognition between symbol_i and symbol_j.
        """
        print("Testing pixel values")
        N = self.DIM[0] * self.DIM[1]
        NUM_IMPORTANT = [N - i * 10 for i in range(1, N // 10)]
        accuracies = []
        converter = Helper.convert_to_important(
            self.weights['{}'.format(symbol_i)]['{}'.format(symbol_j)])
        for num in NUM_IMPORTANT:
            acc = 0
            important = converter(num)
            randIndices = Helper.get_test_pixel_indices(
                self.data[symbol_i], TEST_SIZE)
            for index in randIndices:
                if np.dot(self.data[symbol_i][index], important) > 0:
                    acc += 1
                if np.dot(self.data[symbol_j][index], important) < 0:
                    acc += 1
            accuracies.append(acc/TEST_SIZE)

        plt.plot(NUM_IMPORTANT, accuracies, 'ro')
        plt.xlabel('Number of important pixels')
        plt.ylabel('Accuracy')

        plt.savefig(
            path + 'features_vs_acc_{0}_{1}.png'.format(symbol_i, symbol_j))
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
        img = Helper.image_to_data(image_path, view)
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
        print("Overall prediction: {}".format(max(
            aggregate_predictions.keys(),
            key=lambda key: aggregate_predictions[key])))

    def load_weights(self, path='', DIM=(28, 28)):
        self.symbols, self.weights = WeightsHelper.load_weights(path, DIM)

    def save_weights(self, path_to_weights=''):
        WeightsHelper.save_weights(self.symbols, self.weights)
