# Data manipulation tools
import numpy as np
import pandas as pd

# Visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Custom imports
import Helper


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
