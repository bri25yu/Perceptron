
import os, multiprocessing # General libraries
import numpy as np, pandas as pd # Data processing libraries
import matplotlib.pyplot as plt, seaborn as sns, cv2 # Plotting libraries
from multiprocessing import Pool, Process

class Perceptron:

    class Helper:
        """
        Just an encapsulation of independent helper functions
        """

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
            with multiprocessing.Pool(processes = 10) as pool:    
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

        def shuffle(self, data, ratio = 0.9):
            """
            Shuffles data into randomly selected training and testing data. 

            @param data: data to sort into training/testing sets
            @param ratio: Ratio between training and testing data. 
                            Default is 0.9, which means 90% of data will be training and 10% will be testing

            @return Returns a tuple of arrays of tuples of training data/labels and testing data/labels, respectively
            """
            length = len(data)
            testing_length = int(length*(1-ratio))
            randIndices = np.random.choice(length, testing_length, replace=False)
            training, testing = [], []
            for index in range(length):
                if index in randIndices:
                    testing.append(data[index])
                else:
                    training.append(data[index])
            return training, testing

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
            randIndices = np.random.choice(len(data[0])-1, num//2, replace=False)
            randIndices = randIndices.astype(int)
            randIndices += 1
            return randIndices

    def __init__(self, path_to_data = "images/", symbols = None, DATA_DIM = (28, 28)):
        self.helper = self.Helper()
        self.DIM = DATA_DIM
        self.path = path_to_data
        self.data, self.symbols = self.helper.load_data(self.path, symbols)

    def test_accuracy(self, weights, accuracies, testing_set_i, testing_set_j):
        """
        Tests the current accuracy of the weights on some testing data. 
        """
        acc = 0
        for index in range(min(len(testing_set_i), len(testing_set_j))):
            if np.dot(testing_set_i[index], weights) > 0:
                acc += 1
            if np.dot(testing_set_j[index], weights) < 0:
                acc += 1
        accuracies.append(acc/2/min(len(testing_set_i), len(testing_set_j)))

    def train_weights(self, symbol_i, symbol_j, training_set_i, training_set_j):
        """
        Train weights off of training sets
        """
        for index in range(min(len(training_set_i), len(training_set_j))):
            if np.dot(training_set_i[index], self.weights[symbol_i][symbol_j]) < 0:
                self.weights[symbol_i][symbol_j] += training_set_i[index]
            if np.dot(training_set_j[index], self.weights[symbol_i][symbol_j]) > 0:
                self.weights[symbol_i][symbol_j] -= training_set_j[index]

    def initialize_training_pair(self, symbol_i, symbol_j):
        """
        Initialize weights and accuracy for this specific (symbol_i, symbol_j) pair
        """
        if not self.weights.get(symbol_i):
            self.weights[symbol_i] = {}
        self.weights[symbol_i][symbol_j] = np.random.normal(0, 1, size = self.DIM[0] * self.DIM[1])
        if not self.accuracy.get(symbol_i):
            self.accuracy[symbol_i] = {}
        self.accuracy[symbol_i][symbol_j] = []

    def train_on_symbols(
                            self, 
                            symbol_i, 
                            symbol_j, 
                            ITERATIONS, 
                            RATIO
                        ):
        self.initialize_training_pair(symbol_i, symbol_j)
        for _ in range(ITERATIONS):
            training_set_i, testing_set_i = self.helper.shuffle(self.data[symbol_i], RATIO)
            training_set_j, testing_set_j = self.helper.shuffle(self.data[symbol_j], RATIO)
            self.test_accuracy(
                                self.weights[symbol_i][symbol_j], 
                                self.accuracy[symbol_i][symbol_j], 
                                testing_set_i, 
                                testing_set_j
                              )
            self.train_weights(
                                symbol_i, 
                                symbol_j, 
                                training_set_i, 
                                training_set_j
                              )
        self.pair_accuracy.append([symbol_i, symbol_j, self.accuracy[symbol_i][symbol_j][-1]])

    def train(self, ITERATIONS = 10, RATIO = 0.9, USE_MULTIPROCESSING = False):
        """
        Works for 2-dimensional images
        """
        self.weights = {}
        self.accuracy, self.pair_accuracy = {}, []
        print("Training")
        args = []
        if USE_MULTIPROCESSING:
            for i in range(len(self.symbols)):
                for j in range(i+1, len(self.symbols)):
                    args.append((
                                    self.symbols[i], 
                                    self.symbols[j], 
                                    ITERATIONS, 
                                    RATIO
                                ))
            with multiprocessing.Pool(processes=9) as pool:
                pool.starmap(self.train_on_symbols, args)
        else:
            for i in range(len(self.symbols)):
                print('Training symbol pairs including: {}'.format(self.symbols[i]))
                for j in range(i+1, len(self.symbols)):
                    self.train_on_symbols(
                                            self.symbols[i], 
                                            self.symbols[j], 
                                            ITERATIONS, 
                                            RATIO
                                         )

    def plot_accuracies(
                        self, 
                        symbol_i, 
                        symbol_j, 
                        view = True,
                        path = ''
                       ):
        print("Plotting accuracies for {0} vs {1}".format(symbol_i, symbol_j))
        plt.plot(list(range(len(self.accuracy[symbol_i][symbol_j]))), self.accuracy[symbol_i][symbol_j])
        plt.savefig("iterations_vs_acc_{0}_{1}.jpg".format(symbol_i, symbol_j))
        if view:
            plt.show()
            plt.pause(3)
        plt.close()

    def visualize_weights(
                            self, 
                            symbol_i, 
                            symbol_j, 
                            view = True, 
                            path = ''
                         ):
        """
        Plots the "importance" of each pixel in a comparison between two symbols. 
        """
        print("Visualizing weights")
        image = self.helper.weights_to_image(self.weights[symbol_i][symbol_j], self.DIM)
        image = image.astype(np.uint8)
        cv2.imwrite('weights_{0}_{1}.png'.format(symbol_i, symbol_j), image)
        if view:
            cv2.imshow('Weights for classifying {0} vs {1}'.format(symbol_i, symbol_j), image)
            cv2.waitKey()

    def test_pixel_values(
                            self, 
                            symbol_i, 
                            symbol_j, 
                            TEST_SIZE = 1000, 
                            view = True, 
                            path = ''
                         ):
        print("Testing pixel values")
        N = self.DIM[0] * self.DIM[1]
        NUM_IMPORTANT = [N - i*10 for i in range(1, N//10)]
        accuracies = []
        converter = self.helper.convert_to_important(self.weights[symbol_i][symbol_j])
        for num in NUM_IMPORTANT:
            acc = 0
            important = converter(num)
            randIndices = self.helper.get_test_pixel_indices(self.data[symbol_i], TEST_SIZE)
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

    def plot_pairwise_accuracies(self, view = True, path = ''):
        print("Plotting pairwise accuracies")
        xLabels = [pair[0] for pair in self.pair_accuracy]
        yLabels = [pair[1] for pair in self.pair_accuracy]
        vals = [pair[2] for pair in self.pair_accuracy]
        df = pd.DataFrame({
                            'xLabels': xLabels, 
                            'yLabels': yLabels, 
                            'Values': vals
                        })
        df = df.pivot(index = 'xLabels', columns = 'yLabels', values = 'Values')
        ax = sns.heatmap(df)
        plt.savefig(path + 'pairwise_accuracies.jpg')
        if view:
            plt.show()
            plt.pause(3)
        plt.close()

    def predict(self, image_path, view = True):
        assert image_path[-3:] == 'jpg', "Image must be a jpeg file"
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        show_image = np.where(img > 127, 255, 0)
        cv2.imwrite(image_path[:len(image_path)-4] + '_scaled.jpg', show_image)
        if view:
            cv2.imshow('img', show_image)
            cv2.waitKey()
        img = np.where(img > 127, 0, 1)        
        img = img.flatten()
        predictions = []
        for symbol_i in self.weights.keys():
            for symbol_j in self.weights[symbol_i].keys():
                val = np.dot(self.weights[symbol_i][symbol_j], img)
                print(symbol_i, symbol_j, val)
                if val > 0:
                    predictions.append(symbol_i)
                else:
                    predictions.append(symbol_j)
        print(predictions)
        aggregate_predictions = {}
        for symbol in self.symbols:
            aggregate_predictions[symbol] = 0
        for prediction in predictions:
            aggregate_predictions[prediction] += 1
        print(aggregate_predictions)
        print(max(aggregate_predictions.keys(), key = lambda key: aggregate_predictions[key]))

def main():
    p = Perceptron()
    p.train(10)
    p.predict('zero.jpg', view = False)

if __name__ == '__main__':
    main()
