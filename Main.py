from Perceptron.Perceptron import Perceptron


def main():
    p = Perceptron(path_to_data='MNIST')
    p.load_weights('./Output/')
    p.predict('./Data/one.jpg', view=False)
    p.test_pixel_values(1, 8)
    p.plot_pairwise_accuracies(1, 8)


if __name__ == '__main__':
    main()
