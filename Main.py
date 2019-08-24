from Perceptron.Perceptron import Perceptron


def main():
    p = Perceptron(path_to_data='MNIST')
    p.load_weights('Output/')
    p.predict('Data/one.jpg', view=False)
    p.test_pixel_values(0, 8)


if __name__ == '__main__':
    main()
