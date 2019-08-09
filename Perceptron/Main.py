from Perceptron import Perceptron


def main():
    p = Perceptron(path_to_data='MNIST')
    p.load_weights('../Output/')
    p.predict('../Data/zero.jpg', view=False)
    # p.visualize_weights('0', '8')


if __name__ == '__main__':
    main()
