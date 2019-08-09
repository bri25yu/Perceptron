from Perceptron import Perceptron


def main():
    p = Perceptron(path_to_data='MNIST')
    p.load_weights('../Output/')
    p.predict('../Data/zero.jpg', view=False)


if __name__ == '__main__':
    main()
