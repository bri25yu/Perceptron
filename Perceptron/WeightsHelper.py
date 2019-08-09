import numpy as np


def load_weights(path_to_weights='', DIM=(28, 28)):
    if DIM is not None:
        DIM = DIM
    weights, symbols = {}, set()
    with open(path_to_weights + 'weights.txt', 'r') as file:
        lines = file.read()
        lines = lines.split('\n')
        for i in range(len(lines) // 2):
            symbol_i, symbol_j = lines[2 * i].split(':::')
            symbols.add(symbol_i)
            symbols.add(symbol_j)
            if weights.get(symbol_i) is None:
                weights[symbol_i] = {}
            new_weight = lines[2 * i + 1].split(' ')[:DIM[0]*DIM[1]]
            weights[symbol_i][symbol_j] = np.array(
                new_weight, np.float64)
    symbols = list(symbols)
    symbols.sort()
    return symbols, weights


def save_weights(symbols, weights, path_to_weights=''):
    print("Saving weights")
    with open(path_to_weights + 'weights.txt', 'w') as file:
        to_write = ''
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol_i, symbol_j = symbols[i], symbols[j]
                to_write += str(symbol_i) + r':::' + str(symbol_j) + '\n'
                for val in weights[symbol_i][symbol_j]:
                    to_write += str(val) + ' '
                to_write += '\n'
        file.writelines(to_write)
