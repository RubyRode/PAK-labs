"""Implements a random selection from 2 data sets."""
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Returns a mixture of 2 arrays.")
parser.add_argument("Filepath1", type=str, help="Path to file with real data.")
parser.add_argument("Filepath2", type=str, help="Path to file with synthetic data.")
parser.add_argument("P", type=float, help="Probability of mixing.")
args = parser.parse_args()

f_path_1 = args.Filepath1
f_path_2 = args.Filepath2
P = args.P


def read_file(path):
    """Reads and returns an array from file."""
    with open(path, "r", encoding="UTF-8") as file:
        return np.array(list(map(int, file.readline().split())))


if __name__ == "__main__":
    a = read_file(f_path_1)
    b = read_file(f_path_2)
    S = len(str(P)) - 2
    P = P * (10 ** S)
    d = np.random.randint(10 ** S - 1, size=(len(a))) + 1
    c_bool = d <= P
    a[c_bool] = b[c_bool]
    print(a)

    print(np.where(d <= P, b, a))
