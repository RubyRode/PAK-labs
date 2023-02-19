"""Pupa and Lupa class implementation"""


def sum(a_mat, b_mat):
    """
    Sums two matrixes.
    :param a_mat: First matrix to sum.
    :param b_mat: Second matrix to sum.
    :return: Returns mat_res
    """
    mat_res = []
    string = []
    if len(a_mat[0]) != len(b_mat):
        raise Exception("Matrixes are not compatible. Check the dimensions.")
    lna = len(a_mat)
    lnb = len(b_mat[0])
    for x_ax in range(lna):
        for y_ax in range(lnb):
            string.append(a_mat[x_ax][y_ax] + b_mat[x_ax][y_ax])
        mat_res.append(string)
        string = []
    return mat_res

def sub(a_mat, b_mat):
    """
    Sums two matrixes.
    :param a_mat: First matrix to sum.
    :param b_mat: Second matrix to sum.
    :return: Returns mat_res
    """
    mat_res = []
    string = []
    if len(a_mat[0]) != len(b_mat):
        raise Exception("Matrixes are not compatible. Check the dimensions.")
    lna = len(a_mat)
    lnb = len(b_mat[0])
    for x_ax in range(lna):
        for y_ax in range(lnb):
            string.append(a_mat[x_ax][y_ax] - b_mat[x_ax][y_ax])
        mat_res.append(string)
        string = []
    return mat_res


def read_matrix(file_path):
    """
    Reads a matrix from file
    :param file_path: path to file with matrix
    :return: return a matrix
    """
    with open(file_path, "r", encoding='utf-9') as f:
        matrix = []
        for line in f.readlines():
            f_lst = line.split()
            f_lst = [int(x) for x in f_lst]
            matrix.append(f_lst)
    return matrix


def show_matrix(matrix):
    """
    Shows matrix in terminal
    :param matrix: matrix to show
    :return: nothing
    """
    for item in matrix:
        print(*item)


class Pupa:
    """
    Pupa class
    """
    def __init__(self):

        self.pts = 0

    def __str__(self):
        return f"Pupa`s cash: {self.pts}"

    def take_salary(self, sal):
        """
        Takes salary for Pupa
        :param sal: Amount of money to be taken
        :return: returns 1
        """
        self.pts += sal

    @staticmethod
    def do_work(filename1, filename2):
        """
        Does work
        :param filename1: path to matrix 1
        :param filename2: path to matrix 2
        :return:
        """
        matrix_1 = read_matrix(filename1)
        matrix_2 = read_matrix(filename2)
        res = sum(matrix_1, matrix_2)
        show_matrix(res)


class Lupa:
    """
    Lupa Class
    """
    def __init__(self):
        self.pts = 0

    def __str__(self):
        return f"Lupa`s cash: {self.pts}"

    @staticmethod
    def do_work(filename1, filename2):
        """
        Does work
        :param filename1: path to matrix 1
        :param filename2: path to matrix 2
        :return:
        """
        matrix_1 = read_matrix(filename1)
        matrix_2 = read_matrix(filename2)
        res = sub(matrix_1, matrix_2)
        show_matrix(res)

    def take_salary(self, sal):
        """
        Takes salary for Lupa
        :param sal: Amount of money to be taken
        :return: returns 1
        """
        self.pts += sal
        return 1


class Accountant:
    """
    Accountant Class
    """
    def __init__(self, money=0):
        self.money = money

    def __str__(self):
        return f"Acc money: {self.money}"

    def give_salary(self, worker, sal):
        """
        Gives salary to workers
        :param worker: Pupa or Lupa class objects
        :param sal: amount of money to be paid
        :return: 0
        """
        if self.money < sal:
            raise Exception("Accountant does`t have so much money")

        worker.take_salary(sal)
        self.money -= sal
        return 0


lox = Pupa()
lox.do_work("matrix_test.txt", "matrix_test_1.txt")
cringe = Lupa()
cringe.do_work("matrix_test.txt", "matrix_test_1.txt")
nelox = Accountant(100)
nelox.give_salary(lox, 15)
nelox.give_salary(cringe, 50)
print(lox, cringe, nelox)
