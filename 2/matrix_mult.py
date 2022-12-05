"""matrix multiplication"""
import argparse
import copy


def mult(a_mat, b_mat):
    """Multiplies two matrixes.
    :param a_mat: First matrix to multiply.
    :param b_mat: Second matrix to multiply by.
    :return: Returns a_matrix new matrix
    """
    mat_res = []
    string = []
    if len(a_mat[0]) != len(b_mat):
        raise Exception("Matrixes are not compatible. Check the dimensions.")
    for x_ax in range(len(a_mat)):
        for y_ax in range(len(b_mat[0])):
            string.append(sum(a_mat[x_ax][kk] * b_mat[kk][y_ax] for kk in range(len(b_mat))))
        mat_res.append(string)
        string = []
    return mat_res


def conv(a_matrix, b_matrix):
    """
    Convolves matrix a with matrix b_matrix

    :param a_matrix: Matrix to convolve
    :param b_matrix: Activation core matrix
    :return: Convoluted matrix a
    """
    a_matrix_copy = copy.deepcopy(a_matrix)
    b_copy = copy.deepcopy(b_matrix)
    m_res = []
    size_a = [len(a_matrix), len(a_matrix[0])]
    size_b = [len(b_matrix), len(b_matrix[0])]
    step_x = 0
    step_y = 0
    mat_item = 0
    mat_col = []
    res_size = max([size_a[0] - max(0, size_b[0]-1), size_a[1] - max(0, size_b[1]-1)], [0, 0])
    for p in range(res_size[0]):
        for o in range(res_size[0]):
            for x_ in range(step_x, size_b[0] + step_x):
                for y_ in range(step_y, size_b[1] + step_y):
                    if size_b[1] + step_y >= size_a[1] and y_ == size_a[1]:
                        step_y = 0
                        break
                    b_copy[x_-step_x][y_-step_y] = a_matrix_copy[x_][y_] * b_matrix[x_ - step_x][y_ - step_y]
                    mat_item += b_copy[x_-step_x][y_-step_y]
            mat_col.append(mat_item)
            mat_item = 0
            step_y += 1
            b_copy = copy.deepcopy(b_matrix)
        step_y = 0
        step_x += 1
        m_res.append(mat_col)
        mat_col = []

    return m_res


parser = argparse.ArgumentParser(description="Multiplies 2 matrixes from the file")
parser.add_argument('PATH', type=str, help="Absolute path to the file with matrixes")
args = parser.parse_args()

PATH = args.PATH


with open(PATH, "r", encoding='utf-8') as f:
    m1 = []
    m2 = []
    m = m1
    for line in f.readlines():
        f_lst = line.split()
        f_lst = [int(x) for x in f_lst]
        if line == "\n":
            m = m2
            continue
        m.append(f_lst)

res = conv(m1, m2)

with open("matrix_res.txt", 'w', encoding='utf-8') as xfile:
    for L in res:
        STRING = ''
        llen = len(L)
        for T in range(llen):
            STRING += str(L[T]) + ' '
        STRING += '\n'
        xfile.writelines(STRING)
