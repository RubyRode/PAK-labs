"""jsdhfs"""
import numpy as np


def np_sort(npm):
    """Sorts an array by frequency of appearance of the elements"""
    uni, pos = np.unique(npm, return_inverse=True)  # список уникальных и их индексы
    counts = np.bincount(pos)  # список количества индексов в списке индексов
    max_sort = counts.argsort()[::-1]  # сортировка по убыванию
    res = uni[max_sort]  # сортировка уникальных значений по индексам
    return res


def search_unique(arr):
    """Searches the unique values"""
    res = len(np.unique(arr))
    return res


def float_mid_vector(vec, n):
    """Searches floating middle of the vector"""

    tmp = [0] * (n - 1)
    vec_tmp = vec
    for zero in tmp:
        vec_tmp.append(zero)
        vec_tmp.insert(0, zero)
    cumsum = np.cumsum(vec_tmp)
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def is_triangle(arr):
    """Writes triplets if they can make a triangle"""
    for triple in arr:
        if triple[0] + triple[1] > triple[2] and triple[0] + triple[2] > triple[1] and triple[1] + triple[2] > triple[
            0]:
            print(triple)


mat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(float_mid_vector(mat, 4))
