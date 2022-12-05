"""Bubble sorting realisation"""
import random as r
import argparse as agp


def bubble_sort(lst):
    """bubble sorting"""
    length = len(lst)
    for k in range(length - 1):
        for j in range(length - k - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]


pr = agp.ArgumentParser(description="Simple bubble sorting of a_matrix random list")
pr.add_argument('len', type=int, help="Length of the list to be sorted")
args = pr.parse_args()

LENGTH = args.len
LST = []
for i in range(0, LENGTH):
    LST.append(r.randint(1, 100))
bubble_sort(LST)
print(*LST)
