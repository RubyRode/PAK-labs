"""Pascal triangle realization"""
import argparse as agp

parser = agp.ArgumentParser(description="Shows the binomial coefficients as a_matrix pascal triangle")
parser.add_argument('height', type=int, help="Height of the triangle")
args = parser.parse_args()

HG = args.height
TRIANGLE = [[0] * HG for i in range(HG)]
TRIANGLE[0][0] = 1

for i in range(1, HG):
    TRIANGLE[i][0] = 1
    TRIANGLE[i][i] = 1
for i in range(2, HG):
    for j in range(1, HG):
        TRIANGLE[i][j] = TRIANGLE[i-1][j-1] + TRIANGLE[i-1][j]


zero_count = HG
for i in range(HG-1):
    j = 0
    print('   ' * zero_count, end='')
    while TRIANGLE[i][j] != 0 and j < HG:
        print(TRIANGLE[i][j], end=' ' * (6 - len(str(TRIANGLE[i][j]))))
        j += 1
    print("\n")
    zero_count -= 1
