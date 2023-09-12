import numpy as np
import matplotlib.pyplot as plt


def act(x):
    return 0 if x <= 0 else 1


def start(C):
    x = np.array([C[0], C[1], 1])
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hid = np.array([w1, w2])
    w_out = np.array([-1, 1, -0.5])

    sum_hid = np.dot(w_hid, x)
    sum_out = [act(x) for x in sum_hid]
    sum_out.append(1)
    sum_out = np.array(sum_out)

    sum = np.dot(sum_out, w_out)
    y = act(sum)

    return y


C1 = [(1, 0), (0, 1)]
C2 = [(0, 0), (1, 1)]

print(start(C1[0]), start(C1[1]))
print(start(C2[0]), start(C2[1]))

