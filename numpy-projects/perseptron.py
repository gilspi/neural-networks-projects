import numpy as np
import matplotlib.pyplot as plt


N = 5
B = 3


x1 = np.random.random(N)
x2 = x1 + [np.random.randint(10) / 10 for i in range(N)] + B
C1 = [x1, x2]
print(C1)

x1 = np.random.random(N)
x2 = x1 - [np.random.randint(10) / 10 for i in range(N)] - 0.1 + B
C2 = [x1, x2]

f = [0+B, 1+B]  # границы прямой от 0 до 4 включительно

w2 = 0.5
w3 = -B * w2
w = np.array([-w2, w2, w3])
print('W=', w)

for i in range(N):
    x = np.array([C1[0][i], C1[1][i], 1])
    print('X=', x)
    y = np.dot(x, w)
    print('Y=', y)
    if y >= 0:
        print('Class C1')
    else:
        print('Class C2')


plt.scatter(C1[0][:], C1[1][:], s=10, c='red')
plt.scatter(C2[0][:], C2[1][:], s=10, c='blue')
plt.plot(f)
plt.grid(True)
plt.show()

C1 = [(1, 0), (0, 1)]
C2 = [(0, 0), (1, 1)]
x = np.array([C1[0][0], C1[0][1], 1])
print(x)
