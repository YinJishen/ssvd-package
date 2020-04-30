import sys
sys.path.append("..")

from ssvd_pkg.ssvd import pos_new, sign_new, ssvd_new, ssvd_approx
import numpy as np
import scipy.linalg as la

a = np.array([1, -1, 1, 0])
print(pos_new(a))

a = np.array([2, 3, 0, -2, -3])
print(sign_new(a))

u = np.array([10, 9, 8, 7, 6, 5, 4, 3]+[2]*17+[0]*75).reshape(-1, 1)
u = u / la.norm(u)
v = np.array([10, -10, 8, -8, 5, -5]+[3]*5+[-3]*5+[0]*34).reshape(1, -1)
v = v / la.norm(v)
X1_ = 50 * u @ v
X1 = X1_ + np.random.normal(size=(100, 50));
u, v, s = ssvd_new(X1)
print(u)
print(v)
print(s)

X2_ = np.zeros(shape=(50, 100))
for i in range(50):
    for j in range(25, 75):
        T = (24**2-(i-24)**2-(j-49)**2)/100
        if abs(T) > 1:
            X2_[i, j] = T       
X2 = X2_ + np.random.normal(size=(50, 100))
X_fit, f_dist = ssvd_approx(X2, layer=10)

print(X_fit)
print(f_dist)