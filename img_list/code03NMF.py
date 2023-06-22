import numpy as np
import random


def nmf(X, r, k, e):
    '''
    X是原始矩阵
    r是分解的两个非负矩阵的隐变量维度，要远小于原始矩阵的维度
    k是迭代次数
    e是理想误差

    input X
    output U,V
    '''
    d, n = X.shape
    # print(d,n)
    # U = np.mat(random.random((d, r)))
    U = np.mat(np.random.rand(d, r))
    # V = np.mat(random.random((n, r)))
    V = np.mat(np.random.rand(n, r))
    # print(U, V)

    x = 1
    for x in range(k):
        print('---------------------------------------------------')
        print('开始第', x, '轮迭代')
        # error
        X_pre = U * V.T
        E = X - X_pre
        # print E
        err = 0.0
        for i in range(d):
            for j in range(n):
                err += E[i, j] * E[i, j]
        print('误差：', err)

        if err < e:
            break
        # update U
        a_u = U * (V.T) * V
        b_u = X * V
        for i_1 in range(d):
            for j_1 in range(r):
                if a_u[i_1, j_1] != 0:
                    U[i_1, j_1] = U[i_1, j_1] * b_u[i_1, j_1] / a_u[i_1, j_1]
        # print(U)

        # update V
        a_v = V * (U.T) * U
        b_v = X.T * U
        print(r, n)
        for i_2 in range(n):
            for j_2 in range(r):
                if a_v[i_2, j_2] != 0:
                    V[i_2, j_2] = V[i_2, j_2] * b_v[i_2, j_2] / a_v[i_2, j_2]
        # print(V)
        print('第', x, '轮迭代结束')

    return U, V


if __name__ == "__main__":
    X = [[5, 3, 2, 1, 2, 3],
         [4, 2, 2, 1, 1, 5],
         [1, 1, 2, 5, 2, 3],
         [1, 2, 2, 4, 3, 2],
         [2, 1, 5, 4, 1, 1],
         [1, 2, 2, 5, 3, 2],
         [2, 5, 3, 2, 2, 5],
         [2, 1, 2, 5, 1, 1], ]
    X = np.mat(X)
    # print(X)
    U, V = nmf(X, 2, 100, 0.001)
    print(V.shape)
    print(U * V.T)