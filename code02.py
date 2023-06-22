
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import nimfa
import matplotlib.pyplot as plt
import scipy.sparse as spr
import cv2
import os

from sklearn import decomposition

# n_components：用于指定分解后矩阵的单个维度k，这个参数也可以看做，降维后希望留下的特征的数量；
#
# init：W矩阵和H矩阵的初始化方式，默认为‘nndsvdar’

"""
    采用降维方法
    （1）图像分块
    （2）每块用非负矩阵分解寻找一组基图像
    （3）带压缩图片的每块分解到基图像上
"""

# 边里该文件夹下的文件名称
def read_directory(directory_name):
    file_list = []
    for filename in os.listdir(directory_name):
        str = directory_name + '/' + filename
        file_list.append(str)
    return file_list

def bright_Norm_one(filename, image):

    Gamma = np.log(128.0/255.0) / np.log(cv2.mean(image)[0]/255.0)
    lookUpTable = np.empty((1, 256), np.uint8)

    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, Gamma)*255.0, 0, 255)

    imageNorm = cv2.LUT(image, lookUpTable)

    return imageNorm


#
# # 读取的目录
# bright_Norm('./Original_image')
def divide_method1(img, m, n):  # 分割成m行n列
    print(img.shape)
    h, w = img.shape[0], img.shape[1]
    gx = np.round(h).astype(np.int)
    gy = np.round(w).astype(np.int)
    divide_image = np.zeros([m - 1, n - 1, int(h * 1.0 / (m - 1) + 0.5), int(w * 1.0 / (n - 1) + 0.5), 3],
                            np.uint8)  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息
    for i in range(m - 1):
        for j in range(n - 1):
            print(i)
            print(j)
            # 这样写比a[i,j,...]=要麻烦，但是可以避免网格分块的时候，有些图像块的比其他图像块大一点或者小一点的情况引起程序出错
            print(img[gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :])
            divide_image[i, j, 0:gy[i + 1][j + 1] - gy[i][j], 0:gx[i + 1][j + 1] - gx[i][j], :] = img[
                                                                                                  gy[i][j]:gy[i + 1][
                                                                                                      j + 1],
                                                                                                  gx[i][j]:gx[i + 1][
                                                                                                      j + 1], :]

    return divide_image


def divide_method2(img, m, n):  # 分割成m行n列
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
    grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽

    # 满足整除关系时的高、宽
    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    # 图像缩放
    img_re = cv2.resize(img, (w, h),
                        cv2.INTER_LINEAR)  # 也可以用img_re=skimage.transform.resize(img, (h,w)).astype(np.uint8)
    # plt.imshow(img_re)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int_)
    gy = gy.astype(np.int_)

    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w],
                            np.uint8)  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img_re[
                                      gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1]]
    return divide_image


def save_blocks(title, divide_image):  #
    m, n = divide_image.shape[0], divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            print(divide_image[i, j].shape)
            # plt.imshow(divide_image[i, j, :])
            # # plt.show()
            # plt.axis('off')
            # plotPath = str(title) + "+" + str(i) + str(j) + '.jpg'  # 图片保存路径
            # plt.imsave("./img_list/" + plotPath, divide_image[i, j, :], cmap='gray')  # 保存灰度图像


# def nmf(X, r, k, e):
#     '''
#     X是原始矩阵
#     r是分解的两个非负矩阵的隐变量维度，要远小于原始矩阵的维度
#     k是迭代次数
#     e是理想误差
#
#     input X
#     output U,V
#     '''
#     d, n = X.shape
#     # print(d,n)
#     # U = np.mat(random.random((d, r)))
#     U = np.mat(np.random.rand(d, r))
#     # V = np.mat(random.random((n, r)))
#     V = np.mat(np.random.rand(n, r))
#     # print(U, V)
#
#     x = 1
#     for x in range(k):
#         print('---------------------------------------------------')
#         print('开始第', x, '轮迭代')
#         # error
#         X_pre = U * V.T
#         E = X - X_pre
#         # print E
#         err = 0.0
#         for i in range(d):
#             for j in range(n):
#                 err += E[i, j] * E[i, j]  # 二范数，欧式距离
#         print('误差：', err)
#
#         if err < e:
#             break
#         # update U
#         a_u = U * (V.T) * V
#         b_u = X * V
#         for i_1 in range(d):
#             for j_1 in range(r):
#                 if a_u[i_1, j_1] != 0:
#                     U[i_1, j_1] = U[i_1, j_1] * b_u[i_1, j_1] / a_u[i_1, j_1]
#         # print(U)
#
#         # update V
#         a_v = V * (U.T) * U
#         b_v = X.T * U
#         print(r, n)
#         for i_2 in range(n):
#             for j_2 in range(r):
#                 if a_v[i_2, j_2] != 0:
#                     V[i_2, j_2] = V[i_2, j_2] * b_v[i_2, j_2] / a_v[i_2, j_2]
#         # print(V)
#         print('第', x, '轮迭代结束')
#
#     return U, V


def nmf(V):

    lsnmf = nimfa.Lsnmf(V, max_iter=10, rank=2)
    print(V.shape)
    lsnmf_fit = lsnmf()

    W = lsnmf_fit.basis()
    print("W:", W.shape)
    print('Basis matrix:\n%s' % W)

    H = lsnmf_fit.coef()
    print("H:", H.shape)
    print('Mixture matrix:\n%s' % H)

    print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))

    print('Rss: %5.3f' % lsnmf_fit.fit.rss())
    print('Evar: %5.3f' % lsnmf_fit.fit.evar())
    print('Iterations: %d' % lsnmf_fit.n_iter)
    print('Target estimate:\n%s' % np.dot(W, H))
    return W, H

def create_substrate(intput, epoch, h, w, m, n, bm):
    '''
    构建基底
    划分原始图像并抽取每个子块的基
    epoch为迭代次数
    :return:
    '''

    grid_h = int(h * 1.0 / m)  # 每个网格的高
    grid_w = int(w * 1.0 / n)  # 每个网格的宽
    divide_image = np.zeros([bm, m, n, grid_h, grid_w],
                            np.uint8)  # 用于保存分割后的图像
    print(divide_image.shape)
    title = 1
    # print("==============开始分块处理文件夹内的图片==============")
    for input_path in intput:
        # print("开始处理第", title, "个，其地址为：", input_path)
        img = cv2.imread(input_path, 0)  # 图片地址
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        divide_image2 = divide_method2(img, m + 1, n + 1)  # 该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
        # save_blocks(title, divide_image2)
        divide_image[title - 1] = divide_image2
        # print(divide_image[title-1, ...].shape)
        # print(divide_image2.shape)
        # print("第", title, "个已分块完毕")
        # print("-----------------------------")
        title += 1
    # print("==============完成全部分块处理文件夹的图片==============")

    # (2) 抽取
    X = np.zeros([bm, grid_h * grid_w], np.uint8)  # 用于保存分割后的图像
    for i in range(bm):
        for j in range(m):
            for k in range(n):
                X[i] = divide_image[i, j, k, ...].flatten()

    # print(X.shape)
    # U, V = nmf(X, 2, epoch, 0.001)

    W, H = nmf(X)


    return H.T


if __name__ == '__main__':

    # 读取图像
    intput = read_directory("./BriNorm_image")
    print("图片共有：", len(intput))
    bm = len(intput)  # 原始维度
    epoch = 1  # 迭代次数
    h, w = 1024, 3072
    m = 2 # 8
    n = 6  # 24
    # 划分原始图像并抽取每个子块的基
    H = create_substrate(intput, epoch, h, w, m, n, bm)
    print(H.shape)
    print(H)

    # 对于未知子图，用基表示图像，寻找系数向量c
    # 读取图片

    testimg = read_directory("./Test_image")
    print(len(testimg))
    id = 1
    for test_path in testimg:
        print("开始处理第", id, "个，其地址为：", test_path)
        img = cv2.imread(test_path, 0)  # 图片地址
        # cv2.imshow(str(id), img)
        # cv2.waitKey(0)

        # （1）亮度变换
        norm_img = bright_Norm_one(test_path, img)
        # cv2.imshow(str(id), norm_img)
        # cv2.waitKey()

        # （2）划分图像
        grid_h = int(h * 1.0 / m)  # 每个网格的高
        grid_w = int(w * 1.0 / n)  # 每个网格的宽

        divide_image2 = divide_method2(img, m + 1, n + 1)  # 该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
        print(divide_image2.shape)

        # （3）用基底表示图
        for i in range(divide_image2.shape[0]):
            for j in range(divide_image2.shape[1]):
                print(i, j)
                d_pre = divide_image2[i, j].flatten()
                y = d_pre.reshape(-1, 1)
                print(y.shape)
                # 把公式写出来
                c = np.linalg.inv((H.T) * H) * (H.T)*y
                    # * (H.T) * d_pre
                print(c.shape)
                print(c)

        id += 1





    # print("图片共有：", len(intput))
    # bm = len(intput)  # 原始维度
    #
    # h, w = 1024, 3072
    # m = 2  # 8
    # n = 6  # 24
    # grid_h = int(h * 1.0 / m)  # 每个网格的高
    # grid_w = int(w * 1.0 / n)  # 每个网格的宽
    # divide_image = np.zeros([bm, m, n, grid_h, grid_w],
    #                         np.uint8)   #  用于保存分割后的图像
    # print(divide_image.shape)
    # title = 1
    # # print("==============开始分块处理文件夹内的图片==============")
    # for input_path in intput:
    #     print("开始处理第", title, "个，其地址为：", input_path)
    #     img = cv2.imread(input_path, 0)  # 图片地址
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     divide_image2 = divide_method2(img, m + 1, n + 1)  # 该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
    #     # save_blocks(title, divide_image2)
    #     divide_image[title-1] = divide_image2
    #     # print(divide_image[title-1, ...].shape)
    #     # print(divide_image2.shape)
    #     # print("第", title, "个已分块完毕")
    #     # print("-----------------------------")
    #     title += 1
    # # print("==============完成全部分块处理文件夹的图片==============")
    #
    # # (2) 抽取
    # X = np.zeros([bm, grid_h*grid_w], np.uint8)  # 用于保存分割后的图像
    # for i in range(bm):
    #     for j in range(m):
    #         for k in range(n):
    #             X[i] = divide_image[i, j, k, ...].flatten()
    #
    # print(X.shape)
    # U, V = nmf(X, 2, 10, 0.001)

    # print(V.shape)
    # print(V.T)
    # print(U.shape)
    # print(U)


        #         print(divide_image[i, j, k, ...].shape)
        #         plt.imshow(divide_image[i, j, k, ...], cmap='gray')
        #         plt.show()

    # c = np.zeros(bm, np.uint8)  # 系数向量c  先假设是1个
    # print(c)
    # print(c.shape)

    # for i in range(bm):

        # for j in range(m):
        #     for k in range(n):
        #         print(divide_image[i, j, k, ...].shape)
        #         plt.imshow(divide_image[i, j, k, ...], cmap='gray')
        #         plt.show()


