
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import nimfa
import matplotlib.pyplot as plt
import scipy.sparse as spr
import cv2
import os


"""
    采用降维方法
    （1）图像分块
    （2）每块用非负矩阵分解寻找一组基图像
    （3）带压缩图片的每块分解到基图像上
"""

SPARE_ELEMENT = 0
SPARE_RATE = 0.5
# 边里该文件夹下的文件名称
def read_directory(directory_name):
    file_list = []
    for filename in os.listdir(directory_name):
        str = directory_name + '/' + filename
        file_list.append(str)
    return file_list

def bright_Norm(intput, l, h, w):
    """
        调整采集图片的亮度色度问题
        根据灰度、Gamma归一化亮度
    """
    normImg_list = np.zeros([l, h, w], np.uint8)  # 用于保存亮度变换后的图像

    id = 0
    for input_path in intput:
        # print("开始处理第?????", id, "个，其地址为：", input_path)
        image = cv2.imread(input_path, 0)  # 图片地址
        # print(image.shape)
        Gamma = np.log(128.0/255.0) / np.log(cv2.mean(image)[0]/255.0)
        lookUpTable = np.empty((1, 256), np.uint8)

        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, Gamma)*255.0, 0, 255)

        normImg_list[id] = cv2.LUT(image, lookUpTable)
        id += 1
        # cv2.imshow(filename, imageNorm)
        # cv2.waitKey(0)
        #
        # cv2.imwrite('./BriNorm_image'+'/'+filename, imageNorm)
    return normImg_list

def bright_Norm_one(image):

    Gamma = np.log(128.0/255.0) / np.log(cv2.mean(image)[0]/255.0)
    lookUpTable = np.empty((1, 256), np.uint8)

    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, Gamma)*255.0, 0, 255)

    imageNorm = cv2.LUT(image, lookUpTable)

    return imageNorm

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


def isSparse(matrix):
    """
    Judge spare matrix.
    :param matrix: matrix
    :return: boolean
    """
    sum = len(matrix) * len(matrix[0])
    spare = 0

    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            if matrix[row][column] == SPARE_ELEMENT:
                spare += 1

    if spare / sum >= SPARE_RATE:
        return True
    else:
        return False

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


def nmf(V):

    lsnmf = nimfa.Lsnmf(V, max_iter=100, rank=100)
    # print(V.shape)
    lsnmf_fit = lsnmf()

    W = lsnmf_fit.basis()
    # print("W:", W.shape)
    # print('Basis matrix:\n%s' % W)

    H = lsnmf_fit.coef()
    # print("H:", H.shape)
    # print('Mixture matrix:\n%s' % H)
    #
    # print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))
    #
    # print('Rss: %5.3f' % lsnmf_fit.fit.rss())
    # print('Evar: %5.3f' % lsnmf_fit.fit.evar())
    # print('Iterations: %d' % lsnmf_fit.n_iter)
    # print('Target estimate:\n%s' % np.dot(W, H))
    return W, H

def create_substrate(intput, h, w, m, n):

    '''
    构建基底
    划分原始图像并抽取每个子块的基
    :return:
    '''
    print('input shape:', intput.shape)
    grid_h = int(h * 1.0 / m)  # 每个网格的高
    grid_w = int(w * 1.0 / n)  # 每个网格的宽
    divide_image = np.zeros([bm, m, n, grid_h, grid_w],
                            np.uint8)  # 用于保存分割后的图像
    print('divide_image_shape:', divide_image.shape)

    # print("==============开始分块处理文件夹内的图片==============")
    # for input_path in intput:
        # print("开始处理第", title, "个，其地址为：", input_path)
        # img = cv2.imread(input_path, 0)  # 图片地址
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for title in range(intput.shape[0]):
        img = intput[title]
        divide_image2 = divide_method2(img, m + 1, n + 1)  # 该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
        # save_blocks(title, divide_image2)
        divide_image[title] = divide_image2
        # print(divide_image[title-1, ...].shape)
        # print(divide_image2.shape)
        # print("第", title, "个已分块完毕")
        # print("-----------------------------")
        title += 1
    # print("==============完成全部分块处理文件夹的图片==============")

    # (2) 抽取
    X = np.zeros([m, n, bm, grid_h * grid_w], np.uint8)  # 存储每一块抽取基矩阵的输入向量 [2,6,249,262144]
    Basis = np.zeros([m, n, 100, grid_h * grid_w], np.uint8)  # 存储每一块的基矩阵[2, 6, 100, 262144]
    # print('X.shape:', X.shape)
    for i in range(m):
        for j in range(n):
            for k in range(bm):
                X[i][j][k] = divide_image[k, i, j, ...].flatten()
            # print('X[i][j].shape:', X[i][j].shape)  # [249,262144]
            W, H = nmf(X[i][j])  # H.T 是基矩阵
            # print('H shape:', H.shape)  # [100, 262144]
            Basis[i, j] = H

    return Basis

def compute_coef( testimg, H, l, m, n):
    '''
    计算测试图片用基矩阵表示的向量
    :param testimg: 测试图像集合
    :param m: 纵向划分几块
    :param n: 横向划分几块
    :return: 系数向量
    '''
    id = 1
    for test_path in testimg:
        # print("开始处理第", id, "个，其地址为：", test_path)
        img = cv2.imread(test_path, 0)  # 图片地址
        # cv2.imshow(str(id), img)
        # cv2.waitKey(0)

        # （1）亮度变换
        norm_img = bright_Norm_one(img)
        # cv2.imshow(str(id), norm_img)
        # cv2.waitKey()

        # （2）划分图像
        divide_image2 = divide_method2(norm_img, m + 1, n + 1)  # 该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
        # print(divide_image2.shape)

        # （3）用基底表示图
        cn = np.zeros([l, m, n, 100, 1], np.float64)
        # cn.reshape((l, m, n, 100))
        for i in range(divide_image2.shape[0]):
            for j in range(divide_image2.shape[1]):
                # print(i, j)
                d_pre = divide_image2[i, j].flatten()
                y = d_pre.reshape(-1, 1)
                # print(y.shape)
                # 把公式写出来
                # print("先看看", np.linalg.inv((H.T) * H) * (H.T) * y)

                ci = np.linalg.inv((H.T) * H) * (H.T) * y  # 系数向量c
                # print("ci???????", ci.shape)
                # print(ci)
                cn[id, i, j] = ci
                # print("leixing:", cn.dtype, ci.dtype)
                # print(cn[id, i, j].shape)
                # print(cn[id, i, j])
    id += 1
    return cn



def compute_coef_one(img, Basis, m, n):
    # （1）亮度变换
    norm_img = bright_Norm_one(img)
    # cv2.imshow(str(id), norm_img)
    # cv2.waitKey()

    # （2）划分图像
    divide_image2 = divide_method2(norm_img, m + 1, n + 1)  # 该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
    # print(divide_image2.shape)

    # （3）用基底表示图
    cn = np.zeros([m, n, 100, 1], np.float64)
    # cn.reshape((l, m, n, 100))
    for i in range(divide_image2.shape[0]):
        for j in range(divide_image2.shape[1]):
            # print(i, j)
            d_pre = divide_image2[i, j].flatten()
            y = d_pre.reshape(-1, 1)
            # print(y.shape)
            # 把公式写出来
            # print("先看看", np.linalg.inv((H.T) * H) * (H.T) * y)
            ci = np.dot(np.dot(np.linalg.inv(np.dot(Basis[i, j], Basis[i, j].T)), (Basis[i, j])), y)  # 系数向量c
            # ci = np.linalg.inv((Basis[i, j].T) * Basis[i, j]) * (Basis[i, j].T) * y  # 系数向量c
            # print("ci???????", ci.shape)
            # print(ci)
            cn[i, j] = ci

    #  （4）计算残差
    residuals = np.zeros([m, n, 512*512, 1], np.float64)
    for i in range(divide_image2.shape[0]):
        for j in range(divide_image2.shape[1]):
            d_pre = divide_image2[i, j].flatten()
            y = d_pre.reshape(-1, 1)
            print('打印出来看看', residuals[i, j].shape, y.shape, np.dot(Basis[i, j].T, cn[i, j]).shape)
            residuals[i, j] = y - np.dot(Basis[i, j].T, cn[i, j])
    print('residuals.shape', residuals.shape)
    return cn, residuals

if __name__ == '__main__':

    # 读取图像
    intput = read_directory("./Original_image")
    print("图片共有：", len(intput))
    bm = len(intput)  # 原始维度
    h, w = 1024, 3072
    m = 2 # 8
    n = 6  # 24
    norm_img_list = bright_Norm(intput, bm, h, w)

    # 划分原始图像并抽取每个子块的基
    Basis = create_substrate(norm_img_list, h, w, m, n)
    print('Basis.shape:', Basis.shape)
    # print(Basis)

    # 处理单张图片
    img = cv2.imread('000002.jpg', 0)
    print(img.shape)

    c, residuals = compute_coef_one(img, Basis, m, n)
    # print(c.shape)
    # print(c)
    for i in range(m):
        for j in range(n):
            cv2.imshow('1', residuals[i, j])
            cv2.waitKey()
            print(isSparse(residuals[i, j]))

    # 对于未知子图，用基表示图像，寻找系数向量c
    # 处理一个目录
    # testimg = read_directory("./test")  # 读取图片
    # print(len(testimg))
    # c = compute_coef(testimg, bm, m, n)
    # print(c.shape)
    # print(c)
    # print(c[1, 1, 2])




# 下一步应该就是用系数，恢复成图像块，然后和原图像块做差得到残差。