#-*-encoding:utf8-*-
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

def MDS(D,d):
    D = np.asarray(D)
    N = D.shape[0]
    DSquare = D ** 2
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis = 0)
    rowMean = np.mean(DSquare, axis = 1)
    B = np.zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5 * (DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
    eigVal, eigVec = np.linalg.eig(B)#求特征值及特征向量
    #对特征值进行排序，得到排序索引
    eigValSorted_indices = np.argsort(eigVal)
    # #提取d个最大特征向量
    topd_eigVec = eigVec[:,eigValSorted_indices[:-d-1:-1]] #-d-1前加:才能向左切
    X = np.dot(topd_eigVec, np.sqrt(np.diag(eigVal[:-d-1:-1])))
    # X = np.dot(eigVec[:,:2],np.diag(np.sqrt(eigVal[:2])))

    print('original distance','\tnew distance')
    for i in range(N):
        for j in range(i+1,N):
            # print(X[i],X[j])
            print(np.str(D[i,j]),'\t\t',np.str("%.4f"%np.linalg.norm(X[i]-X[j])))
    # return X

data  = pd.read_excel("./题目1.xlsx")
D = np.array([col[1:] for col in data.values])

D = [[0.0,1.75,4.25,6.0,15.5,3.66,12.75,6.35,3.83],
[1.75,0.0,2.5,3.25,8.33,8.15,16.07,8.25,2.07],
[4.25,2.5,0.0,11.75,5.57,15.75,27.1,12.0,4.42],
[6.0,3.25,11.75,0.0,17.0,16.75,24.33,17.07,8.95],
[15.5,8.33,5.57,17.0,0.0,9.66,11.0,14.9,1.18],
[3.66,8.15,15.75,16.75,9.66,0.0,6.15,8.7,7.43],
[12.75,16.07,27.1,24.33,11.0,6.15,0.0,11.0,1.26],
[6.35,8.25,12.0,17.07,14.9,8.7,11.0,0.0,5.28],
[3.83,2.07,4.42,8.95,1.18,7.43,1.26,5.28,0.0]]
# D = [[0.0,1.75,4.25,6.0,15.5,3.66,12.75,6.35,3.83,4.63],
# [1.75,0.0,2.5,3.25,8.33,8.15,16.07,8.25,2.07,6.88],
#  [4.25,2.5,0.0,11.75,5.57,15.75,27.1,12.0,4.42,8.56],
#  [6.0,3.25,11.75,0.0,17.0,16.75,24.33,17.07,8.95,23.46],
#  [15.5,8.33,5.57,17.0,0.0,9.66,11.0,14.9,1.18,10.68],
#  [3.66,8.15,15.75,16.75,9.66,0.0,6.15,8.7,7.43,8.27],
#  [12.75,16.07,27.1,24.33,11.0,6.15,0.0,11.0,1.26,10.92],
#  [6.35,8.25,12.0,17.07,14.9,8.7,11.0,0.0,5.28,11.92],
#  [3.83,2.07,4.42,8.95,1.18,7.43,1.26,5.28,0.0,9.53],
#  [4.63, 6.88, 8.56, 23.46, 10.68, 8.27, 10.92, 11.92, 9.53, 0.0]]
print(D)
print(MDS(D,2))
# N = D.shape[0]
# T = np.zeros((N,N))
#
# D2 = D**2
# H = np.eye(N) - 1/N
# T = -0.5*np.dot(np.dot(H,D2),H)
#
# eigVal,eigVec = np.linalg.eig(T)
# X = np.dot(eigVec[:,:2],np.diag(np.sqrt(eigVal[:2])))
#
# print('original distance','\tnew distance')
# for i in range(N):
#     for j in range(i+1,N):
#         print(np.str(D[i,j]),'\t\t',np.str("%.4f"%np.linalg.norm(X[i]-X[j])))
