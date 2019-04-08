"""
@file    AutoCalibUtils.py
@author  rohithjayarajan
@date 03/21/2019

Licensed under the
GNU General Public License v3.0
"""

import numpy as np
import cv2
import math
import scipy

debug = False


class AutoCalibUtils:
    def EstimateHomography(self, points1, points2):
        A = np.zeros([2*points1.shape[0], 9])

        idX = 0
        idY = 0
        while (idX < 2*points1.shape[0]):
            ydash = points1[idY][1]
            xdash = points1[idY][0]
            y = points2[idY][1]
            x = points2[idY][0]
            A[idX][0] = x
            A[idX][1] = y
            A[idX][2] = 1
            A[idX][6] = -xdash*x
            A[idX][7] = -xdash*y
            A[idX][8] = -xdash
            idX += 1
            A[idX][3] = x
            A[idX][4] = y
            A[idX][5] = 1
            A[idX][6] = -ydash*x
            A[idX][7] = -ydash*y
            A[idX][8] = -ydash
            idX += 1
            idY += 1

            _, _, V = np.linalg.svd(A, full_matrices=True)
            Hpred = V[-1, :].reshape((3, 3))
            Hpred = Hpred/Hpred[2][2]

        return Hpred

    def CreateVMatrix(self, H, i, j):
        return np.array([H[0][i]*H[0][j],
                         H[0][i]*H[1][j] + H[1][i]*H[0][j],
                         H[1][i]*H[1][j],
                         H[2][i]*H[0][j] + H[0][i]*H[2][j],
                         H[2][i]*H[1][j] + H[1][i]*H[2][j],
                         H[2][i]*H[2][j]])

    def EstimateBMatrix(self, HMatrices):
        print("##########Estimating B Matrix##########")
        VMatrix = []
        for H in HMatrices:
            v12 = self.CreateVMatrix(H, 0, 1)
            v11 = self.CreateVMatrix(H, 0, 0)
            v22 = self.CreateVMatrix(H, 1, 1)
            VMatrix.append(v12)
            VMatrix.append(v11-v22)
        VMatrix = np.array(VMatrix)

        _, _, V = np.linalg.svd(VMatrix, full_matrices=True)
        BMatrixEstimate = V[-1, :]
        print("##########Estimating B Matrix Done##########")
        print("BMatrixEstimate: {}".format(BMatrixEstimate))
        return BMatrixEstimate

    def EstimateAMatrix(self, B):
        print("##########Estimating A Matrix##########")
        v0 = (B[1]*B[3]-B[0]*B[4])/(B[0]*B[2] - B[1]**2)
        print("v0: {}".format(v0))
        lamdiv = B[5]-(B[3]**2 + v0*(B[1]*B[3]-B[0]*B[4]))/B[0]
        print("lamdiv: {}".format(lamdiv))
        alpha = math.sqrt(lamdiv/B[0])
        print("alpha: {}".format(alpha))
        beta = math.sqrt(lamdiv*B[0]/(B[0]*B[2]-B[1]**2))
        print("beta: {}".format(beta))
        gamma = -B[1]*(alpha**2)*beta/lamdiv
        print("gamma: {}".format(gamma))
        u0 = gamma*v0/beta - B[3]*(alpha**2)/lamdiv
        print("u0: {}".format(u0))

        AMatrix = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
        print("##########Estimating A Matrix Done##########")
        return AMatrix

    def EstimateInitialK(self):
        print("##########Estimating k Matrix##########")
        print("##########Estimating k Matrix Done##########")
        return np.array([[0], [0]])

    def EstimateExtrinsic(self, A, HMatrices):
        RTStack = []
        RT4Stack = []
        print("##########Estimating R|t Matrix##########")
        for H in HMatrices:
            lambdiv = 1 / \
                (np.linalg.norm(np.matmul(np.linalg.inv(A), H[:, 0]), ord=2))
            r1 = lambdiv*(np.matmul(np.linalg.inv(A), H[:, 0]))
            r2 = lambdiv*(np.matmul(np.linalg.inv(A), H[:, 1]))
            r3 = np.cross(r1, r2)
            t = lambdiv*(np.matmul(np.linalg.inv(A), H[:, 2]))

            # print("r1: {}".format(r1))
            # print("r2: {}".format(r2))
            # print("r3: {}".format(r3))
            # print("t: {}".format(t))

            R = np.column_stack((r1, r2))
            R = np.column_stack((R, r3))
            R = np.reshape(R, (3, 3))
            # print("R: {}".format(R))

            # U, _, V = np.linalg.svd(R, full_matrices=True)
            # R = np.matmul(U, V)
            # print("R: {}".format(R))
            RT = np.column_stack((R[:, :2], t))
            RT4 = np.column_stack((R, t))
            # print("RT: {}".format(RT))
            # print("RT4: {}".format(RT4))
            RTStack.append(RT)
            RT4Stack.append(RT4)
        RTStack = np.array(RTStack)
        RT4Stack = np.array(RT4Stack)
        print("##########Estimating R|t Matrix Done##########")
        return RTStack, RT4Stack

    def computeMeanReprojectionError(self, InitAMatrix, InitKMatrix, InitRTStack, InitRT4Stack, ImgPoints, WorldPoints):
        s = 0
        i = 0
        d = 0
        for worldPts in WorldPoints:
            RT = InitRTStack[i]
            RT4 = InitRT4Stack[i]
            ARt = np.matmul(InitAMatrix, RT)
            j = 0
            for Pts in worldPts:
                d += 1
                world3d = np.array([Pts[0], Pts[1], 0, 1])
                world3d = np.reshape(world3d, (4, 1))
                # print("world3d: {}".format(world3d))
                xy = np.matmul(RT4, world3d)
                x = xy[0]/xy[2]
                y = xy[1]/xy[2]
                mij = ImgPoints[i][j].reshape((2, 1))
                mij = np.append(mij, 1)
                mij = mij.reshape((3, 1))
                #
                # print("M b4: {}".format(M))
                Mhat = np.matmul(ARt, Pts)
                Mhat = Mhat/Mhat[2]
                # print("Mhat: {}".format(Mhat))
                uCap = Mhat[0] + \
                    (Mhat[0]-InitAMatrix[0][2])*(InitKMatrix[0] *
                                                 (x**2 + y**2) + InitKMatrix[1]*(x**2 + y**2)**2)
                vCap = Mhat[1] + \
                    (Mhat[0]-InitAMatrix[1][2])*(InitKMatrix[0] *
                                                 (x**2 + y**2) + InitKMatrix[1]*(x**2 + y**2)**2)
                M = np.array([[uCap[0]], [vCap[0]], [1]])
                # print("M after: {}".format(M))
                #
                s += np.linalg.norm((mij-M), ord=2)
                j += 1
            i += 1
        print(math.sqrt(s/d))
        return math.sqrt(s/d)

    def vectorizeAK(self, InitAMatrix, InitKMatrix):
        x0 = np.array([InitAMatrix[0][0], InitAMatrix[0][1], InitAMatrix[0][2],
                       InitAMatrix[1][1], InitAMatrix[1][2], InitKMatrix[0], InitKMatrix[1]])
        return x0

    def recoverAK(self, x0):
        InitKMatrix = np.array([[x0[5]], [x0[6]]])
        InitAMatrix = np.array(
            [[x0[0], x0[1], x0[2]], [0, x0[3], x0[4]], [0, 0, 1]])
        InitAMatrix = InitAMatrix.reshape((3, 3))
        return InitAMatrix, InitKMatrix

    def reprojectPoints(self, ImageList, A, K, InitRTStack, InitRT4Stack, ImgPoints, WorldPoints):
        s = 0
        i = 0
        d = 0
        for worldPts in WorldPoints:
            RT = InitRTStack[i]
            RT4 = InitRT4Stack[i]
            ARt = np.matmul(A, RT)
            j = 0
            for Pts in worldPts:
                d += 1
                world3d = np.array([Pts[0], Pts[1], 0, 1])
                world3d = np.reshape(world3d, (4, 1))
                # print("world3d: {}".format(world3d))
                xy = np.matmul(RT4, world3d)
                x = xy[0]/xy[2]
                y = xy[1]/xy[2]
                mij = ImgPoints[i][j].reshape((2, 1))
                mij = np.append(mij, 1)
                mij = mij.reshape((3, 1))
                #
                # print("M b4: {}".format(M))
                Mhat = np.matmul(ARt, Pts)
                Mhat = Mhat/Mhat[2]
                # print("Mhat: {}".format(Mhat))
                uCap = Mhat[0] + \
                    (Mhat[0]-A[0][2])*(K[0] *
                                       (x**2 + y**2) + K[1]*(x**2 + y**2)**2)
                vCap = Mhat[1] + \
                    (Mhat[0]-A[1][2])*(K[0] *
                                       (x**2 + y**2) + K[1]*(x**2 + y**2)**2)
                M = np.array([[uCap[0]], [vCap[0]], [1]])
                # print("M after: {}".format(M))
                #
                cv2.circle(ImageList[i], (mij[0], mij[1]), 15, (0, 255, 0),
                           thickness=8, lineType=8, shift=0)
                cv2.circle(ImageList[i], (M[0], M[1]), 5, (0, 0, 255),
                           thickness=5, lineType=8, shift=0)
                s += np.linalg.norm((mij-M), ord=2)
                j += 1
            cv2.imwrite('reprojection'+str(i)+'.png', ImageList[i])
            i += 1
        print((s/d))
        return s/d
