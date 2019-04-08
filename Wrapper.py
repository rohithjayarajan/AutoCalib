#!/usr/bin/evn python

"""
@file    Wrapper.py
@author  rohithjayarajan
@date 03/21/2019

Licensed under the
GNU General Public License v3.0
"""

import numpy as np
import cv2
import argparse
import math
import glob
from Misc.AutoCalibUtils import AutoCalibUtils
import scipy.optimize as opt

debug = False


class AutoCalib:
    def __init__(self, DataPath, EdgeLength, Rows, Cols):
        InputImageList = []
        for filename in sorted(glob.glob(DataPath + '/*.jpg')):
            ImageTemp = cv2.imread(filename)
            # ImageTemp = cv2.cvtColor(ImageTemp, cv2.COLOR_BGR2GRAY)
            # ImageTemp = cv2.resize(ImageTemp, (300, 400))
            InputImageList.append(ImageTemp)
        self.Images = np.array(InputImageList)
        self.EdgeLength = float(EdgeLength)
        self.Rows = int(Rows)
        self.Cols = int(Cols)
        self.AutoCalibHelper = AutoCalibUtils()

    def getImageAndWorldPoints(self):
        Pts3D = np.zeros((self.Cols*self.Rows, 3), np.float32)
        Pts3D[:, :2] = np.mgrid[0:self.Rows,
                                0:self.Cols].T.reshape(-1, 2)
        Pts3D = Pts3D*self.EdgeLength
        Pts3D[:, 2] = 1
        WorldPoints = []
        ImgPoints = []
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        hits = 0
        for image in self.Images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, (self.Rows, self.Cols), None)
            if ret == True:
                hits += 1
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                WorldPoints.append(Pts3D)
                ImgPoints.append(corners2)
                # print(ImgPoints)
                # print(WorldPoints)

                if debug:
                    print(WorldPoints)
                    img = cv2.drawChessboardCorners(
                        image, (self.Rows, self.Cols), corners2, ret)
                    # img = cv2.resize(img, (1080, 720))
                    cv2.imshow('img', img)
                    cv2.waitKey()
                    cv2.destroyAllWindows()

        ImgPoints = np.array(ImgPoints)
        WorldPoints = np.array(WorldPoints)
        WorldPoints = np.reshape(WorldPoints, (hits, self.Cols*self.Rows, 3))
        ImgPoints = np.reshape(ImgPoints, (hits, self.Cols*self.Rows, 2))

        # print(ImgPoints)
        # print(WorldPoints)
        # print(ImgPoints.shape)
        # print(WorldPoints.shape)

        return ImgPoints, WorldPoints

    def getInitialEstimates(self):
        ImgPoints, WorldPoints = self.getImageAndWorldPoints()
        n = ImgPoints.shape[0]
        HStack = []
        print("##########Estimating Homography Matrices##########")
        for i in range(n):
            H = self.AutoCalibHelper.EstimateHomography(
                ImgPoints[i], WorldPoints[i])
            HStack.append(H)
        HStack = np.array(HStack)
        # print("H: {}".format(HStack))
        print("##########Estimating Homography Matrices Done##########")
        InitBMatrix = self.AutoCalibHelper.EstimateBMatrix(HStack)
        InitAMatrix = self.AutoCalibHelper.EstimateAMatrix(InitBMatrix)
        InitKMatrix = self.AutoCalibHelper.EstimateInitialK()
        InitRTStack, InitRT4Stack = self.AutoCalibHelper.EstimateExtrinsic(
            InitAMatrix, HStack)
        print("InitAMatrix: {}".format(InitAMatrix))
        print("InitKMatrix: {}".format(InitKMatrix))
        return InitAMatrix, InitKMatrix, InitRTStack, InitRT4Stack, ImgPoints, WorldPoints, self.Images


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default='/home/rohith/CMSC733/git/AutoCalib/Data',
                        help='Path for image data path, Default:/home/rohith/CMSC733/git/AutoCalib/Data')
    Parser.add_argument('--EdgeLength', default='21.5',
                        help='Edge Length of Square, Default:0.0215')
    Parser.add_argument('--Rows', default='9',
                        help='Inner rows in grid, Default:7')
    Parser.add_argument('--Cols', default='6',
                        help='Inner columns in grid, Default:5')

    Args = Parser.parse_args()
    DataPath = Args.DataPath
    EdgeLength = Args.EdgeLength
    Rows = Args.Rows
    Cols = Args.Cols
    AutoCalibHelper = AutoCalibUtils()

    calibrator = AutoCalib(DataPath, EdgeLength, Rows, Cols)
    InitAMatrix, InitKMatrix, InitRTStack, InitRT4Stack, ImgPoints, WorldPoints, ImageList = calibrator.getInitialEstimates()

    def f(x0, InitRTStack, InitRT4Stack, ImgPoints, WorldPoints):
        s = []
        i = 0
        InitAMatrix, InitKMatrix = AutoCalibHelper.recoverAK(x0)
        for worldPts in WorldPoints:
            RT = InitRTStack[i]
            RT4 = InitRT4Stack[i]
            ARt = np.matmul(InitAMatrix, RT)
            j = 0
            ssum = 0
            for Pts in worldPts:
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
                ssum += np.linalg.norm((mij-M), ord=2)
                j += 1
            i += 1
            s.append(ssum)
        return np.array(s)

    x0 = AutoCalibHelper.vectorizeAK(InitAMatrix, InitKMatrix)
    res = opt.least_squares(fun=f, x0=x0, method="lm", args=[
        InitRTStack, InitRT4Stack, ImgPoints, WorldPoints])
    x1 = res.x
    AMatrix, KMatrix = AutoCalibHelper.recoverAK(x1)
    print("AMatrix: {}".format(AMatrix))
    print("KMatrix: {}".format(KMatrix))
    AutoCalibHelper.reprojectPoints(
        ImageList, AMatrix, KMatrix, InitRTStack, InitRT4Stack, ImgPoints, WorldPoints)


if __name__ == '__main__':
    main()
