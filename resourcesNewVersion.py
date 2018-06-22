import csv
import numpy as nu
import scipy as sc
from scipy import linalg as la
from collections import namedtuple
import matplotlib.pyplot as pl
import sympy as sy
import math as ma
from datetime import time
import multiprocessing as mp
import os

LARGE_NUMBER = 1e8
STEP = 0.0001
DELTA = 1e-5
validDirections = {'u': [0, STEP], 'd': [0, -STEP], 'r': [STEP, 0], 'l': [-STEP, 0],
    'ur': [STEP, STEP], 'ul': [-STEP, STEP], 'dr': [STEP, -STEP], 'dl': [-STEP, -STEP]}
DATASET_ID = 'A'
recogLineFile = 'results/recogLineOfNeResult' + DATASET_ID + '.csv'
figureFile = 'ResultFigure' + DATASET_ID + '.png'
plotAxisDict = {
    'A': [-10, 70, 50, 100],
    'B': [-13, 55, 38, 100],
    'C': [-10, 70, 10, 62],
    'D': [-10, 70, 50, 103]
}


# NOTE: - readFile operations:  ************************************************  readFile operations  ****************

def readFileOfPoints(FILE):
    with open(FILE, 'rt') as file:
        lines = csv.reader(file, delimiter=' ')
        points = [  XYPoint([float(line[0]), float(line[1])]) for line in lines  ]
    return points


def readFileOfCorrectClasses(FILE):
    with open(FILE, 'rt') as file:
        lines = csv.reader(file, delimiter=' ')
        correctClasses = [  int(line[2]) for line in lines if line[2]  ]
    return correctClasses



# NOTE: - class definitions:  **************************************************  class definitions  ********************

class XYPoint():
    def __init__(self, listPoint):
        # self.rawValue = listPoint
        self.x, self.y = listPoint[0], listPoint[1]

    zero = nu.array([0, 0])

    @property
    def rawValue(self):
        return [self.x, self.y]

    @property
    def ndarrayValue(self):
        return nu.array(self.rawValue).reshape(2,1)

    def distanceFrom(self, point, weight=nu.diag([1,1]) ):
        return nu.linalg.norm( nu.dot(weight, (self.ndarrayValue - point.reshape(2,1))) )

    def distanceFromXYPoint(self, point):
        return ma.sqrt( (self.rawValue[0]-point.rawValue[0])**2 + (self.rawValue[1]-point.rawValue[1])**2 )

    def similarityToPoint(self, point):
        inner = nu.dot(self.ndarrayValue.reshape(1,2), point.reshape(2,1))
        distanceProduct = self.distanceFrom(XYPoint.zero) * nu.linalg.norm(point)
        angle = ma.acos(inner / distanceProduct)
        return angle

    def move(self, direction):
        global STEP, validDirections
        if direction in validDirections:
            return XYPoint([ self.x + validDirections[direction][0], self.y + validDirections[direction][1] ])
        else:
            return None


class ClassData():
    def __init__(self, file):
        points = readFileOfPoints(file)
        self.points = points
        self.len = len(points)

    @property
    def toNdarray(self):
        return nu.array([ point.rawValue for point in self.points ])

    @property
    def mean(self):
        return self.toNdarray.sum(axis=0).reshape(2,1) / self.len


class TestData(ClassData):
    def __init__(self, file):
        super().__init__(file)
        correctClasses = readFileOfCorrectClasses(file)
        self.correctClasses = correctClasses

    def nearestNeighborMethod(self, class1, class2):
        nearestNeighborArray = []
        for point in self.points:
            leastDistanceInClass1 = nu.array([ point.distanceFrom(class1Point.ndarrayValue) for class1Point in class1.points ]).min()
            leastDistanceInClass2 = nu.array([ point.distanceFrom(class2Point.ndarrayValue) for class2Point in class2.points ]).min()
            nearestNeighborArray.append( 1 if leastDistanceInClass1 < leastDistanceInClass2 else 2  )
        return nearestNeighborArray

    def euclideanDistanceMethod(self, class1, class2):
        return [ 1 if point.distanceFrom(class1.mean) < point.distanceFrom(class2.mean) else 2 for point in self.points ]

    def weightDistanceMethod(self, class1, class2, weight):
        return [ 1 if point.distanceFrom(class1.mean, weight) < point.distanceFrom(class2.mean, weight) else 2 for point in self.points ]

    def similarityMethod(self, class1, class2):
        return [ 1 if point.similarityToPoint(class1.mean) < point.similarityToPoint(class2.mean) else 2 for point in self.points ]



# NOTE: - covariances:  ********************************************************  covariances  ***********************

def covarianceWithinClasses(class1, class2):
    ''' within-class-covariance '''
    sumOfClass1 = nu.array([ (row.reshape(2,1) - class1.mean) * (row.reshape(2,1) - class1.mean).T for row in class1.toNdarray ]).sum(axis=0)
    sumOfClass2 = nu.array([ (row.reshape(2,1) - class2.mean) * (row.reshape(2,1) - class2.mean).T for row in class2.toNdarray ]).sum(axis=0)
    return (sumOfClass1 + sumOfClass2) / (class1.len + class2.len)


def covarianceBetweenClasses(class1, class2):
    ''' between-class-convariance '''
    meanTotal = (class1.mean + class2.mean) / 2.0
    a = class1.len * (class1.mean - meanTotal) * (class1.mean - meanTotal).T
    b = class2.len * (class2.mean - meanTotal) * (class2.mean - meanTotal).T
    return (a + b) / (class1.len + class2.len)


def covarianceTotal(class1, class2):
    ''' total covariance '''
    meanTotal = (class1.mean + class2.mean) / 2.0
    allPoints = nu.insert(class1.toNdarray, class1.len, class2.toNdarray, axis=0)

    sum = nu.array([ (row.reshape(2,1) - meanTotal) * (row.reshape(2,1) - meanTotal).T for row in allPoints ]).sum(axis=0)
    return sum / (class1.len + class2.len)



# NOTE: - eigen value and vectors:  ********************************************  eigen value and vectors  ***************

Eigen = namedtuple('Eigen', 'values vectors')

def discriminantOf(class1, class2):
    A = covarianceBetweenClasses(class1, class2)
    B = covarianceWithinClasses(class1, class2)
    values, vectors = la.eig(A, B)
    vectors = nu.array([ vector.reshape(2,1) / nu.linalg.norm(vector) for vector in vectors ])

    return Eigen(values, vectors)


class Projection(ClassData):
    def __init__(self, classData, vector):
        vector = vector.reshape(2,1)
        ndarrayPoints = nu.array([ nu.dot(row.reshape(1,2), vector) * vector for row in classData.toNdarray ])
        self.points = [ XYPoint([float(ndarrayPoint[0]), float(ndarrayPoint[1])]) for ndarrayPoint in ndarrayPoints ]
        self.len = classData.len


def pricipalComponentOf(class1, class2):
    A = covarianceTotal(class1, class2)
    B = nu.eye(2)
    values, vectors = la.eig(A, B)
    vectors = nu.array([ vector.reshape(2,1) / nu.linalg.norm(vector) for vector in vectors ])
    return Eigen(values, vectors)



# NOTE: - layout of results:  **************************************************  layout of results  ********************

def errataOf(checkList, correctList):
    checkArray = nu.array(checkList)
    correctArray = nu.array(correctList)
    return checkArray == correctArray


def recognitionRateOf(errata):
    intUniversalFunc = nu.frompyfunc(int, 1, 1)
    return intUniversalFunc(errata).sum() / len(errata)



# NOTE: recognition lines:  ****************************************************  recognition lines  **********************

RecogLine = namedtuple('RecogLine', 'polyExpr vectorExpr')

def recogLineOfEu(class1, class2):
    vector = nu.flipud(class1.mean - class2.mean)
    vector[0,0] = -vector[0,0]

    x, y = sy.symbols('x,y')
    midlePoint = (class1.mean + class2.mean) / 2.0
    poly = y - midlePoint[1,0] - vector[1,0]/vector[0,0] * (x - midlePoint[0,0])
    return RecogLine(poly, vector)


def recogLineOfWe(class1, class2, weight):
    x,y = sy.symbols('x,y')
    poly = 0
    for var, m1, m2, w in zip([x,y], class1.mean[:,0], class2.mean[:,0], weight[:,0]):
        poly += (w*(var - m1))**2 - (w*(var - m2))**2
    return RecogLine(poly, 'NONE')


def recogLineOfSi(class1, class2):
    vector = class1.mean + class2.mean
    vector /= nu.linalg.norm(vector)

    x,y = sy.symbols('x,y')
    poly = y - vector[1,0]/vector[0,0]*x
    return RecogLine(poly, vector)


def recogLineOfNe(class1, class2):
    global recogLineFile
    if os.path.exists(recogLineFile):
        points = readFileOfPoints(recogLineFile)
        return points
    else:
        print('processing expensive tasks...')
        process1 = mp.Process(target=subprocessForSearchingrecogLineOfSi, args=(class1, class2, recogLineFile, 'u', 'wt'))
        process1.start()
        process1.join(timeout=300)
        process1.terminate()
        if os.path.exists(recogLineFile) == False:
            print('timeout, work not finished.')
            print(error)

        print('second process...\n')
        existingPoints = readFileOfPoints(recogLineFile)
        endPoint = existingPoints[0]
        startPoint = XYPoint([45.0, 60.0])
        process2 = mp.Process(target=subprocessForSearchingrecogLineOfSi, args=(class1, class2, recogLineFile, 'd', 'at', startPoint, endPoint))
        process2.start()
        existingPoints = []
        process2.join(timeout=300)
        process2.terminate()

    points = readFileOfPoints(recogLineFile)
    return points


def subprocessForSearchingrecogLineOfSi(class1, class2, fileName, initDirection, fileMode, startPoint=None, endPoint=None):
    # define important constants
    if startPoint is None:
        meanTotal = (class1.mean + class2.mean) / 2.0
        currentLocation = XYPoint([ meanTotal[0,0], meanTotal[1,0] ])
    else:
        currentLocation = startPoint

    lastDirection = initDirection
    recogLinePoints = []
    criticalPointsOfClass1 = sorted(class1.points, key=currentLocation.distanceFromXYPoint)[0:5]
    criticalPointsOfClass2 = sorted(class2.points, key=currentLocation.distanceFromXYPoint)[0:5]

    # points for plot completed or not
    while currentLocation.distanceFromXYPoint(criticalPointsOfClass1[0]) <= 20 or currentLocation.distanceFromXYPoint(criticalPointsOfClass2[0]) <= 20:

        for counter in range(int(1.0/STEP)):
            currentLocation, lastDirection, isRecogLinePoint = moveToNextLocation(currentLocation, lastDirection, criticalPointsOfClass1, criticalPointsOfClass2)
            if isRecogLinePoint:
                recogLinePoints.append(currentLocation.rawValue)

        if endPoint != None:
            if currentLocation.x <= endPoint.x and currentLocation.y <= endPoint.y:
                break
        # reset criticalPoints every 1000 STEPs
        criticalPointsOfClass1 = sorted(class1.points, key=currentLocation.distanceFromXYPoint)[0:5]
        criticalPointsOfClass2 = sorted(class2.points, key=currentLocation.distanceFromXYPoint)[0:5]
        print('passing loop, current point:', currentLocation.rawValue, 'dir:', lastDirection)

    with open(fileName, fileMode) as file:
        csvFile = csv.writer(file, delimiter=' ')
        csvFile.writerows(recogLinePoints)


def moveToNextLocation(currentLocation, lastDirection, cPoints1, cPoints2):
    minDifferences = []
    for direction in list(validDirections.keys()):
        if validDirections[lastDirection][0] == -validDirections[direction][0] and validDirections[lastDirection][1] == -validDirections[direction][1]:
            continue

        minDistanceOfClass1 = sorted([ point.distanceFromXYPoint(currentLocation.move(direction)) for point in cPoints1 ])[0]
        minDistanceOfClass2 = sorted([ point.distanceFromXYPoint(currentLocation.move(direction)) for point in cPoints2 ])[0]
        difference = {'distance': abs(minDistanceOfClass1 - minDistanceOfClass2), 'direction': direction}
        minDifferences.append(difference)

    bestPoint = sorted(minDifferences, key=lambda difference: difference['distance'])[0]
    nextDirection = bestPoint['direction']
    nextLocation = currentLocation.move(nextDirection)
    isRecogLinePoint =  bestPoint['distance'] <= DELTA

    return (nextLocation, nextDirection, isRecogLinePoint)



# NOTE: - plot results:  *******************************************************  plot results  **************************

def plotResultOf(result, class1, class2, testData, title, recogLine=False):

    # MARK: - plot learned data
    # note that all learned data are represented by color gray with different marker type.
    pl.plot(class1.toNdarray[:,0], class1.toNdarray[:,1], 'o', color='tab:gray', label='preexist class1 points')
    pl.plot(class1.mean[0], class1.mean[1], 'o', color='k', markersize=8)
    pl.plot(class2.toNdarray[:,0], class2.toNdarray[:,1], '+', color='tab:gray', label='preexist class2 points')
    pl.plot(class2.mean[0], class2.mean[1], '+', color='k', markersize=10)

    # MARK: - plot test data distinguished by the correct class of each point. (given by correctClasses array)
    # note that all test points are plotted with color blue.
    testPointsInClass1 = testData.toNdarray[ nu.array(result)==1 ]
    testPointsInClass2 = testData.toNdarray[ nu.array(result)==2 ]
    pl.plot(testPointsInClass1[:,0], testPointsInClass1[:,1], 'o', color='tab:blue', label='test points in class1')
    pl.plot(testPointsInClass2[:,0], testPointsInClass2[:,1], '+', color='tab:blue', label='test points in class2')

    # plot the recognition line (if exists)
    x,y = sy.symbols('x,y')
    if type(recogLine) is RecogLine: # for euclideanDistanceMethod, weightDistanceMethod and similarityMethod
        poly = recogLine.polyExpr
        xValues = nu.arange(-10, 80, 10)
        yValues = [ sy.solve(poly.subs({x: value}), y) for value in xValues ]
        pl.plot(xValues, yValues, color='r')

    elif type(recogLine) is list: # for nearestNeighborMethod
        rawValues = [ point.rawValue for point in recogLine ]
        xValues, yValues = [ value[0] for value in rawValues ], [ value[1] for value in rawValues ]
        pl.plot(xValues, yValues, '.', color='r', markersize=1)

    elif type(recogLine) is Eigen: # for 主成分分析
        for vector in recogLine.vectors:
            midlePoint = (class1.mean + class2.mean) / 2.0
            poly = vector[1,0]/vector[0,0] * (x - midlePoint[0,0]) + midlePoint[1,0]
            xValues = nu.arange(-10, 80, 10)
            yValues = [ poly.subs({x: value}) for value in xValues ]
            pl.plot(xValues, yValues, color='r')

    pl.axis(plotAxisDict[DATASET_ID])
    pl.legend()
    pl.title(title)
    pl.savefig('results/' + title + figureFile)
    pl.figure()
