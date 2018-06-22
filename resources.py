import csv
import numpy as nu
import scipy as sc
from scipy import linalg as la
from collections import namedtuple

LARGE_NUMBER = 1e8


# MARK: - readFile operation:

def readFile(FILE):
    with open(FILE, 'rt') as file:
        lines = csv.reader(file, delimiter=' ')
        points = [  [ float(line[0]), float(line[1]) ] for line in lines  ]
    return nu.array(points)



# MARK: - class definition:

class ClassData():
    def __init__(self, file):
        points = readFile(file)
        self.points = points
        self.len = len(points)

    @property
    def mean(self):
        sum = {'x': 0.0, 'y': 0.0}
        for point in self.points:
            sum['x'] += point[0]
            sum['y'] += point[1]

        x = sum['x'] / self.len
        y = sum['y'] / self.len
        return nu.array([x, y]).reshape(2,1)


    zero = nu.array([0, 0])
    def distanceFrom(self, otherPoint, weight=nu.array([1,1]) ):
        distance = [ nu.linalg.norm( point - weight.reshape(2,1) * otherPoint.reshape(1,2) ) for point in self.points ]
        return nu.array(distance)



class TestData(ClassData):
    def nearestNeighbor(self, class1, class2):
        nearestNeighborArray = []
        for point in self.points:
            leastDistance1 = LARGE_NUMBER
            for class1Point in class1.points:
                distance = nu.linalg.norm(point - class1Point)
                if leastDistance1 > distance:
                    leastDistance1 = distance

            leastDistance2 = LARGE_NUMBER
            for class2Point in class2.points:
                distance = nu.linalg.norm(point - class2Point)
                if leastDistance2 > distance:
                    leastDistance2 = distance

            nearestNeighbor = 1 if leastDistance1 < leastDistance2 else 2
            nearestNeighborArray.append(nearestNeighbor)

        return nu.array(nearestNeighborArray)


    def euclideanDistanceMethod(self, class1, class2):
        results = self.distanceFrom(class1.mean) < self.distanceFrom(class2.mean)
        return nu.array([ 1 if result==True else 2 for result in results ])


    def weightDistanceMethod(self, class1, class2, weight):
        results = self.distanceFrom(class1.mean, weight) < self.distanceFrom(class2.mean, weight)
        return nu.array([ 1 if result==True else 2 for result in results ])


    def similarityMethod(self, class1, class2):
        inners1 = nu.inner(self.points, class1.mean.T)
        # print(self.points, 'shape:', self.points.shape)
        # print(class1.mean, 'shape:', class1.mean.shape)
        # print(inners1, 'shape:', inners1.shape)
        distanceProduct1 = nu.linalg.norm(class1.mean) * self.distanceFrom(ClassData.zero)
        print(distanceProduct1, 'shape', distanceProduct1.shape)
        angle1 = inners1 / distanceProduct1.reshape(40,1)
        print(angle1, 'shape', angle1.shape)

        inners2 = nu.inner(self.points, class2.mean.T)
        distanceProduct2 = nu.linalg.norm(class1.mean) * self.distanceFrom(ClassData.zero)
        angle2 = inners2 / distanceProduct2

        results = angle1 < angle2
        return nu.array([ 1 if result==True else 2 for result in results ])



# MARK: - covariances:

def covarianceWithinClasses(class1, class2):
    ''' within-class-covariance '''
    mean1 = class1.mean
    sum1 = 0
    for point in class1.points:
        point = point.reshape(2,1)
        sum1 += (point - mean1) * (point - mean1).T

    mean2 = class2.mean
    sum2 = 0
    for point in class2.points:
        point = point.reshape(2,1)
        sum2 += (point - mean2) * (point - mean2).T

    return (sum1 + sum2) / (class1.len + class2.len)


def covarianceBetweenClasses(class1, class2):
    ''' between-class-convariance '''
    meanTotal = (class1.mean + class2.mean) / 2.0
    a = class1.len * (class1.mean - meanTotal) * (class1.mean - meanTotal).T
    b = class2.len * (class2.mean - meanTotal) * (class2.mean - meanTotal).T
    return (a + b) / (class1.len + class2.len)


def covarianceTotal(class1, class2):
    ''' total covariance '''
    meanTotal = (class1.mean + class2.mean) / 2.0
    allPoints = nu.insert(class1.points, class1.len, class2.points, axis=0)

    sum = 0
    for point in allPoints:
        point = point.reshape(2,1)
        sum += (point - meanTotal) * (point - meanTotal).T
    return sum / (class1.len + class2.len)



# MARK: - eigen value and vectors:

Eigen = namedtuple('Eigen', 'values vectors')

def discriminantOf(class1, class2):
    A = covarianceBetweenClasses(class1, class2)
    B = covarianceWithinClasses(class1, class2)
    values, vectors = la.eig(A, B)
    vectors = nu.array([ vector/nu.linalg.norm(vector) for vector in vectors ])

    return Eigen(values, vectors)


class Projection():
    def __init__(self, classData, vector):
        self.points = nu.array([ nu.inner(point, vector) * vector for point in classData.points ])
        self.len = classData.len


def pricipalComponentOf(class1, class2):
    A = covarianceTotal(class1, class2)
    B = nu.eye(2)
    values, vectors = la.eig(A, B)
    vectors = nu.array([ vector/nu.linalg.norm(vector) for vector in vectors ])
    return Eigen(values, vectors)
