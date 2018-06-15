import math
import numpy
import scipy.special
import scipy.misc

def isNumpyObject(obj):
    return (type(obj).__module__ == "numpy")

def countEdges(A):
    ZERO_APPROX = 0.000001
    thresholdedMatrix = numpy.copy(A)
    thresholdedMatrix[numpy.abs(A) < ZERO_APPROX] = 0
    return numpy.count_nonzero(thresholdedMatrix) / 2.0

# tested
def getBinaryClusterAssignment(assignmentRepNr, nrDataPoints):
    assert(assignmentRepNr < 2 ** (nrDataPoints - 1))
    assignment = numpy.ones(nrDataPoints, dtype = numpy.int_)
    
    binaryString = str(bin(assignmentRepNr)).split("b")[1]
    # print binaryString
    assert(len(binaryString) < nrDataPoints)
    for i, val in enumerate(reversed(list(binaryString))):
        clusterId = int(val) + 1
        assert(clusterId == 1 or clusterId == 2)
        assignment[nrDataPoints - i - 1] = clusterId
      
    return assignment

# checked
# each row of data corresponds to one data sample
# normalizes data such that each dimension (=feature/variable) has mean 0 and variance 1 
def normalizeData(data):
    meanVec = numpy.mean(data, axis = 0)
    
    # print "meanVec.shape[0] = ", meanVec.shape[0]
    assert(meanVec.shape[0] >= 2 and meanVec.shape[0] <= 10000) # number of dimensions
    stdVec = numpy.std(data, axis = 0)
    assert(numpy.sum(stdVec == 0) == 0)
    
    # print "meanVec = "
    # print meanVec
    # print "stdVec = "
    # print stdVec
    # assert(False)
    normalizedData =  (data - meanVec) / stdVec
    
    return normalizedData


def centerData(data):
    meanVec = numpy.mean(data, axis = 0)
    normalizedData =  (data - meanVec)
    return normalizedData


# checked
def simulatedAnnealing(probsEachAssignment, T): 
    assert(T > 0.0)   
    logProbsEachAssignment = numpy.log(probsEachAssignment)
    logProbsEachAssignment = (1.0 / T) * logProbsEachAssignment
    normalization = scipy.misc.logsumexp(logProbsEachAssignment)
    probs = numpy.exp(logProbsEachAssignment - normalization)
    
    # print "probsEachAssignment = ", probsEachAssignment
    # print "logProbsEachAssignment = ", logProbsEachAssignment
    # print "normalization = ", normalization
    # print "probs = ", probs
    # assert(numpy.all(numpy.greater(probsEachAssignment, 0.0)))
    
    return probs

# checked
def multivariateBetaLog(alpha):
    assert(isinstance(alpha, numpy.ndarray))
    n = numpy.sum(alpha)
    totalLogGammaPart = - scipy.special.gammaln(n)
    
    for i in xrange(alpha.shape[0]):
        totalLogGammaPart += scipy.special.gammaln(alpha[i])
    
    return totalLogGammaPart

# tested
def getSquaredDistMatrix(data):
    NR_DATA_POINTS = data.shape[0]
    distMatrix = numpy.zeros(shape = (NR_DATA_POINTS, NR_DATA_POINTS))
    for i in xrange(NR_DATA_POINTS):
        for j in xrange(i):
            distMatrix[i,j] = numpy.sum((data[i,:] - data[j,:]) ** 2)
            distMatrix[j,i] = distMatrix[i,j]
            
    return distMatrix

# USAGE: spectral clustering with rbf kernel is parameterized by gamma and not by l
# reference: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
def getGamma(l):
    return 0.5 * (1.0 / (float(l) ** 2)) 

# get covariance matrix using squared exponential function
def getSECovarianceMatrix(data, l):
    squaredDistMatrix = getSquaredDistMatrix(data)
    seMatrix = -0.5 * (1.0 / (float(l) ** 2)) * squaredDistMatrix
    return numpy.exp(seMatrix)

# returns the median of the euclidean distance matrix
def getMedianEuclideanDistance(data):
    squaredDistMatrix = getSquaredDistMatrix(data)
    mediaDistSqr = numpy.median(squaredDistMatrix)
    mediaDist = numpy.sqrt(mediaDistSqr)
    return numpy.asscalar(mediaDist)
 
# tested
def getSubK(fullK, indices):
    subColumns = fullK[:,indices]
    subColumnsAndRows = subColumns[indices, :]
    return subColumnsAndRows

def getObjectIds(assignment, clusterId):
    return numpy.where(assignment == clusterId)[0]
    
    
# only for debugging
def evalQuadFormEmpty(m, v):
    return 0.0

# tested
def evalQuadForm(m, v):
    assert(False)
    assert(len(v.shape) == 1)
    assert(len(m.shape) == 2)
    assert(m.shape[0] == m.shape[1] and m.shape[1] == v.shape[0])
    
    leftR = numpy.dot(v.transpose(), m)
    result = numpy.dot(leftR, v)
    return result

# tested
def evalQuadFormNew(m, v):
    assert(len(v.shape) == 1)
    assert(len(m.shape) == 2)
    assert(m.shape[0] == m.shape[1] and m.shape[1] == v.shape[0])
    
    leftR = numpy.dot(v.transpose(), m)
    result = numpy.dot(leftR, v)
    return result

# tested
def equalsApprox(n1, n2):
    return math.fabs(n1 - n2) < 0.00001

# tested
def multinomialDist_logPMF(eventProbs, observations):
    assert(numpy.all(numpy.greater(eventProbs, 0.0)))
    assert(numpy.all(numpy.greater_equal(observations, 0)))
    assert(len(eventProbs) == len(observations))
    assert(equalsApprox(sum(eventProbs), 1.0))
    
    n = sum(observations)
    totalLogGammaPart = scipy.special.gammaln(n + 1.0)
    
    for obsCount in observations:
        totalLogGammaPart -= scipy.special.gammaln(obsCount + 1.0)
    
    totalLogProb = totalLogGammaPart
    for i, prob in enumerate(eventProbs):
        obsCount = observations[i]
        totalLogProb += obsCount * math.log(prob)
        
    return totalLogProb



# tested
def getCounts(assignment, numberOfClusters):
    assert(isinstance(assignment, numpy.ndarray))
    counts = numpy.zeros(numberOfClusters, dtype = numpy.int_)
    for clusterId in xrange(numberOfClusters):
        counts[clusterId] = numpy.sum(assignment == (clusterId + 1))
    
    return counts




def someTests():

    squaredDistMatrix = numpy.asarray([[3, 3, 0], [12, 11, 10], [2, 4,5]])
    print(squaredDistMatrix)
    mediaDistSqr = numpy.median(squaredDistMatrix)
    print(mediaDistSqr)
    print(numpy.sqrt(mediaDistSqr))

    testA = [0.1, 0.5]
    # print sum(testA)
    
    probs = [0.2,0.7, 0.1]
    
    print numpy.all(numpy.greater(probs,0.0))
    
    # obs = [3, 7, 1]
    obs = numpy.zeros(3, dtype = numpy.int_)
    obs[0] = 3
    obs[1] = 7
    obs[2] = 1
    print len(obs)
    
    # x = obs[0]
    # n = sum(obs)
    # p = probs[0]
    # print scipy.stats.binom.logpmf(x, n , p)
    
    # assignment = numpy.asarray([1, 2, 1, 3, 3, 3, 3, 2, 1, 1])
    # assignment = [1, 2, 1, 3, 3, 3, 3, 2, 1]
    assignment = numpy.asarray([3, 3, 0])
    
    
    print "counts = "
    obs = getCounts(assignment, 3)
    print obs
    print multinomialDist_logPMF(probs, obs)
    
    # log sum exp test:
    l1 = math.log(0.1)
    l2 = math.log(0.2)
    print "l1 = ", l1
    print "l2 = ", l2
    print "log(sum) = ", math.log(0.1 + 0.2)
    print "logsumexp = ", scipy.misc.logsumexp([l1, l2])
    
    
    # clusterPriorProbs = numpy.repeat(1.0/3.0, 3)
    clusterPriorProbs = [0.3,0.3,0.4]
    print "original probs = "
    print clusterPriorProbs
    
    print "annealed probs = "
    print simulatedAnnealing(clusterPriorProbs, 0.5)
    
    
    numberOfClusters = 3
    allClusterIds = numpy.arange(1,numberOfClusters + 1,1)
    
    numberOfDataPoints = 100
    assignment = numpy.random.choice(allClusterIds, numberOfDataPoints)
    
    # print numpy.random.choice(allClusterIds, size = 20, p = clusterPriorProbs)

# someTests()