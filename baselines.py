import numpy
import scipy.cluster.hierarchy
from collections import defaultdict
import idcHelper
import math
import baselinesADMM


def getTikhonovClustering(sampledCov, lambdaValue, Q, eigVals):
    n = sampledCov.shape[0]
    
    # invCov = numpy.linalg.inv(sampledCov + lambdaValue * numpy.eye(n))

    # eigVals, eigVecs = numpy.linalg.eigh(sampledCov)
    # Q = numpy.asmatrix(eigVecs)

    allNewEigVals = 1.0 / (eigVals + lambdaValue)
    
    n = eigVals.shape[0]
    ones = numpy.asmatrix(numpy.ones(n))
    eigValsTimesQ = numpy.asmatrix(numpy.multiply(ones.transpose() * allNewEigVals, Q))
    invCov = Q * eigValsTimesQ.transpose()
    
    return invCov, (Q, eigVals)


def fastAllL2(sampledCov, lambdaValue):
    p = 2 * lambdaValue
    invCov = idcHelper.getFastDiagonalSolution(- 1 * (sampledCov), p)
    return invCov, invCov







def getClusteringGraphicalLassoVanilla(sampledCov, lambdaValue, penalizationType, initMatrix):
    if penalizationType == "tikhonov" or penalizationType == "tikhonov2":
        eigVecs = initMatrix[0]
        eigVals = initMatrix[1]
        return getTikhonovClustering(sampledCov, lambdaValue, eigVecs, eigVals)
    elif penalizationType == "allL1ExceptDiagonalADMM" or penalizationType == "allL1ExceptDiagonalADMM2":
        return baselinesADMM.fastGraphicalLassoExceptDiagonal(sampledCov, lambdaValue, initMatrix)
    elif penalizationType == "sampleCov":
        return numpy.asmatrix(numpy.copy(sampledCov)), None
    else:
        assert(False)
    


# double checked
def hierarchicalClustering(A, numberOfClusters, linkageType):
    assert(linkageType == "single" or linkageType == "average")
    distances = numpy.max(numpy.abs(A)) - numpy.abs(A)
    # distances = - numpy.abs(A)
    
    if linkageType == "single":
        linkageM = scipy.cluster.hierarchy.single(distances)
    else:
        linkageM = scipy.cluster.hierarchy.average(distances)    
        
    # print linkageM
    numberOfVariables = A.shape[0]
    return getClusteringFromLinkageM(numberOfVariables, linkageM, numberOfClusters)


# checked
def setAllIds(hiddenVarIds, allVarsInCluster, clusterLabelId):
    for varId in allVarsInCluster:
        assert(hiddenVarIds[varId] == 0)
        hiddenVarIds[varId] = clusterLabelId
    

# checked
def getClusteringFromLinkageM(numberOfVariables, linkageM, numberOfClusters):
    numberOfIterations = linkageM.shape[0]
    
    clusterIdToVarIds = {}
    for i in xrange(numberOfVariables):
        clusterIdToVarIds[i] = set([i])
    
    subordinateClusters = defaultdict(lambda: False)   
    for i in xrange(numberOfIterations - numberOfClusters + 1):
        clusterId1 = linkageM[i,0]
        clusterId2 = linkageM[i,1]
        clusterIdToVarIds[numberOfVariables + i] = clusterIdToVarIds[clusterId1] | clusterIdToVarIds[clusterId2]
        subordinateClusters[clusterId1] = True
        subordinateClusters[clusterId2] = True

    hiddenVarIds = numpy.zeros(numberOfVariables, dtype = numpy.int_)

    clusterLabelId = 1
    for clusterId in xrange(numberOfVariables + numberOfIterations - numberOfClusters + 1):
        if not subordinateClusters[clusterId]:
            setAllIds(hiddenVarIds, clusterIdToVarIds[clusterId], clusterLabelId)
            clusterLabelId += 1
     
    assert(clusterLabelId == numberOfClusters + 1)        
    return hiddenVarIds


# double checked
# tested
def getRandomSetM(p, card):
    allPairs = set()
    
    while(len(allPairs) < card):
        i1 = numpy.random.randint(low = 0, high = p)
        i2 = numpy.random.randint(low = 0, high = p)
        if i1 != i2:
            allPairs.add(str(i1) + "\t" + str(i2))
            allPairs.add(str(i2) + "\t" + str(i1))
    
    M = []
    for pair in allPairs:
        i1 = int(pair.split("\t")[0])
        i2 = int(pair.split("\t")[1])
        M.append([i1,i2])
    
    # print "len(M) = ", len(M)
    assert(len(M) >= card)
    return M

# double checked
# tested
def getSStar(Sabs, M):
    SStar = numpy.copy(Sabs)
    MasMatrix = numpy.zeros_like(SStar)
    for pair in M:
        i1 = pair[0]
        i2 = pair[1]
        SStar[i1, i2] = 0
        MasMatrix[i1,i2] = 1
    
#     print "MasMatrix = "
#     idcHelper.showMatrix(MasMatrix)
#     print "S = "
#     idcHelper.showMatrix(Sabs)
#     print "SStar = "
#     idcHelper.showMatrix(SStar)
    
    nonzeroCountsInRow = SStar.shape[0] - numpy.sum(MasMatrix, 1)
    allRowSums = numpy.sum(SStar, 1)
    allRowMeans = allRowSums / nonzeroCountsInRow
    
#     print "all RowMeans correct = "
#     print allRowMeans
#     print "all RowMeans wrong = "
#     print numpy.mean(SStar, 1)
#     assert(False)
    
    for pair in M:
        i1 = pair[0]
        i2 = pair[1]
        SStar[i1, i2] = 0.5 * (allRowMeans[i1] + allRowMeans[i2])
    
    return SStar

# double checked
# tested
def getMSE(SStar, clusteringResult, correctS, M):
    assert(numpy.min(clusteringResult) == 1 and numpy.max(clusteringResult) > 1)
    
    betweenClusterS = numpy.copy(SStar)
    p = SStar.shape[0]
    numberOfClusters = numpy.max(clusteringResult)
    clusterSizes = numpy.zeros(numberOfClusters)
    for i in xrange(0, p):
        clusterSizes[clusteringResult[i] - 1] += 1
    
    clusterSumEntries = numpy.zeros(numberOfClusters)
    clusterSumEntriesCount = numpy.zeros(numberOfClusters)
    for i in xrange(0, p):
        for j in xrange(i, p):
            if clusteringResult[i] == clusteringResult[j]:
                betweenClusterS[i, j] = 0
                betweenClusterS[j, i] = 0
                if i != j:
                    clusterSumEntries[clusteringResult[i] - 1] += 2 * SStar[i, j]
                    clusterSumEntriesCount[clusteringResult[i] - 1] += 2
    
    withinClusterVals = numpy.zeros(numberOfClusters)
    for z in xrange(0, numberOfClusters):
        assert(clusterSumEntriesCount[z] == clusterSizes[z] * (clusterSizes[z] - 1))
        if (clusterSizes[z] * (clusterSizes[z] - 1)) == 0:
            # happens if the cluster contains only one variable
            assert(clusterSizes[z] == 1)
            withinClusterVals[z] = float("NaN")
        else:
            withinClusterVals[z] = clusterSumEntries[z] / (clusterSizes[z] * (clusterSizes[z] - 1))
        
    
    assert((p * p - numpy.sum(numpy.square(clusterSizes)) > 0.0))
    betweenClusterVal = numpy.sum(betweenClusterS) / (p * p - numpy.sum(numpy.square(clusterSizes)))
    assert(not math.isnan(betweenClusterVal))
    
    sumSquaredError = 0.0
    for pair in M:
        i1 = pair[0]
        i2 = pair[1]
        assert(i1 != i2)
        bVal = 0
        if clusteringResult[i1] == clusteringResult[i2]:
            bVal = withinClusterVals[clusteringResult[i1] - 1]
            assert(not math.isnan(bVal))
        else:
            bVal = betweenClusterVal
        sumSquaredError += (correctS[i1,i2] - bVal) ** 2
    
    return sumSquaredError / float(len(M))

# double checked
# determines number of clusters for hierarchical clustering using Algorithm 2 of 
# "The cluster graphical lasso for improved estimation of Gaussian graphical models", 2015
# "single" = single linkage clustering, "average" = average linkage clustering
def mseClusterEval(sampleCov, clustersCandidatesNrs, clusteringType):
    assert(clusteringType == "single" or clusteringType == "average")
    assert(clustersCandidatesNrs[0] >= 2)
    p = sampleCov.shape[0]
    T = min(100, int(p / 2))
    Sabs = numpy.abs(sampleCov)
    sizeOfM = int(p * (p-1) / (2.0 * T))
    assert(sizeOfM >= 1)
    
    nrCandidates = clustersCandidatesNrs.shape[0]
    assert(clustersCandidatesNrs[nrCandidates - 1] <= p)
    
    allMSEs = numpy.zeros((nrCandidates, T))
    # print "T = ", T
    # print "nrCandidates = ", nrCandidates
    # print "sizeOfM = ", sizeOfM
    
    for i in xrange(T):
        print "i = ", i
        M = getRandomSetM(p, sizeOfM)
        
        # print "M = ", M
        # assert(False)
        
        SStar = getSStar(Sabs, M)
        
        for z in xrange(nrCandidates):
            numberOfClusters = clustersCandidatesNrs[z]
            clusteringResult = hierarchicalClustering(SStar, numberOfClusters, clusteringType)
            # print "clusteringResult = ", clusteringResult
            # print "numberOfClusters = ", numberOfClusters
            allMSEs[z, i] = getMSE(SStar, clusteringResult, Sabs, M)
            
    meanMSEs = numpy.mean(allMSEs, axis = 1)
    stdMSEs = numpy.std(allMSEs, axis = 1)
    
    minKindex = None
    for z in xrange(nrCandidates - 2, -1, -1):
        if meanMSEs[z] <= meanMSEs[z + 1] + 1.5 * stdMSEs[z + 1]:
            minKindex = z
            # print "this is a candidate = ", clustersCandidatesNrs[minKindex]
        
    if (minKindex is None):
        print "WARNING CANNOT DECIDE CLUSTER NR - USE minKindex = 0"
        minKindex = 0
    
    return clustersCandidatesNrs[minKindex]




def test():
    # sampleCov = numpy.asarray([[0.05, -0.8, 0.6, 0.3], [-0.8, 1.0, 0.05, 0.2], [0.6, 0.05, 0.05, 0.1], [0.3, 0.2, 0.1, 1.0]])
    sampleCov = numpy.asarray([[1.0, 0.8, 0.6, 0.3], [0.8, 1.0, 0.5, 0.2], [0.6, 0.5, 1.0, 0.1], [0.3, 0.2, 0.1, 1.0]])
    print "sampleCov = "
    print sampleCov
    clustersCandidatesNrs = numpy.asarray([2,3])
    bestClusterNr = mseClusterEval(sampleCov, clustersCandidatesNrs, "single")
    print "bestClusterNr = ", bestClusterNr
    clustering = hierarchicalClustering(sampleCov, 1, "single")
    print clustering
# test()
