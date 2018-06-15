import numpy
import scipy.stats
from tabulate import tabulate
import sklearn.cluster
import sklearn.metrics
import scipy.special
import scipy.linalg


def createFullX(p, allX):
    X = numpy.asmatrix(numpy.zeros((p,p)))
   
    startId = 0
    for j in xrange(len(allX)):
        endId = startId + allX[j].shape[0]
        X[startId:endId, startId:endId] = allX[j]
        startId = endId

    return X

# assume that A and B are symmetric matrices
def matrixInnerProdSymmetric(A,B):
    return numpy.sum(numpy.multiply(A,B))


def matrixInnerProd(A,B):
    assert(type(A) == numpy.matrixlib.defmatrix.matrix)
    assert(type(B) == numpy.matrixlib.defmatrix.matrix)
    return numpy.trace(A * B)

def showLowestEigVals(m):
    if type(m) == numpy.matrixlib.defmatrix.matrix:
        eigVals, eigVecs = numpy.linalg.eigh(m)
        print "lowest eigenvalues = ", eigVals[0:min(5,m.shape[0])]
    else:
        print "lowest eigenvalue = ", m
    return

# checked
def assertValidAndNotInfinite(value):
    assert((not numpy.isnan(value)) and numpy.isfinite(value))

# checked
def getIdsOfEachCluster(clusterAssignments):
    
    p = clusterAssignments.shape[0]
    assert(p >= 2)
    assert(numpy.min(clusterAssignments) == 1)
    numberOfClusters = numpy.max(clusterAssignments)
    
    idsForEachCluster = []
    for z in xrange(numberOfClusters):
        idsForEachCluster.append([])
    
    for i in xrange(p):
        idsForEachCluster[clusterAssignments[i] - 1].append(i)
        
    return idsForEachCluster

# reading checked
def getIdsOfEachClusterAsSet(clusterAssignments):
    
    p = clusterAssignments.shape[0]
    assert(p >= 2)
    assert(numpy.min(clusterAssignments) == 1)
    numberOfClusters = numpy.max(clusterAssignments)
    
    idsForEachCluster = {}
    for clusterId in xrange(1, numberOfClusters + 1, 1):
        idsForEachCluster[clusterId] = set()
    
    for i in xrange(p):
        idsForEachCluster[clusterAssignments[i]].add(i)
        
    return idsForEachCluster

# checked
def getSubCovarianceMatrix(S, varIds):
    nrVariables = len(varIds)
    subCov = numpy.zeros((nrVariables,nrVariables))
    for i in xrange(0, nrVariables, 1):
        for j in xrange(i, nrVariables, 1):
            subCov[i,j] = S[varIds[i], varIds[j]]
            subCov[j,i] = subCov[i,j]
    
    return numpy.asmatrix(subCov)


# double checked
def getBlockCovariance(S, clusterAssignments):

    idsInClusters = getIdsOfEachCluster(clusterAssignments)
    
    allBlockCovs = []
    for varIdsInCurrentCluster in idsInClusters:
        blockSize = len(varIdsInCurrentCluster)
        blockCov = numpy.zeros((blockSize,blockSize))
        for i in xrange(0, blockSize, 1):
            for j in xrange(i, blockSize, 1):
                blockCov[i,j] = S[varIdsInCurrentCluster[i], varIdsInCurrentCluster[j]]
                blockCov[j,i] = blockCov[i,j]
        allBlockCovs.append(numpy.asmatrix(blockCov))

    return allBlockCovs

def showLogLikelihood(objValue):
    print("log-likelihood = %.3f" % (-objValue))
    
def conv2corrMatrix(covMatrix):
    assert(False)
    normM = numpy.diag(1.0 / (numpy.sqrt(numpy.diag(covMatrix))))
    corrM = numpy.dot(normM, covMatrix)
    corrM = numpy.dot(corrM, normM)
    return corrM


        
# checked
def getAllConnectingNodes(connectedNodes, M, focusId):
    
    connectedNodes[focusId] = True
    for i in xrange(M.shape[0]):
        if (connectedNodes[i] == False) and (M[focusId, i] > 0):
            getAllConnectingNodes(connectedNodes, M, i)
    
    return connectedNodes

# checked
def isConnected(Morig):
    M = numpy.abs(Morig)
    NUMBER_OF_VARIABLES = M.shape[0]
    connectedNodes = numpy.zeros(NUMBER_OF_VARIABLES) > 0
    connectedNodes = getAllConnectingNodes(connectedNodes, M, 0)
    return numpy.all(connectedNodes)

def shrinke(corrMatrix, shrinkageRatio):
    
    if shrinkageRatio is None:
        return corrMatrix
    else:
        diagBackup = numpy.diag(corrMatrix)
        scaledCorrMatrix = corrMatrix * shrinkageRatio
        scaledCorrMatrix[numpy.diag_indices_from(scaledCorrMatrix)] = diagBackup
        return scaledCorrMatrix

# returns uniform noise matrix with standard deviation = std
# def sampleUniformSymmetricNoise(p, std):
#     a = numpy.sqrt(3.0) * std
#     print "MAXIMUM NOISE VALUE = ", a
#     noiseM = numpy.random.uniform(low = -a, high = a, size = (p, p))
#     for i in xrange(p):
#         for j in xrange(i + 1, p, 1):
#             noiseM[j, i] = noiseM[i,j]
#     
#     return noiseM


def sampleClusterCovMatrix(size, sparsityRatio):
    initVector = numpy.random.normal(loc = 0.0, scale = 1.0, size = size)
    scaleMatrix = numpy.outer(initVector, initVector) + 0.1 * numpy.eye(size)
    covMatrix = scipy.stats.wishart.rvs(df = scaleMatrix.shape[0], scale = scaleMatrix, size=1)
    
    if sparsityRatio > 0.0:
        precisionMatrix = numpy.copy(covMatrix) # numpy.linalg.inv(covMatrix)
        desiredZeroEntries = size * size * sparsityRatio
        zeroEntries = 0
        while zeroEntries < desiredZeroEntries:
            i = numpy.random.randint(low = 0, high = size)
            j = numpy.random.randint(low = 0, high = size)
            if i != j and precisionMatrix[i,j] != 0.0: 
                backup = precisionMatrix[i,j]
                precisionMatrix[i,j] = 0.0
                precisionMatrix[j,i] = 0.0
                if isConnected(precisionMatrix):
                    zeroEntries += 1
                else:
                    assert(False) # just in case, this is probably too sparse, better decrease sparsityRatio
                    precisionMatrix[i,j] = backup
                    precisionMatrix[j,i] = backup
                    
        assert(isConnected(precisionMatrix)) 
        eigVals, eigVecs = numpy.linalg.eigh(precisionMatrix)
        if eigVals[0] < 0.001:
            # print "eigVals[0] = ", eigVals[0]
            reg = numpy.abs(eigVals[0]) + 0.001
            precisionMatrix = precisionMatrix + reg * numpy.eye(size)
        covMatrix = precisionMatrix # numpy.linalg.inv(precisionMatrix)
        
    return covMatrix

# checked
# calculates the normalization factor Z_{nu0, Sigma0} of the inverse Wishart
def getNormalizationConstantInvWishart(nu0, Sigma0):
    d = Sigma0.shape[0]
    sign, detSigma0 = numpy.linalg.slogdet(Sigma0)
    assert(sign > 0)
    assert(nu0 >= d)
      
    firstFac = 0.5 * nu0 * detSigma0
    secondFac = 0.5 * (d * nu0) * numpy.log(2)
    thirdFac = scipy.special.multigammaln(nu0 * 0.5, d)
      
    return - firstFac + secondFac + thirdFac


def getNormalizationConstantWishart(nu0, Sigma0):
    d = Sigma0.shape[0]
    sign, detSigma0 = numpy.linalg.slogdet(Sigma0)
    assert(sign > 0)
    assert(nu0 >= d)
      
    firstFac = 0.5 * nu0 * detSigma0
    secondFac = 0.5 * (d * nu0) * numpy.log(2)
    thirdFac = scipy.special.multigammaln(nu0 * 0.5, d)
      
    return firstFac + secondFac + thirdFac

# checked
# returns the log pdf for covariance matrix "covMat"
def getInverseWishartLogPDF(nu0, Sigma0, covMat):
    p = covMat.shape[0]
    sign, logDetCovMat = numpy.linalg.slogdet(covMat)
    invCov = numpy.asmatrix(numpy.linalg.inv(covMat))
    Sigma0 = numpy.asmatrix(Sigma0)
    assert(sign > 0)
    assert(type(Sigma0) == numpy.matrixlib.defmatrix.matrix)
    assert(type(invCov) == numpy.matrixlib.defmatrix.matrix)
    unnormalizedLogProb = - 0.5 * ((nu0 + p + 1) * logDetCovMat + numpy.trace(Sigma0 * invCov))
    return unnormalizedLogProb - getNormalizationConstantInvWishart(nu0, Sigma0)

# checked
def getWishartLogPDF(nu0, Sigma0, mat):
    p = mat.shape[0]
    sign, logDetCovMat = numpy.linalg.slogdet(mat)
    mat = numpy.asmatrix(mat)
    Sigma0 = numpy.asmatrix(Sigma0)
    invSigma0 = numpy.asmatrix(numpy.linalg.inv(Sigma0))
    assert(sign > 0)
    assert(type(Sigma0) == numpy.matrixlib.defmatrix.matrix)
    assert(type(mat) == numpy.matrixlib.defmatrix.matrix)
    unnormalizedLogProb = 0.5 * ((nu0 - p - 1) * logDetCovMat - numpy.trace(invSigma0 * mat))
    return unnormalizedLogProb - getNormalizationConstantWishart(nu0, Sigma0)

# # assume non-informative prior
# def getLogMarginalOfInvWishartOneBlock(sampleCovBlock, n, priorScale):
#     d = sampleCovBlock.shape[0]
#     priorPart = getNormalizationConstantInvWishart(nu0 = d + 1, Sigma0 = priorScale * numpy.eye(d))
#     posteriorPart = getNormalizationConstantInvWishart(nu0 = d + 1 + n, Sigma0 = priorScale * numpy.eye(d) + n * sampleCovBlock)
#     return  posteriorPart - priorPart


def getLogDet(covMatrix):
    if (len(covMatrix.shape) == 2):
        sign, logdet = numpy.linalg.slogdet(covMatrix)
        assert(sign > 0)
    else:
        # covMatrix is single number
        assert(covMatrix > 0)
        return numpy.log(covMatrix)
    
    return logdet

def getLogDetWithNegInf(covMatrix):
    if (len(covMatrix.shape) == 2):
        sign, logdet = numpy.linalg.slogdet(covMatrix)
        if sign <= 0:
            return float("-inf")
        assert(sign > 0)
    else:
        # covMatrix is single number
        assert(covMatrix > 0)
        return numpy.log(covMatrix)
    
    return logdet


def getCorrelation(x1, x2):
    assert(len(x1.shape) == 1 and len(x2.shape) == 1)
    X = numpy.vstack((x1,x2))
    corr = numpy.corrcoef(X)
    assert(len(corr.shape) == 2 and corr.shape[0] == 2 and corr.shape[1] == 2)
    return corr[0,1]


def tests():
    # vec1 = numpy.asarray([1.0, 1.0, -1.0, 0.0])
    # vec2 = numpy.asarray([-1.0, -1.0, 1.0, 1.0])
    vec1 = numpy.asarray([0.8, 0.8, 0.8, 0.9, 0.9, 0.9])
    vec2 = numpy.asarray([1000.0, 1000.0, 1000.0, 2000.0, 2000.0, 2000.0])
    
    
    def getCorrelation(x1, x2):
        assert(len(x1.shape) == 1 and len(x2.shape) == 1)
        X = numpy.vstack((x1,x2))
        corr = numpy.corrcoef(X)
        assert(len(corr.shape) == 2 and corr.shape[0] == 2 and corr.shape[1] == 2)
        return corr[0,1]
    
    print getCorrelation(vec1, vec2)
    return

# tests()

def getLogLikelihood(invCov, dataCov):
    assert(type(invCov) == numpy.matrixlib.defmatrix.matrix)
    assert(type(dataCov) == numpy.matrixlib.defmatrix.matrix)
    return getLogDet(invCov) - numpy.trace(invCov * dataCov)

def getAvgAndStdWithDigitRound(vec, digit):
    if vec.shape[0] == 1:
        m = numpy.mean(vec)
        return str(round(m, digit))
    else:
        m = numpy.mean(vec)
        s = numpy.std(vec)
        return str(round(m, digit)) + " (" + str(round(s, digit)) + ")"


def getAvgAndStd(vec):
    return getAvgAndStdWithDigitRound(vec, 2)

def getAvg(vec):
    m = numpy.mean(vec)
    return str(round(m, 2))

# calculates L_{rw} according to "A tutorial on spectral clustering", Section 3.2
def getLaplacianNormalization(L, diagVec):
    assert(type(L) == numpy.matrixlib.defmatrix.matrix)
    degreeVecInv = 1.0 / diagVec
    Dinv = numpy.asmatrix(numpy.diag(degreeVecInv))
    assert(type(Dinv) == numpy.matrixlib.defmatrix.matrix)
    return Dinv * L




# tested
def getLaplacianSquare(A, variant):
    assert(variant == "noNormSC") # or variant == "normSC")
    assert(type(A) == numpy.matrixlib.defmatrix.matrix)
    n = A.shape[0]
    
    Asquare = numpy.square(A)
    onesVec = numpy.asmatrix(numpy.ones(n)).transpose()
    
    assert(type(Asquare) == numpy.matrixlib.defmatrix.matrix)
    assert(type(onesVec) == numpy.matrixlib.defmatrix.matrix)
    diagVec = numpy.reshape(numpy.asarray(Asquare * onesVec), n)
    L = numpy.diag(diagVec) - Asquare
    
    if variant == "normSC":
        L = getLaplacianNormalization(L, diagVec)
        
    return L


def getLaplacianAbs(A, variant):
    assert(variant == "noNormSC")#  or variant == "normSC")
    assert(type(A) == numpy.matrixlib.defmatrix.matrix)
    n = A.shape[0]
     
    Aabs = numpy.abs(A)
    onesVec = numpy.asmatrix(numpy.ones(n)).transpose()
     
    assert(type(Aabs) == numpy.matrixlib.defmatrix.matrix)
    assert(type(onesVec) == numpy.matrixlib.defmatrix.matrix)
    diagVec = numpy.reshape(numpy.asarray(Aabs * onesVec), n)
    L = numpy.diag(diagVec) - Aabs
    
    if variant == "normSC":
        L = getLaplacianNormalization(L, diagVec)
    
    return L


def letLabelsStartWith1(clusterlabels):
    if (numpy.min(clusterlabels) == 1):
        return clusterlabels
    else:
        assert(numpy.min(clusterlabels) == 0)
        return (clusterlabels + 1)


def getClusteringFast(eigVecsL, m):
    # print "first m eigenvectors = "
    lowestEigVecs = eigVecsL[:, 0:m]
    kmeans = sklearn.cluster.KMeans(n_clusters=m).fit(lowestEigVecs)  # sometimes label 0 is missing => use letLabelsStartWith1
    # print "kmeans.labels_ = "
    # print kmeans.labels_
    return letLabelsStartWith1(kmeans.labels_) 

            

def showVector(vec):
    print getVectorAsStr(vec)

def getVectorAsStr(vec):
    if vec is None:
        return "None"
    else:
        allNumbersAsString = [str(z) for z in vec]
        return "[" + " ".join(allNumbersAsString) + "]"
    return


def showMatrix(M):
    print(tabulate(numpy.asarray(M), tablefmt="latex", floatfmt=".2f"))

def getSparsePrecisionMatrix(assignments, precisionM):
    assert(precisionM.shape[0] == precisionM.shape[1])
    assert(precisionM.shape[0] == len(assignments))
    
    sparsePrecisonM = numpy.asmatrix(numpy.copy(precisionM))
    
    for i in xrange(0, len(assignments), 1):
        for j in xrange(i + 1, len(assignments), 1):
            if assignments[i] != assignments[j]:
                sparsePrecisonM[i,j] = 0
                sparsePrecisonM[j,i] = 0
            
    return sparsePrecisonM

# def thresholdMatrix(A, noiseThreshold):
#    thresholdIndices = numpy.abs(A) <  noiseThreshold
#    A[thresholdIndices] = 0.0
#    return 
 
def getThresholdOutOfBlockMatrix(assignments, A, threshold):
    assert(A.shape[0] == A.shape[1])
    assert(A.shape[0] == len(assignments))
    
    sparseA = numpy.asmatrix(numpy.copy(A))
    
    for i in xrange(0, len(assignments), 1):
        for j in xrange(i + 1, len(assignments), 1):
            if assignments[i] != assignments[j] and numpy.abs(sparseA[i,j]) < threshold:
                sparseA[i,j] = 0
                sparseA[j,i] = 0
            
    return sparseA



# not optimized version:
# def getUltraFastDiagonalSolution(Q, eigVals, rho):
#       
#     n = eigVals.shape[0]
#     allNewEigVals = numpy.zeros(n)
#     for i in xrange(n):
#         newEigVal = (eigVals[i] + numpy.sqrt(eigVals[i] * eigVals[i] + 4 * rho)) / (2 * rho)
#         assert(newEigVal >= 0.0)
#         allNewEigVals[i] = newEigVal
#       
#     D = numpy.asmatrix(numpy.diag(allNewEigVals))
#     invCov = Q * D * Q.transpose()
#     return invCov


# optimized and tested
def getUltraFastDiagonalSolution(Q, eigVals, rho):
    
    allNewEigVals = (eigVals + numpy.sqrt(numpy.square(eigVals) + 4 * rho)) / (2 * rho)
    
    n = eigVals.shape[0]
    ones = numpy.asmatrix(numpy.ones(n))
    eigValsTimesQ = numpy.asmatrix(numpy.multiply(ones.transpose() * allNewEigVals, Q))
    
    invCov = Q * eigValsTimesQ.transpose()
    return invCov

    


def getFastDiagonalSolution(rightHandSideM, rho):
    
    eigVals, eigVecs = numpy.linalg.eigh(rightHandSideM)
    
    # print "eigVals = "
    # print eigVals[n-5:n]
    # assert(False)
    
    # print "first m eigenvectors = "
    # lowestEigVecs = eigVecs[:, 0:m]
    
    Q = numpy.asmatrix(eigVecs)
    return getUltraFastDiagonalSolution(Q, eigVals, rho)


def getFastDiagonalSolutionMoreStable(rightHandSideM, rho):
    assert(rho >= 0.0)
    if rho == 0.0:
        return - numpy.linalg.inv(rightHandSideM)
    
    try:
        eigVals, eigVecs = numpy.linalg.eigh(rightHandSideM)
    
    except numpy.linalg.linalg.LinAlgError:
        print "---- TRY SCIPY LINALG --- "
        eigVals, eigVecs = scipy.linalg.eigh(rightHandSideM)
        
    # print "eigVals = "
    # print eigVals[n-5:n]
    # assert(False)
    # print "first m eigenvectors = "
    # lowestEigVecs = eigVecs[:, 0:m]
    
    Q = numpy.asmatrix(eigVecs)
    return getUltraFastDiagonalSolution(Q, eigVals, rho)



def getNonZeroPatternArray(X):
    p = X.shape[0]
    zeroPattern = numpy.abs(numpy.sign(X))
    return numpy.reshape(zeroPattern, p * p)


def evalSparisityPattern(trueX, estimatedX):
    p = estimatedX.shape[0]
    
    estimatedTresholdedX = numpy.copy(estimatedX)
    estimatedTresholdedX[numpy.abs(estimatedX) < 0.001] = 0
    
    nonZeroPatternTrue = getNonZeroPatternArray(trueX)
    nonZeroPatternEstimated = getNonZeroPatternArray(estimatedTresholdedX)
    
    print "sparisty f1-score = ", sklearn.metrics.f1_score(nonZeroPatternTrue, nonZeroPatternEstimated)
    
    print "number of non-zeros (true) = ", numpy.sum(nonZeroPatternTrue)
    print "number of non-zeros (estimated) = ", numpy.sum(nonZeroPatternEstimated)
    
    print "trueX = "
    showMatrix(trueX)
    print "estimatedTresholdedX = "
    showMatrix(estimatedTresholdedX)
     
    return 
    
    # p = trueX.shape[0]
    # for i in xrange(p):
    #     for j in xrange(i+1, p, 1):


def test():
    # M = numpy.asarray([[2.0, 0.8, 0.6, 0.3], [0.8, 1.5, 0.5, 0.2], [0.6, 0.5, 1.2, 0.9], [0.3, 0.2, 0.9, 3.0]])
    # M = numpy.asmatrix(M)
    
    trueX = numpy.asarray([[2.0, 0.8, 0.6, 0], [0.8, 1.5, -0.5, 0.2], [0.6, 0.5, 1.2, -0], [0.3, 0.2, 0.9, 3.0]])
    estimatedX = numpy.asarray([[2.0, 0, 0.6, 0.0005], [0.8, 1.5, -0.5, 0.2], [0.6, 0.5, 1.2, -0], [0.3, 0.2, 0.9, 3.0]])
    
    evalSparisityPattern(trueX, estimatedX)
    
    
    # n = M.shape[0]
    # onesVec = numpy.asmatrix(numpy.ones(n)).transpose()
    # diagVec = numpy.reshape(numpy.asarray(M * onesVec), n)
    # L = getLaplacianNormalization(M, diagVec)
    # print L
    
# test()