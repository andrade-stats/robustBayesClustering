import numpy
import idcHelper
import marginalHelper
import KLapproximation
import marginalLikelihood_approx
import scipy.optimize

# checked
def getObjValue(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, dataSampleCov, allClusterSizes, noisePrecisionMode, allClusterPrecisionModes, beta, df):
    assert(type(dataSampleCov) == numpy.matrixlib.defmatrix.matrix)
    assert(type(noisePrecisionMode) == numpy.matrixlib.defmatrix.matrix)
    
    p = dataSampleCov.shape[0]
    priorNuNoise = df + p + 1
    priorSigmaNoise = numpy.asmatrix(PRIOR_SCALE_FOR_NOISE * numpy.eye(p))
    
    k = len(allClusterSizes)
    assert(k >= 1)
    
    X = idcHelper.createFullX(p, allClusterPrecisionModes)
    
    # idcHelper.assertValidAndNotInfinite(idcHelper.getLogDetWithNegInf(X))
    # idcHelper.assertValidAndNotInfinite(idcHelper.getLogDetWithNegInf(noisePrecisionMode))
    # print "idcHelper.getLogDetWithNegInf(X) = ", idcHelper.getLogDetWithNegInf(X)
    # print "idcHelper.getLogDetWithNegInf(noisePrecisionMode) = ", idcHelper.getLogDetWithNegInf(noisePrecisionMode)
    # print "idcHelper.getLogDetWithNegInf(X + noisePrecisionMode) = ", idcHelper.getLogDetWithNegInf(X + noisePrecisionMode)
    # print "ANALYSIS OF OBJECTIVE FUNCTION:"
    # print "first + second part = ", - n * idcHelper.getLogDetWithNegInf(X + noisePrecisionMode) + n * idcHelper.matrixInnerProd(X + noisePrecisionMode, dataSampleCov)
    
    objValue = - n * idcHelper.getLogDetWithNegInf(X + beta * noisePrecisionMode) + n * idcHelper.matrixInnerProd(X + beta * noisePrecisionMode, dataSampleCov)
    # idcHelper.assertValidAndNotInfinite(objValue)
    
    objValue += - (priorNuNoise + p + 1) * idcHelper.getLogDetWithNegInf(noisePrecisionMode) + idcHelper.matrixInnerProd(noisePrecisionMode, priorSigmaNoise)
    # idcHelper.assertValidAndNotInfinite(objValue)
    
    for j in xrange(k):
        assert(type(allClusterPrecisionModes[j]) == numpy.matrixlib.defmatrix.matrix)
        clusterSize = allClusterSizes[j]
        priorNuCluster = df + clusterSize + 1
        priorSigmaCluster = numpy.asmatrix(PRIOR_SCALE_FOR_CLUSTER * numpy.eye(clusterSize))
        objValue += - (priorNuCluster + clusterSize + 1) * idcHelper.getLogDetWithNegInf(allClusterPrecisionModes[j]) + idcHelper.matrixInnerProd(allClusterPrecisionModes[j], priorSigmaCluster)
        # idcHelper.assertValidAndNotInfinite(objValue)
        
    return objValue



# use the normal Bayesian estimate as an initial guess
def getInitialPrecisionMatrix(n, aFactor, S, priorSigma):
    return (n + aFactor) * numpy.linalg.inv(n * S + priorSigma)


# uses the 3-Block ADMM as proposed by Takeda-Sensei (see pdf attached to her email)
def findPosteriorMode3Block(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, dataSampleCov, allClusterSizes, beta, df):
    assert(type(dataSampleCov) == numpy.matrixlib.defmatrix.matrix)
    
    clusterAssignments = marginalHelper.getSimpleClusterAssignments(allClusterSizes)
    
    p = dataSampleCov.shape[0]
    priorNuNoise = df + p + 1
    aFacNoise = priorNuNoise + p + 1
    priorSigmaNoise = PRIOR_SCALE_FOR_NOISE * numpy.eye(p)
    
    # nS = n * dataSampleCov
    
    assert(PRIOR_SCALE_FOR_NOISE >= 0.01)
    assert(PRIOR_SCALE_FOR_CLUSTER >= 0.01)
    
    k = len(allClusterSizes)
    assert(k >= 1)
    
    # initialization
    allSblocks = idcHelper.getBlockCovariance(dataSampleCov, clusterAssignments)
    U = numpy.zeros((p,p))
    Z = numpy.zeros((p,p))
    allX = []
    allAFactors = []
    allPriorSigmas = []
    for j in xrange(k):
        clusterSize = allClusterSizes[j]
        priorNuCluster = df + clusterSize + 1
        priorSigmaCluster = PRIOR_SCALE_FOR_CLUSTER * numpy.eye(clusterSize)
        aFactor = priorNuCluster + clusterSize + 1
        allAFactors.append(aFactor)
        allPriorSigmas.append(priorSigmaCluster)
        allX.append(getInitialPrecisionMatrix(n, aFactor, allSblocks[j], priorSigmaCluster)) # initial X
    
    
    Xnoise = getInitialPrecisionMatrix(n, aFacNoise, dataSampleCov, priorSigmaNoise)
    
    MAX_ADMM_ITERATIONS = 100000
    rho = 1.0
    
    previousObjValue = float("inf")
    
       
    for admmIt in xrange(MAX_ADMM_ITERATIONS):
        
        allUs = idcHelper.getBlockCovariance(U, clusterAssignments)
        allZs = idcHelper.getBlockCovariance(Z, clusterAssignments)
        
        # perform X update
        allXnoise = idcHelper.getBlockCovariance(Xnoise, clusterAssignments)
        for j in xrange(k):
            rightHandSideM = (-1.0 / allAFactors[j]) * (allPriorSigmas[j] + rho * (beta * allXnoise[j] - allZs[j]) + allUs[j])
            allX[j] = idcHelper.getFastDiagonalSolutionMoreStable(rightHandSideM, rho / allAFactors[j])
        
        # perform X_epsilon update
        X = idcHelper.createFullX(p, allX)
        rightHandSideM = (-1.0 / aFacNoise) * (priorSigmaNoise + beta * rho * (X - Z) + beta * U)
        Xnoise = idcHelper.getFastDiagonalSolutionMoreStable(rightHandSideM, (rho * beta * beta) / aFacNoise)
        
        # perform Z update
        rightHandSideM = (1.0 / float(n)) * (U + rho * (X + beta * Xnoise)) - dataSampleCov
        Z = idcHelper.getFastDiagonalSolutionMoreStable(rightHandSideM, rho / float(n))
        
        # perform U update
        U = U + rho * (X + beta * Xnoise - Z)
        
        objValue = getObjValue(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, dataSampleCov, allClusterSizes, Xnoise, allX, beta, df)
        
        if admmIt % 100 == 0:
            print(str(admmIt) + ", objValue = " + str(objValue))
            rho = rho * 1.1
            # print "rho = ", rho
            
        # idcHelper.assertValidAndNotInfinite(objValue)
        
        VERY_ACCURATE = 0.000001
        
        if numpy.isfinite(objValue) and numpy.abs(objValue - previousObjValue) < VERY_ACCURATE:
            print "REACHED CONVERGENCE"
            break
            
        
        previousObjValue = objValue
        
    idcHelper.assertValidAndNotInfinite(objValue)
    
    return Xnoise, allX, objValue


def optimizePosteriorNusNew(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, sortedClustering, noisePrecisionMode, allClusterPrecisionModes, beta, df):
    
    p = sortedSampleCovariance.shape[0]
    allBlockSampleCovs = idcHelper.getBlockCovariance(sortedSampleCovariance, sortedClustering)
    allJointTraces, allClusterTraces, noiseTrace = KLapproximation.precalcuateRelevantTraces(sortedClustering, PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allBlockSampleCovs, noisePrecisionMode, allClusterPrecisionModes, beta)
    
    def funcPosteriorNuNoiseScalar(posteriorNu_local):
        priorNuNoise = float(df + p + 1)
        return KLapproximation.oneSigmaPart(priorNuNoise, n, p, noiseTrace, posteriorNu_local, 0)
    
    SEARCH_INTERVAL = (float(p + 1), float(100 * (p + n + 1)))
    result = scipy.optimize.minimize_scalar(funcPosteriorNuNoiseScalar, bounds = SEARCH_INTERVAL, method = 'bounded')
    posteriorNuNoise = result.x
    
    totalKL = funcPosteriorNuNoiseScalar(posteriorNuNoise)
    
    allPosteriorNuClusters = numpy.zeros(len(allClusterSizes))
        
    for j in xrange(len(allClusterSizes)):
        clusterSize = allClusterSizes[j]
        assert(clusterSize == allClusterPrecisionModes[j].shape[0])
        priorNuCluster = float(df + clusterSize + 1)
        
        def funcPosteriorNuClusterScalar(posteriorNu_local):
            returnValue = KLapproximation.oneSigmaPart(priorNuCluster, n, clusterSize, allClusterTraces[j], posteriorNu_local, n)
            return returnValue
        
        SEARCH_INTERVAL = (float(clusterSize + 1), float(100.0 * (clusterSize + n + 1)))
        result = scipy.optimize.minimize_scalar(funcPosteriorNuClusterScalar, bounds = SEARCH_INTERVAL, method = 'bounded')
        allPosteriorNuClusters[j] = result.x
    
        totalKL += funcPosteriorNuClusterScalar(allPosteriorNuClusters[j])
        
    infoStr = str(posteriorNuNoise) + "," + str(allPosteriorNuClusters)
    print infoStr + " klDivApprox = " + str(totalKL)
    return posteriorNuNoise, allPosteriorNuClusters


def getLogMarginalPrincipledNew(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, originalSampleCovariance, clustering, SOLVER, beta, df):
    assert(n >= 10)
    
    sortedSampleCovariance, allClusterSizes, sortedClustering = marginalLikelihood_approx.getPermutedCovarianceMatrixAndClusterSizes(originalSampleCovariance, clustering)
    # print "sortedClustering = ", sortedClustering
    # print "allClusterSizes = ", allClusterSizes
    # print "sortedSampleCovariance = "
    # idcHelper.showMatrix(sortedSampleCovariance)
    
    if SOLVER == "CVXPY":
        assert(False)
        # noisePrecisionMode, allClusterPrecisionModes, objValue = findPosteriorModeWithCVXPY(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, beta, df)
    elif SOLVER == "ADMM-3BLOCK":
        noisePrecisionMode, allClusterPrecisionModes, objValue = findPosteriorMode3Block(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, beta, df)
    else:
        assert(False)
    
    print "final objValue = ", objValue
    
    bestPosteriorNuNoise, allBestPosteriorNuClusters = optimizePosteriorNusNew(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, sortedClustering, noisePrecisionMode, allClusterPrecisionModes, beta, df)
    
    # bestPosteriorNuNoise = float(n)
    # allBestPosteriorNuClusters = numpy.ones(len(allClusterSizes)) * float(n)
    
    jointLogProb = marginalHelper.getJointLogProb(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, noisePrecisionMode, allClusterPrecisionModes, beta, df)
    posteriorLogProb = marginalHelper.getLogPosteriorApproximation(noisePrecisionMode, allClusterPrecisionModes, bestPosteriorNuNoise, allBestPosteriorNuClusters, noisePrecisionMode, allClusterPrecisionModes)
    logMarginalApproximation = jointLogProb - posteriorLogProb
    return logMarginalApproximation, noisePrecisionMode, allClusterPrecisionModes


def miniTest():
    RANDOM_GENERATOR_SEED = 9899832
    numpy.random.seed(RANDOM_GENERATOR_SEED)
    
    
    n = 10000 # 2000
    # allTrueClusterSizes = [4,6]
    allTrueClusterSizes = [3,3]
    p = sum(allTrueClusterSizes)
    USE_NOISE_IN_TRUE_DATA = "InverseWishart"
    hiddenDataIds, sampleCovariance = marginalHelper.createExampleData(p, allTrueClusterSizes, n, USE_NOISE_IN_TRUE_DATA)
    

    # SOLVER = "CVXPY"
    SOLVER = "ADMM-3BLOCK"
    PRIOR_SCALE_FOR_CLUSTER = 1.0
    PRIOR_SCALE_FOR_NOISE = 1.0
    beta = 0.02
    df = 1.0
    clusterAssignments = hiddenDataIds
    # clusterAssignments = numpy.ones(len(hiddenDataIds), dtype = numpy.int)
    
    logProb = getLogMarginalPrincipledNew(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampleCovariance, clusterAssignments, SOLVER, beta, df)
    
    
    
    print "clusterAssignments = ", clusterAssignments
    print "logProb = ", logProb
    
    print "normal log prob = ", ( - marginalHelper.getNegLogMarginalLikelihoodNew(sampleCovariance, n, clusterAssignments, df))
    
# miniTest()

