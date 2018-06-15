import numpy
import scipy.stats
import idcHelper
import scipy.special


# checked
def getDiagammaSummation(nu, p):
    assert(nu > p)
    
    allDiagammaValues = numpy.zeros(p)
    
    for i in xrange(p):
        allDiagammaValues[i] = scipy.special.digamma(0.5 * (nu - p + (i + 1)))
    
    return numpy.sum(allDiagammaValues)
    
    
def getExpLogDetPrecisionOnly(nu, Sigma, logFac):
    
    p = Sigma.shape[0]
    sign, detSigmaHalf = numpy.linalg.slogdet(0.5 * Sigma)
    assert(sign > 0)
    assert(nu >= p)
    
    logPart = logFac * (getDiagammaSummation(nu, p) - detSigmaHalf)
    return logPart

# checked
# return E[logFac * ln|Sigma^{-1}| - trace(traceMat * Sigma^{-1})], where Sigma is distributed according to inverse wishart with nu, sigma
def getExpLogDetPrecisionMinusTrace(nu, Sigma, invSigma, logFac, traceMat):
    assert(type(invSigma) == numpy.matrixlib.defmatrix.matrix)
    assert(type(traceMat) == numpy.matrixlib.defmatrix.matrix)
    
    logPart = getExpLogDetPrecisionOnly(nu, Sigma, logFac)
    
    # print "nu = ", nu
    # print "numpy.trace(traceMat * invSigma) = ", numpy.trace(traceMat * invSigma)
    tracePart = nu * numpy.trace(traceMat * invSigma)
    return logPart - tracePart

# checked
# returns Entropy[V] = - E[log(V)], where V is distributed according to inverse Wishart
def getEntropy(nu, Sigma):   
    p = Sigma.shape[0]
    sign, logDetSigmaHalf = numpy.linalg.slogdet(0.5 * Sigma)
    assert(sign > 0)
    assert(nu >= p)
    
    firstPart = 0.5 * (p + 1) * logDetSigmaHalf
    secondPart = 0.5 * (p * nu) 
    thirdPart = scipy.special.multigammaln(nu * 0.5, p)
    forthPart = 0.5 * (nu + p + 1) * getDiagammaSummation(nu, p)
    
    entropy = firstPart + secondPart + thirdPart - forthPart
    print "entropy = ", entropy
    return entropy

def getApproxMode(PRIOR_SCALE_FOR_NOISE, PRIOR_SCALE_FOR_CLUSTER, df, beta, n, sortedSampleCovariance, allClusterSizes):
    numberOfClusters = len(allClusterSizes)
    p = sortedSampleCovariance.shape[0]
    samplingDF = beta * n + p + 1 + df
    noisePrecisionMode = numpy.linalg.inv( (beta * n * sortedSampleCovariance + PRIOR_SCALE_FOR_NOISE * numpy.eye(p)) / (samplingDF + p + 1))
    
    allClusterPrecisionModes = []
    nextClusterStartsAt = 0
    for clusterId in xrange(numberOfClusters):
        startId = nextClusterStartsAt
        endId = nextClusterStartsAt + allClusterSizes[clusterId]
        thisClusterCov = (1.0 - beta) * n * sortedSampleCovariance[startId:endId, startId:endId] + PRIOR_SCALE_FOR_CLUSTER * numpy.eye(allClusterSizes[clusterId])
        samplingDF = (1.0 - beta) * n + allClusterSizes[clusterId] + 1 + df
        allClusterPrecisionModes.append(numpy.linalg.inv(thisClusterCov / (samplingDF + allClusterSizes[clusterId] + 1)))
        nextClusterStartsAt += allClusterSizes[clusterId]

    return noisePrecisionMode, allClusterPrecisionModes

# for example for allClusterSizes = [3, 4] it returns the cluster assignments [1,1,1, 2,2,2,2]
def getSimpleClusterAssignments(allClusterSizes):
    clusterAssignments = numpy.zeros(numpy.sum(allClusterSizes), dtype = numpy.int)
    clusterId = 1
    currentStartId = 0
    for clusterSize in allClusterSizes:
        clusterAssignments[currentStartId:(currentStartId + clusterSize)] = clusterId
        clusterId += 1
        currentStartId += clusterSize
    return clusterAssignments

# checked
def createExampleData(p, allClusterSizes, n, USE_NOISE): 
    
    inClusterScale = 1.0
    noiseScale = 100.0
    
    fullPrecisionMatrix = numpy.zeros((p,p))
    hiddenDataIds = numpy.zeros(p, dtype = numpy.int_)
    
    assert(numpy.sum(allClusterSizes) == p)
    
    nextClusterStartsAt = 0
    currentClusterId = 0
        
    for i in xrange(p):
        if i == nextClusterStartsAt:
            nrVariablesInCluster = allClusterSizes[currentClusterId]
            
            nu0 = nrVariablesInCluster + 1
            Sigma0 = inClusterScale * numpy.eye(nrVariablesInCluster)
            clusterCovMatrix = scipy.stats.invwishart.rvs(df = nu0, scale = Sigma0, size=1)
    
            startId = nextClusterStartsAt
            endId = nextClusterStartsAt + nrVariablesInCluster
            if nrVariablesInCluster > 1:
                fullPrecisionMatrix[startId:endId, startId:endId] = numpy.linalg.inv(clusterCovMatrix)
            else:
                fullPrecisionMatrix[startId:endId, startId:endId] = 1.0 / clusterCovMatrix
                
            nextClusterStartsAt += nrVariablesInCluster
            currentClusterId += 1
        hiddenDataIds[i] = currentClusterId
    
    
    if USE_NOISE == "InverseWishart":
        nu0 = p + 1
        Sigma0 = noiseScale * numpy.eye(p)
        noiseCovMatrix = scipy.stats.invwishart.rvs(df = nu0, scale = Sigma0, size=1)
        fullCovMatrix = numpy.linalg.inv(fullPrecisionMatrix + numpy.linalg.inv(noiseCovMatrix))
    elif USE_NOISE == "Uniform":
        assert(False)
        # uniformNoiseStdOnPrec = 0.1
        # fullCovMatrix = simulation.addUniformNoiseToPrec(numpy.linalg.inv(fullPrecisionMatrix), uniformNoiseStdOnPrec)
    elif USE_NOISE == "NoiseOnCov":
        nu0 = p + 1
        Sigma0 = noiseScale * numpy.eye(p)
        noiseCovMatrix = scipy.stats.invwishart.rvs(df = nu0, scale = Sigma0, size=1)
        fullCovMatrix = numpy.linalg.inv(fullPrecisionMatrix) + noiseCovMatrix
    else:
        assert(USE_NOISE == "noNoise")
        fullCovMatrix = numpy.linalg.inv(fullPrecisionMatrix)
    
    
    print "true precision matrix (without noise) = "
    idcHelper.showMatrix(fullPrecisionMatrix)
    
    modelMeansAppended = numpy.zeros(p)
    allSamples = numpy.random.multivariate_normal(mean = modelMeansAppended, cov = fullCovMatrix, size = n)
    
    sampleCovariance = numpy.asmatrix(numpy.cov(allSamples.transpose(), rowvar=True, bias=True))
    
    print "true precision matrix (with noise) = "
    idcHelper.showMatrix(numpy.linalg.inv(fullCovMatrix))
    print "sample precision matrix = "
    idcHelper.showMatrix(numpy.linalg.inv(sampleCovariance))
    
    return hiddenDataIds, sampleCovariance


# checked
# def getNegLogMarginalLikelihood(S, n, clusterAssignments, priorScale):
#     p = S.shape[0]
#      
#     allBlockCovs = idcHelper.getBlockCovariance(S, clusterAssignments)
#      
#     totalLogMarginal = -0.5 * n * p * numpy.log(2.0 * numpy.pi)
#     for blockCov in allBlockCovs:
#         totalLogMarginal += idcHelper.getLogMarginalOfOneBlock(blockCov, n, priorScale)
#      
#     return -1.0 * totalLogMarginal


def getNormalizationConstantWishart(nu0, Sigma0_INVERSE):
    p = Sigma0_INVERSE.shape[0]
    sign, detSigma0 = numpy.linalg.slogdet(Sigma0_INVERSE)
    assert(sign > 0)
    assert(nu0 >= p)
     
    firstFac = 0.5 * nu0 * detSigma0
    secondFac = 0.5 * (p * nu0) * numpy.log(2)
    thirdFac = scipy.special.multigammaln(nu0 * 0.5, p)
    
    # since we had the inverse(!) of Sigma0, we add "- firstFac" instead of "firstFac" !!
    return - firstFac + secondFac + thirdFac


def getNormalizationConstantInvWishart(nu0, Sigma0):
    return getNormalizationConstantWishart(nu0, Sigma0)


# assumes a wishart prior over the precision matrix with df = p + 2, and Scale0 = I
# is equivalent to a G-Wishart Distribution with delta = 3 and Scale0 = I
def getLogMarginalOfWishartOneBlock(sampleCovBlock, n, df):
    p = sampleCovBlock.shape[0]
    priorPart = getNormalizationConstantWishart(nu0 = p + 1 + df, Sigma0_INVERSE = numpy.eye(p))
    posteriorPart = getNormalizationConstantWishart(nu0 = p + 1 + df + n, Sigma0_INVERSE = numpy.eye(p) + n * sampleCovBlock)
    return  posteriorPart - priorPart


def getPosteriorPredictiveProb_invWishartModel(sampleCovBlockTest, nTest, sampleCovBlockTrain, nTrain, df):
    p = sampleCovBlockTrain.shape[0]
    posteriorProbNormalization = getNormalizationConstantInvWishart(p + 1 + df + nTrain, numpy.eye(p) + nTrain * sampleCovBlockTrain)
    predictionProbNormalizaion = getNormalizationConstantInvWishart(p + 1 + df + nTrain + nTest, numpy.eye(p) + nTrain * sampleCovBlockTrain + nTest * sampleCovBlockTest)
    gaussRemainer = - nTest * p * 0.5 * numpy.log(2 * numpy.pi)
    return  gaussRemainer + predictionProbNormalizaion - posteriorProbNormalization


# assumes a wishart prior over the precision matrix with df = p + 2, and Scale0 = I
# is equivalent to a G-Wishart Distribution with delta = 3 and Scale0 = I
def getNegLogMarginalLikelihoodNew(S, n, clusterAssignments, df):
    p = S.shape[0]
    
    allBlockCovs = idcHelper.getBlockCovariance(S, clusterAssignments)
    
    totalLogMarginal = -0.5 * n * p * numpy.log(2.0 * numpy.pi) # needs to be updated
    for blockCov in allBlockCovs:
        totalLogMarginal += getLogMarginalOfWishartOneBlock(blockCov, n, df)
    
    return -1.0 * totalLogMarginal



# def getNegLogMarginalLikelihoodWithThresholding(S, n, clusterAssignments, priorScale, uniformNoiseStdOnPrec):
#     precisionThreshold = numpy.sqrt(3.0) * uniformNoiseStdOnPrec
#     thresholdedS = thresholding.getSampleCovarianceWithTresholdedPrecisionMatrix(S, precisionThreshold)
#     return getNegLogMarginalLikelihood(thresholdedS, n, clusterAssignments, priorScale)











# checked
def getObjValue(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, dataSampleCov, allClusterSizes, noisePrecisionMode, allClusterPrecisionModes):
    assert(type(dataSampleCov) == numpy.matrixlib.defmatrix.matrix)
    assert(type(noisePrecisionMode) == numpy.matrixlib.defmatrix.matrix)
    
    p = dataSampleCov.shape[0]
    priorNuNoise = p + 1
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
    
    objValue = - n * idcHelper.getLogDetWithNegInf(X + noisePrecisionMode) + n * idcHelper.matrixInnerProd(X + noisePrecisionMode, dataSampleCov)
    # idcHelper.assertValidAndNotInfinite(objValue)
    objValue += - (priorNuNoise + p + 1) * idcHelper.getLogDetWithNegInf(noisePrecisionMode) + idcHelper.matrixInnerProd(noisePrecisionMode, priorSigmaNoise)
    # idcHelper.assertValidAndNotInfinite(objValue)
    
    for j in xrange(k):
        assert(type(allClusterPrecisionModes[j]) == numpy.matrixlib.defmatrix.matrix)
        clusterSize = allClusterSizes[j]
        priorNuCluster = clusterSize + 1
        priorSigmaCluster = numpy.asmatrix(PRIOR_SCALE_FOR_CLUSTER * numpy.eye(clusterSize))
        objValue += - (priorNuCluster + clusterSize + 1) * idcHelper.getLogDetWithNegInf(allClusterPrecisionModes[j]) + idcHelper.matrixInnerProd(allClusterPrecisionModes[j], priorSigmaCluster)
        # idcHelper.assertValidAndNotInfinite(objValue)
        
    return objValue





# returns (positive!) log likelihood
def getNormalLogLikelihood(n, dataSampleCov, invModelCov):
    assert(type(dataSampleCov) == numpy.matrixlib.defmatrix.matrix)
    assert(type(invModelCov) == numpy.matrixlib.defmatrix.matrix)
    
    p = invModelCov.shape[0]
    assert(p == dataSampleCov.shape[0])
    
    unnormalizedLogLikelihood = idcHelper.getLogDet(invModelCov) - numpy.trace(dataSampleCov * invModelCov)
    return (float(n) / 2.0) *  (unnormalizedLogLikelihood - p * numpy.log(2.0 * numpy.pi))


# returns the log liklihood of the proposed model
def getLogLikelihoodProposed(n, dataSampleCov, noisePrecision, allClusterPrecisions, beta):
    p = dataSampleCov.shape[0]
    
    # Cinv = C^{-1}
    Cinv = numpy.zeros((p,p))
    
    nextClusterStartsAt = 0
    
    for clusterPrecision in allClusterPrecisions:
        nrVariablesInCluster = clusterPrecision.shape[0]
        startId = nextClusterStartsAt
        endId = nextClusterStartsAt + nrVariablesInCluster
        Cinv[startId:endId, startId:endId] = clusterPrecision
        nextClusterStartsAt += nrVariablesInCluster
    
    
    Cinv += beta * noisePrecision
    Cinv = numpy.asmatrix(Cinv)
    
    # print "Cinv = "
    # idcHelper.showMatrix(Cinv)
    
    logLikelihood = getNormalLogLikelihood(n, dataSampleCov, Cinv)
    return logLikelihood


def isValidFullCovarianceMatrix(noisePrecision, allClusterPrecisions, beta):
    p = noisePrecision.shape[0]
    
    # Cinv = C^{-1}
    Cinv = numpy.zeros((p,p))
    
    nextClusterStartsAt = 0
    
    for clusterPrecision in allClusterPrecisions:
        nrVariablesInCluster = clusterPrecision.shape[0]
        startId = nextClusterStartsAt
        endId = nextClusterStartsAt + nrVariablesInCluster
        Cinv[startId:endId, startId:endId] = clusterPrecision
        nextClusterStartsAt += nrVariablesInCluster
        
    Cinv += beta * noisePrecision
    Cinv = numpy.asmatrix(Cinv)
    
    
    if (len(Cinv.shape) == 2):
        sign, logdet = numpy.linalg.slogdet(Cinv)
        return (sign > 0)
    else:
        # covMatrix is single number
        return (Cinv > 0)
        

# checked
def getJointLogProb(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, dataSampleCov, noisePrecision, allClusterPrecisions, beta, df):
    p = dataSampleCov.shape[0]
    
    logLikelihood = getLogLikelihoodProposed(n, dataSampleCov, noisePrecision, allClusterPrecisions, beta)
    
    priorNu = df + p + 1
    priorSigma = PRIOR_SCALE_FOR_NOISE * numpy.eye(p)
    logPrior = idcHelper.getInverseWishartLogPDF(priorNu, priorSigma, numpy.linalg.inv(noisePrecision))

        
    for clusterPrecision in allClusterPrecisions:
        clusterSize = clusterPrecision.shape[0]
        priorSigma = PRIOR_SCALE_FOR_CLUSTER * numpy.eye(clusterSize)
        priorNu = df + clusterSize + 1
        logPrior += idcHelper.getInverseWishartLogPDF(priorNu, priorSigma, numpy.linalg.inv(clusterPrecision))
        
    return logLikelihood + logPrior




# checked
# calculates log g(noisePrecisionSample, allClusterPrecisionSamples | noisePrecisionParam, allClusterPrecisionParams, nuNoiseParam, nuClusterParam)
def getLogPosteriorApproximation(noisePrecisionParam, allClusterPrecisionParams, nuNoiseParam, nuClusterParams, noisePrecisionSample, allClusterPrecisionSamples):
    assert(len(allClusterPrecisionParams) >= 1 and len(allClusterPrecisionParams) == len(allClusterPrecisionSamples))
   
    # from noise model
    p = noisePrecisionParam.shape[0]
    assert(nuNoiseParam >= p + 1)
    noiseCovarianceParam = numpy.linalg.inv(noisePrecisionParam)
    noiseCovarianceSample = numpy.linalg.inv(noisePrecisionSample)
    logProb = idcHelper.getInverseWishartLogPDF(nuNoiseParam, (nuNoiseParam + p + 1) * noiseCovarianceParam, noiseCovarianceSample)
        
    # from cluster model
    for clusterId in xrange(len(allClusterPrecisionParams)):
        clusterPrecisionParam = allClusterPrecisionParams[clusterId]
        clusterPrecisionSample = allClusterPrecisionSamples[clusterId]
        clusterSize = clusterPrecisionParam.shape[0]
        clusterCovarianceParam = numpy.linalg.inv(clusterPrecisionParam)
        clusterCovarianceSample = numpy.linalg.inv(clusterPrecisionSample)
        assert(nuClusterParams[clusterId] >= clusterSize + 1)
        logProb += idcHelper.getInverseWishartLogPDF(nuClusterParams[clusterId], (nuClusterParams[clusterId] + clusterSize + 1) * clusterCovarianceParam, clusterCovarianceSample)
    
    return logProb


def setupR():
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    ro.r('source(\'imports.R\')')
    
    ro.r('''
    set.seed(9899832)
    ''')
    
    return ro


def symmetrizeMatrix(M):
    symmetrizedM = numpy.asarray(M)
    symmetrizedM = 0.5 * (numpy.transpose(symmetrizedM) + symmetrizedM)
    return symmetrizedM

# checked
def getInverseWishartSample(ro, nu0, Sigma0):
    
    ro.globalenv['nu0'] = nu0
    symmetrizedSigma0 = numpy.asarray(Sigma0)
    symmetrizedSigma0 = 0.5 * (numpy.transpose(symmetrizedSigma0) + symmetrizedSigma0)
    ro.globalenv['Sigma0'] =  symmetrizedSigma0
    ro.r('''
    sample <- rinvwishart(nu = nu0, S = Sigma0)
    ''')
    
    return numpy.asmatrix(ro.r('sample'))

def getLogPdfInverseWishartR(ro, nu0, Sigma0, covMat):
    ro.globalenv['nu0'] = nu0
    ro.globalenv['Sigma0'] = symmetrizeMatrix(Sigma0)
    ro.globalenv['covMat'] = symmetrizeMatrix(covMat)
#     ro.r('''
#     pdf <- diwish(covMat, nu0, Sigma0)
#     ''')
    # , TRUE)
    ro.r('''
    logpdf <- dinvwishart(Sigma = covMat, nu = nu0, S = Sigma0, log=TRUE) 
    ''')
    return ro.r('logpdf')[0]

def getWishartSample(nu0, Scale0):
    return scipy.stats.wishart.rvs(df=nu0, scale=Scale0, size = 1)
    
def getWishartLogPDF(nu0, Scale0, precMat):
    return scipy.stats.wishart.logpdf(x = precMat, df = nu0, scale = Scale0)
    
    
