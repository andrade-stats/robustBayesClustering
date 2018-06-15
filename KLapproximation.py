
import numpy
import marginalHelper
import scipy.special
import idcHelper

# checked
def newFastCalculationOfKLApproxForOneSigma(priorNu, n, p, tracePart, posteriorNu):
    
    firstLogPart = (n + priorNu) * p * numpy.log(posteriorNu + p + 1) # checked
    digammaPart = (posteriorNu - priorNu - n) * marginalHelper.getDiagammaSummation(posteriorNu, p) # checked
    fullTracePart = (posteriorNu / (posteriorNu + p + 1)) * tracePart # checked
    multiGammaPart = 2 * scipy.special.multigammaln(posteriorNu * 0.5, p) # checked
    linearPart = posteriorNu * p # checked
    
    return firstLogPart + digammaPart + fullTracePart - multiGammaPart - linearPart # checked

# checked
def getKLApproximationFastNoise(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampleCov, allBlockSampleCovs, noisePrecisionMode, allClusterPrecisionModes, USE_NOISE_IN_MODEL, posteriorNuNoise):
    
    if USE_NOISE_IN_MODEL:
        p = noisePrecisionMode.shape[0]
        priorNu = p + 1
        priorSigma = PRIOR_SCALE_FOR_NOISE * numpy.eye(p)
        traceMat = n * sampleCov + priorSigma
        
        assert(type(noisePrecisionMode) == numpy.matrixlib.defmatrix.matrix)
        assert(type(traceMat) == numpy.matrixlib.defmatrix.matrix)
        tracePart = numpy.trace(traceMat * noisePrecisionMode)
        upperBound = newFastCalculationOfKLApproxForOneSigma(priorNu, n, p, tracePart, posteriorNuNoise)
    else:
        upperBound = 0
    
    return upperBound

# checked
def getKLApproximationFastCluster(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampleCov, allBlockSampleCovs, noisePrecisionMode, allClusterPrecisionModes, USE_NOISE_IN_MODEL, posteriorNuCluster):
    
    upperBound = 0
    
    for j in xrange(len(allBlockSampleCovs)):
        clusterPrecisionMode = allClusterPrecisionModes[j]
        clusterSize = clusterPrecisionMode.shape[0]
        priorNu = clusterSize + 1
        priorSigma = PRIOR_SCALE_FOR_CLUSTER * numpy.eye(clusterSize)
        assert(clusterSize == clusterPrecisionMode.shape[0])
        traceMat = n * allBlockSampleCovs[j] + priorSigma
        
        assert(type(noisePrecisionMode) == numpy.matrixlib.defmatrix.matrix)
        assert(type(traceMat) == numpy.matrixlib.defmatrix.matrix)
        tracePart = numpy.trace(traceMat * clusterPrecisionMode)
        upperBound += newFastCalculationOfKLApproxForOneSigma(priorNu, n, clusterSize, tracePart, posteriorNuCluster)
        
    return upperBound


# reading checked
def precalcuateRelevantTraces(sortedClustering, PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampleCov, allBlockSampleCovs, noisePrecisionMode, allClusterPrecisionModes, beta):
    
    allNoisePrecisionModeBlocks = idcHelper.getBlockCovariance(noisePrecisionMode, sortedClustering)
    assert(len(allNoisePrecisionModeBlocks) == len(allClusterPrecisionModes))
    
    allJointTraces = []
    
    for j in xrange(len(allClusterPrecisionModes)):
        covModeBlock = numpy.linalg.inv(allClusterPrecisionModes[j])
        jointTrace = idcHelper.matrixInnerProdSymmetric(covModeBlock,allNoisePrecisionModeBlocks[j])
        allJointTraces.append(jointTrace)
    
    allClusterTraces = []
    
    for j in xrange(len(allClusterPrecisionModes)):
        clusterPrecisionMode = allClusterPrecisionModes[j]
        clusterSize = clusterPrecisionMode.shape[0]
        priorSigma = PRIOR_SCALE_FOR_CLUSTER * numpy.eye(clusterSize)
        assert(clusterSize == clusterPrecisionMode.shape[0])
        traceMat = n * allBlockSampleCovs[j] + priorSigma
        clusterTrace = idcHelper.matrixInnerProdSymmetric(traceMat, clusterPrecisionMode)
        allClusterTraces.append(clusterTrace)
    
    p = noisePrecisionMode.shape[0]
    priorSigma = PRIOR_SCALE_FOR_NOISE * numpy.eye(p)
    traceMatNoise = beta * n * sampleCov + priorSigma
    noiseTrace = idcHelper.matrixInnerProdSymmetric(traceMatNoise, noisePrecisionMode)
    
    return allJointTraces, allClusterTraces, noiseTrace

    

# reading checked
# corresponds to sum of second and third line
# if used for cluster then set addtionalN to n, otherwise to 0
def oneSigmaPart(priorNu, n, p, tracePart, posteriorNu, additionalN):
    assert(additionalN == 0 or additionalN == n)
    assert(isinstance(posteriorNu, float))
    assert(isinstance(tracePart, float))
    assert(isinstance(priorNu, float))
    assert(isinstance(n, int))
    assert(isinstance(p, int))
    
    fullTracePart = (posteriorNu / (posteriorNu + p + 1.0)) * tracePart  # checked
    multiGammaPart = 2.0 * scipy.special.multigammaln(posteriorNu * 0.5, p) # checked
    linearPart = posteriorNu * p  # checked
    logPart =  p * (priorNu + additionalN) * numpy.log(posteriorNu + p + 1.0) # checked
    digammaPart = (posteriorNu - priorNu - additionalN) * marginalHelper.getDiagammaSummation(posteriorNu, p) # checked
    
    return fullTracePart - multiGammaPart - linearPart + logPart + digammaPart # checked


# reading checked
def noiseClusterInteractionPart(posteriorNuCluster, posteriorNuNoise, p, pCluster, n, jointTracePart):
    assert(isinstance(posteriorNuCluster, float))
    assert(isinstance(posteriorNuNoise, float))
    assert(isinstance(jointTracePart, float))
    assert(isinstance(n, int))
    assert(isinstance(p, int))
    
    posteriorNoiseFac = posteriorNuNoise / (posteriorNuNoise + p + 1.0)
    posteriorClusterFac = (posteriorNuCluster + pCluster + 1) / (posteriorNuCluster - pCluster - 1.0)
    
    return -1.0 * n * posteriorNoiseFac * posteriorClusterFac * jointTracePart




# reading checked
def evalKLApproximation(allClusterSizes, allClusterPrecisionModes, allJointTraces, allClusterTraces, noiseTrace, n, p, posteriorNuNoise, allPosteriorNuClusters):
    assert(len(allClusterPrecisionModes) == len(allClusterSizes))
    
    priorNuNoise = float(p + 1)
    noisePart = oneSigmaPart(priorNuNoise, n, p, noiseTrace, posteriorNuNoise, 0)
    
    allClusterPartSum = 0.0
    for j in xrange(len(allClusterSizes)):
        clusterSize = allClusterSizes[j]
        assert(clusterSize == allClusterPrecisionModes[j].shape[0])
        priorNuCluster = float(clusterSize + 1)
        allClusterPartSum += oneSigmaPart(priorNuCluster, n, clusterSize, allClusterTraces[j], allPosteriorNuClusters[j], n)
        allClusterPartSum += noiseClusterInteractionPart(allPosteriorNuClusters[j], posteriorNuNoise, p, clusterSize, n, allJointTraces[j])
    
    return noisePart + allClusterPartSum

