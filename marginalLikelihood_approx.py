
import numpy
import marginalHelper
import idcHelper
import scipy.special
import KLapproximation
import scipy.optimize

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


# def getClusterPrecisionMat(p, allClusterPrecisions):
#     # Cinv = C^{-1}
#     Cinv = numpy.zeros((p,p))
#     
#     nextClusterStartsAt = 0
#     
#     for clusterPrecision in allClusterPrecisions:
#         nrVariablesInCluster = clusterPrecision.shape[0]
#         startId = nextClusterStartsAt
#         endId = nextClusterStartsAt + nrVariablesInCluster
#         Cinv[startId:endId, startId:endId] = clusterPrecision
#         nextClusterStartsAt += nrVariablesInCluster
#     
#     
#     Cinv = numpy.asmatrix(Cinv)
#     return Cinv

# def getKLApproximationSlowAndWrong(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampleCov, allBlockSampleCovs, noisePrecisionMode, allClusterPrecisionModes, USE_NOISE_IN_MODEL, posteriorNuNoise, posteriorNuCluster):
#     
#     if USE_NOISE_IN_MODEL:
#         p = noisePrecisionMode.shape[0]
#         priorNu = p + 1
#         priorSigma = PRIOR_SCALE_FOR_NOISE * numpy.eye(p)
#         noiseCovarianceMode = numpy.linalg.inv(noisePrecisionMode)
#         logFacPrior = (priorNu + p + 1)
#         logFac = (n)
#         traceMat = n * sampleCov + priorSigma
#         
#         posteriorSigma = noiseCovarianceMode * (posteriorNuNoise + p + 1)
#         clusterPrecisionModeAsOneMatrix = getClusterPrecisionMat(p, allClusterPrecisionModes)
#         posteriorSigmaNewApprox = numpy.linalg.inv(noisePrecisionMode + clusterPrecisionModeAsOneMatrix) * (posteriorNuNoise + p + 1)
#         posteriorInvSigma = numpy.linalg.inv(posteriorSigma)
#         print "traceMat = "
#         idcHelper.showMatrix(traceMat)
#         print "posteriorInvSigma = "
#         idcHelper.showMatrix(posteriorInvSigma)
#         upperBound = - 0.5 * (marginalHelper.getExpLogDetPrecisionMinusTrace(posteriorNuNoise, posteriorSigmaNewApprox, posteriorInvSigma, logFacPrior, traceMat) 
#                               + marginalHelper.getExpLogDetPrecisionOnly(priorNu + p + 1, posteriorSigma, logFac)) - marginalHelper.getEntropy(posteriorNuNoise, posteriorSigma)
#         print "KL (noise only) = ", upperBound        
#         
#     else:
#         upperBound = 0
#     
#     
#     for j in xrange(len(allBlockSampleCovs)):
#         clusterPrecisionMode = allClusterPrecisionModes[j]
#         clusterCovarianceMode = numpy.linalg.inv(clusterPrecisionMode)
#         clusterSize = clusterPrecisionMode.shape[0]
#         priorNu = clusterSize + 1
#         priorSigma = PRIOR_SCALE_FOR_CLUSTER * numpy.eye(clusterSize)
#         assert(clusterSize == clusterPrecisionMode.shape[0])
#         logFac = (priorNu + clusterSize + 1 + n)
#         traceMat = n * allBlockSampleCovs[j] + priorSigma
#         
#         posteriorSigma = clusterCovarianceMode * (posteriorNuCluster + clusterSize + 1) # n * allBlockSampleCovs[j] + priorSigma
#         posteriorInvSigma = numpy.linalg.inv(posteriorSigma)
#         upperBound += - 0.5 * marginalHelper.getExpLogDetPrecisionMinusTrace(posteriorNuCluster, posteriorSigma, posteriorInvSigma, logFac, traceMat) - marginalHelper.getEntropy(posteriorNuCluster, posteriorSigma)
#      
# #         assert(type(noisePrecisionMode) == numpy.matrixlib.defmatrix.matrix)
# #         assert(type(traceMat) == numpy.matrixlib.defmatrix.matrix)
# #         tracePart = numpy.trace(traceMat * clusterPrecisionMode)
# #         upperBound += newFastCalculationOfKLApproxForOneSigma(priorNu, n, clusterSize, tracePart, posteriorNuCluster)
#         
#     return upperBound


def checkIsSorted(idsInOrder):
    for i in xrange(idsInOrder.shape[0] - 1):
        assert(idsInOrder[i] <= idsInOrder[i + 1])
    return

# checked
# uses a permutation matrix P such that 
# P * clusteringResult = ids sorted by clusters
def getPermutedCovarianceMatrixAndClusterSizes(sampleCovariance, clusteringResult):
    p = sampleCovariance.shape[0]
    clusterIds = idcHelper.getIdsOfEachCluster(clusteringResult)
    
    clusterSizes = []
    for oneCluster in clusterIds:
        clusterSizes.append(len(oneCluster))
    
    permutationMatrix = numpy.zeros((p,p), dtype = numpy.int)
    i = 0
    for oneCluster in clusterIds:
        for elemId in oneCluster:
            permutationMatrix[i,elemId] = 1
            i += 1
    
    sortedClustering = getReorderIds(permutationMatrix, clusteringResult)
    checkIsSorted(sortedClustering)
    return getReorderedSampleCov(permutationMatrix, sampleCovariance), clusterSizes, sortedClustering



# def getLogMarginalForRuntimeTest(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, originalSampleCovariance, clustering, SOLVER):
#     assert(n >= 10)
#     
#     USE_NOISE_IN_MODEL = True
#     
#     sortedSampleCovariance, allClusterSizes, sortedClustering = getPermutedCovarianceMatrixAndClusterSizes(originalSampleCovariance, clustering)
#     # print "sortedClustering = ", sortedClustering
#     # print "sortedSampleCovariance = "
#     # idcHelper.showMatrix(sortedSampleCovariance)
#     
#     # noisePrecisionMode, allClusterPrecisionModes, debugReturnValue = newtonMethod.findPosteriorModeNC(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#     
#     import time
#     startTime = time.time()
#     
#     if SOLVER == "ADMM-3BLOCK":
#         noisePrecisionMode, allClusterPrecisionModes, debugReturnValue = convexSolutionWithADMM.findPosteriorMode3Block(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#     elif SOLVER == "ADMM-2BLOCK":
#         noisePrecisionMode, allClusterPrecisionModes, debugReturnValue = convexSolutionWithADMM.findPosteriorMode2Block(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#     elif SOLVER == "proposed-EXACT" or SOLVER == "proposed-CG" or SOLVER == "proposedWithPythonNewton":
#         noisePrecisionMode, allClusterPrecisionModes, debugReturnValue = newtonMethod.findPosteriorMode(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL, SOLVER, None)
#     elif SOLVER == "primalCoordinateDescent":
#         noisePrecisionMode, allClusterPrecisionModes, debugReturnValue = primalMethod.findPosteriorMode(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#     else:
#         assert(SOLVER == "CVXPY")
#         noisePrecisionMode, allClusterPrecisionModes, debugReturnValue = marginalHelper.findPosteriorModeWithCVXPY(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
# 
#     print "objValue (my solution) = ", debugReturnValue
#     runtime = (time.time() - startTime) / 60.0
#     print "one optimization time = ", runtime
#     return debugReturnValue, runtime
    
    
# # checked
# def getLogMarginal(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, originalSampleCovariance, clustering, SOLVER):
#     assert(n >= 10)
#     
#     USE_NOISE_IN_MODEL = True
#     initialNoisePrecision = None
#     
#     sortedSampleCovariance, allClusterSizes, sortedClustering = getPermutedCovarianceMatrixAndClusterSizes(originalSampleCovariance, clustering)
#     # print "sortedClustering = ", sortedClustering
#     # print "sortedSampleCovariance = "
#     # idcHelper.showMatrix(sortedSampleCovariance)
#     
#     # noisePrecisionMode, allClusterPrecisionModes, debugReturnValue = newtonMethod.findPosteriorModeNC(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#     
#     import time
#     startTime = time.time()
#     
#     errorOccured = False
#     
#     if SOLVER == "compareCVXPYandProposed":
#         # for debugging only
#         # SOLVER_PROPOSED = "proposed-EXACT"
#         # SOLVER_PROPOSED = "proposedWithBFGS_dual"
#         # noisePrecisionMode, allClusterPrecisionModes, objValueProposed = newtonMethod.findPosteriorMode(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL, SOLVER_PROPOSED, initialNoisePrecision)
#         noisePrecisionMode, allClusterPrecisionModes, objValueProposed = convexSolutionWithADMM.findPosteriorMode3Block(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#     
#         _, _, objValueCVXPY = marginalHelper.findPosteriorModeWithCVXPY(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#         
#         print "objValue (my solution) = ", objValueProposed
#         print "objValue (CVXPY) = ", objValueCVXPY
#         if objValueProposed >= objValueCVXPY:
#             errorOccured = True
#     
#     elif SOLVER == "ADMM-3BLOCK":
#         noisePrecisionMode, allClusterPrecisionModes, objValue = convexSolutionWithADMM.findPosteriorMode3Block(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#     
#     elif SOLVER == "ADMM-2BLOCK":
#         noisePrecisionMode, allClusterPrecisionModes, objValue = convexSolutionWithADMM.findPosteriorMode2Block(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#         
#     elif SOLVER == "proposed-EXACT" or SOLVER == "proposed-CG" or SOLVER == "proposedWithBFGS_dual":
#         noisePrecisionMode, allClusterPrecisionModes, objValue = newtonMethod.findPosteriorMode(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL, SOLVER, initialNoisePrecision)
#         # noisePrecisionMode, allClusterPrecisionModes, objValue = convexSolutionWithADMM.findPosteriorMode(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#         # noisePrecisionMode, allClusterPrecisionModes, objValue = convexSolutionWithDual.findPosteriorMode(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#     else:
#         assert(SOLVER == "proposed-CVXPY")
#         noisePrecisionMode, allClusterPrecisionModes, objValue = marginalHelper.findPosteriorModeWithCVXPY(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, USE_NOISE_IN_MODEL)
#     
#     # print "objValue (my solution) = ", objValue
#     print "one optimization time = ", (time.time() - startTime) / 60.0
#     
#     #  print "clustering = " +  str(clustering) + ", objFunctionValue = ", debugReturnValue
#     
#     
#     allBlockSampleCovs = idcHelper.getBlockCovariance(sortedSampleCovariance, sortedClustering)
#     
#     p = sortedSampleCovariance.shape[0]
#     testRatios = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
#     
# #     lowestKL = float("inf")
# #     bestRatio = None
# #     for ratio in testRatios:
# #         posteriorNuNoiseTry = p + ratio * n
# #         klApproximation = getKLApproximationFastNoise(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allBlockSampleCovs, noisePrecisionMode, allClusterPrecisionModes, USE_NOISE_IN_MODEL, posteriorNuNoiseTry)
# #         if klApproximation < lowestKL:
# #             lowestKL = klApproximation
# #             bestRatio = ratio
#     
#     bestRatio = 1.0
#     
#     bestPosteriorNuNoise = p + bestRatio * n
#     
#     print "bestRatio (noise) = ", bestRatio
#     
# #     lowestKL = float("inf")
# #     bestRatio = None
# #     for ratio in testRatios:
# #         posteriorNuClusterTry = p + ratio * n
# #         klApproximation = getKLApproximationFastCluster(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allBlockSampleCovs, noisePrecisionMode, allClusterPrecisionModes, USE_NOISE_IN_MODEL, posteriorNuClusterTry)
# #         if klApproximation < lowestKL:
# #             lowestKL = klApproximation
# #             bestRatio = ratio
#         
#     bestPosteriorNuCluster = p + bestRatio * n
#     
#     print "bestRatio (cluster) = ", bestRatio
#     print "bestPosteriorNuNoise = ", bestPosteriorNuNoise
#     print "bestPosteriorNuCluster = ", bestPosteriorNuCluster
#     # assert(False)
#     
#     jointLogProb = marginalHelper.getJointLogProb(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, noisePrecisionMode, allClusterPrecisionModes, USE_NOISE_IN_MODEL)
#     # print "jointLogProb = ", jointLogProb
#     posteriorLogProb = marginalHelper.getLogPosteriorApproximation(noisePrecisionMode, allClusterPrecisionModes, bestPosteriorNuNoise, bestPosteriorNuCluster, USE_NOISE_IN_MODEL, noisePrecisionMode, allClusterPrecisionModes)
#     logMarginalApproximation = jointLogProb - posteriorLogProb
#     return logMarginalApproximation, errorOccured




    




# for testing only
# returns permutationMatrix P, used for P * assignVec = permutated assignVec
def getPermutationMatrixForTesting(p):
    assert(int(p / 2) * 2  == p)
    
    permMat = numpy.zeros((p,p), dtype = numpy.int_)
    for i in xrange(p / 2):
        permMat[i*2,i] = 1
        permMat[i*2 + 1,3 + i] = 1
    
    return permMat


def getReorderedSampleCov(P, sampleCovariance):
    P = numpy.asmatrix(P)
    assert(type(P) == numpy.matrixlib.defmatrix.matrix)
    assert(type(sampleCovariance) == numpy.matrixlib.defmatrix.matrix)
    return P * sampleCovariance * P.transpose()
    
def getReorderIds(permutationMatrix, clustering):
    assert(permutationMatrix.shape[0] == clustering.shape[0])
    reorderIds = numpy.dot(permutationMatrix, clustering)
    assert(len(reorderIds.shape) == 1 and reorderIds.shape[0] == permutationMatrix.shape[0])
    return reorderIds


# used only for testing
def getClusteringCandidatesForTesting(allClusteringSizes):
    
    allClusterings = []
    for oneClusterSizes in allClusteringSizes:
        allClusterings.append(marginalHelper.getSimpleClusterAssignments(oneClusterSizes))
    
    return allClusterings
    

def permutateEveryThing(hiddenDataIds, sampleCovariance, clusteringCandidates):
    
    print "RUN PERMUTATION TEST"
    # print "hiddenDataIds (before permutation)  = ", hiddenDataIds
    # print "sample precision matrix (before permutation) = "
    # idcHelper.showMatrix(numpy.linalg.inv(sampleCovariance))
     
    P = getPermutationMatrixForTesting(sampleCovariance.shape[0])
    hiddenDataIds = getReorderIds(P, hiddenDataIds)
    sampleCovariance = getReorderedSampleCov(P, sampleCovariance)
    
    newClusteringCandidates = []
    for oneClusteringCandidate in clusteringCandidates:
        newClusteringCandidates.append(getReorderIds(P, oneClusteringCandidate))
     
    print "hiddenDataIds (after permutation) = ", hiddenDataIds
    print "sample precision matrix (after permutation) = "
    idcHelper.showMatrix(numpy.linalg.inv(sampleCovariance))
    
    return hiddenDataIds, sampleCovariance, newClusteringCandidates




    


# checked
def getRelevantVariableSubset(clusteringA, clusteringB, minNrVariables):
    assert(clusteringA.shape[0] == clusteringB.shape[0])
    p = clusteringA.shape[0]
    
    assert(minNrVariables <= p and minNrVariables >= 2 * numpy.max(clusteringA))
    
    clusterIdToVars_A = idcHelper.getIdsOfEachClusterAsSet(clusteringA)
    clusterIdToVars_B = idcHelper.getIdsOfEachClusterAsSet(clusteringB)
    
    assert(numpy.min(clusteringA) == 1 and numpy.min(clusteringB) == 1)
    assert(len(clusterIdToVars_A) == numpy.max(clusteringA))
    assert(len(clusterIdToVars_B) == numpy.max(clusteringB))
    
    allSelectVariables = set()
    
    differenceDetected = False
    
    while len(allSelectVariables) < minNrVariables:
        for c_A in clusterIdToVars_A.keys():
            if len(clusterIdToVars_A[c_A]) > 0:
                x = clusterIdToVars_A[c_A].pop()
                c_B = clusteringB[x]
                clusterIdToVars_B[c_B].remove(x)
                
                allSelectVariables.add(x)
                
                symmetricSetDifference = clusterIdToVars_A[c_A] ^ clusterIdToVars_B[c_B]
                if len(symmetricSetDifference) > 0:
                    y = symmetricSetDifference.pop()
                    assert((y in clusterIdToVars_A[c_A] and y not in clusterIdToVars_B[c_B]) or (y not in clusterIdToVars_A[c_A] and y in clusterIdToVars_B[c_B]))
                    clusterIdToVars_A[clusteringA[y]].remove(y)
                    clusterIdToVars_B[clusteringB[y]].remove(y)
                    allSelectVariables.add(y)
                    differenceDetected = True
                    
    
    assert(differenceDetected)
    return list(allSelectVariables)


# def getRelevantVariableSubsetEvenly(clusteringA, clusteringB, minNrVariables):
#     assert(clusteringA.shape[0] == clusteringB.shape[0])
#     p = clusteringA.shape[0]
#     
#     assert(minNrVariables <= p and minNrVariables >= 2 * numpy.max(clusteringA))
#     
#     clusterIdToVars_A = idcHelper.getIdsOfEachClusterAsSet(clusteringA)
#     clusterIdToVars_B = idcHelper.getIdsOfEachClusterAsSet(clusteringB)
#     
#     assert(numpy.min(clusteringA) == 1 and numpy.min(clusteringB) == 1)
#     assert(len(clusterIdToVars_A) == numpy.max(clusteringA))
#     assert(len(clusterIdToVars_B) == numpy.max(clusteringB))
#     
#     allSelectVariables = set()
#     
#     print "len(clusterIdToVars_A) = ", len(clusterIdToVars_A)
#     print "len(clusterIdToVars_B) = ", len(clusterIdToVars_B)
#     
#     allSelectVariables.update(list(clusterIdToVars_A[1])[0:10])
#     allSelectVariables.update(list(clusterIdToVars_A[2])[0:20])
#     allSelectVariables.update(list(clusterIdToVars_A[3])[0:5])
#     allSelectVariables.update(list(clusterIdToVars_A[4])[0:5])
#     print "allSelectVariables = "
#     print allSelectVariables
#     return list(allSelectVariables)
#     assert(False)
#     
#     differenceDetected = False
#     
#     while len(allSelectVariables) < minNrVariables:
#         for c_A in clusterIdToVars_A.keys():
#             if len(clusterIdToVars_A[c_A]) > 0:
#                 x = clusterIdToVars_A[c_A].pop()
#                 c_B = clusteringB[x]
#                 clusterIdToVars_B[c_B].remove(x)
#                 
#                 allSelectVariables.add(x)
#                 
#                 symmetricSetDifference = clusterIdToVars_A[c_A] ^ clusterIdToVars_B[c_B]
#                 if len(symmetricSetDifference) > 0:
#                     y = symmetricSetDifference.pop()
#                     assert((y in clusterIdToVars_A[c_A] and y not in clusterIdToVars_B[c_B]) or (y not in clusterIdToVars_A[c_A] and y in clusterIdToVars_B[c_B]))
#                     clusterIdToVars_A[clusteringA[y]].remove(y)
#                     clusterIdToVars_B[clusteringB[y]].remove(y)
#                     allSelectVariables.add(y)
#                     differenceDetected = True
#                     
# #                 elif len(clusterIdToVars_A[c_A]) > 0:
# #                     # the two clusterings are exactly the same AND there is still an element in it.
# #                     y = clusterIdToVars_A[c_A].pop()
# #                     assert(clusteringB[y] == c_B)
# #                     clusterIdToVars_B[c_B].remove(y)
# #                     allSelectVariables.add(y)
#     
#                 # print "x = ", x
#                 # print "y = ", y
#                 # break
#     
#     assert(differenceDetected)
#     return list(allSelectVariables)

# checked
def getSubClustering(clustering, selectedVarIds):
    
    subClustering = clustering[selectedVarIds]
    
    # print "subClustering = ", subClustering
    subClusteringVars = set(subClustering)
    
    currentValidClusterId = 1
    # rename such that we start with 1 and each cluster number is represented
    for clusterId in xrange(1, numpy.max(subClustering) + 1, 1):
        if clusterId in subClusteringVars:
            subClustering[subClustering == clusterId] = currentValidClusterId
            currentValidClusterId += 1
    
    return subClustering


def testSubClustering():
    clustering = numpy.asarray([1,2,7,3,2])
    
    print clustering
    subClustering = getSubClustering(clustering, [2,1,3,4])
    
    print subClustering
    print subClustering.shape[0]
    
    clusteringA = numpy.asarray([1,1,1,1,1,1,1])
    clusteringB = numpy.asarray([1,1,1,1,2,1,1])
    minNrVariables = 4
    selectedVars = getRelevantVariableSubset(clusteringA, clusteringB, minNrVariables)
    
    selectedVars = [3]
    print "selectedVars = ", selectedVars
    
    S = numpy.asarray([[2.0, 0.8, 0.6, 0.3], [0.8, 1.5, 0.5, 0.2], [0.6, 0.5, 1.2, 0.9], [0.3, 0.2, 0.9, 3.0]])

    print "S  = "
    print S
    print "sub cov matrix = "
    print idcHelper.getSubCovarianceMatrix(S, selectedVars)
    
# testSubClustering()


# reading checked
# def randomSubSampleEvaluation(allCollectedClusteringsOriginal, minNrVariables, PRIOR_SCALE_FOR_NOISE, n, sampledCovTrain, SOLVER, beta, df):
#     
#     PRIOR_SCALE_FOR_CLUSTER = 1.0
#     
#     allCollectedClusterings = copy.copy(allCollectedClusteringsOriginal) # make a copy, because we pop
#     bestClustering = allCollectedClusterings.pop()
#                 
#     for currentClustering in allCollectedClusterings:
#         
#         selectedVarIds = getRelevantVariableSubset(bestClustering, currentClustering, minNrVariables)
#         sampledCovTrainVarSubSet = idcHelper.getSubCovarianceMatrix(sampledCovTrain, selectedVarIds)
#         currentClusteringVarSubSet = getSubClustering(currentClustering, selectedVarIds)
#         bestClusteringVarSubSet = getSubClustering(bestClustering, selectedVarIds)
#         
#         
#         if PRIOR_SCALE_FOR_NOISE != "AVG":
#             assert(PRIOR_SCALE_FOR_NOISE >= 0.01 and PRIOR_SCALE_FOR_NOISE <= 200)
#             marginalLikelihoodCurrentAssignment = betaModel.getLogMarginalPrincipledNew(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampledCovTrainVarSubSet, currentClusteringVarSubSet, SOLVER, beta, df)
#             marginalLikelihoodBestAssignment = betaModel.getLogMarginalPrincipledNew(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampledCovTrainVarSubSet, bestClusteringVarSubSet, SOLVER, beta, df)
#         else:
#             assert(False) # needs to be updated
#             marginalLikelihoodCurrentAssignment = getLogMarginalAverage(PRIOR_SCALE_FOR_CLUSTER, n, sampledCovTrainVarSubSet, currentClusteringVarSubSet, SOLVER)
#             marginalLikelihoodBestAssignment = getLogMarginalAverage(PRIOR_SCALE_FOR_CLUSTER, n, sampledCovTrainVarSubSet, bestClusteringVarSubSet, SOLVER)
#         
#         assert(marginalLikelihoodCurrentAssignment != marginalLikelihoodBestAssignment)
# 
#         # print "currentClusteringVarSubSet = "
#         # idcHelper.showVector(currentClusteringVarSubSet)
#         # print "bestClusteringVarSubSet = "
#         # idcHelper.showVector(bestClusteringVarSubSet)
#         # print "marginalLikelihoodCurrentAssignment = ", marginalLikelihoodCurrentAssignment
#         # print "marginalLikelihoodBestAssignment = ", marginalLikelihoodBestAssignment
#         
#         if marginalLikelihoodCurrentAssignment >= marginalLikelihoodBestAssignment:
#             bestClustering = currentClustering
#         
#     return numpy.copy(bestClustering)




# def optimizePosteriorNusOld(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, sortedClustering, noisePrecisionMode, allClusterPrecisionModes):
#     
#     p = sortedSampleCovariance.shape[0]
#     allBlockSampleCovs = idcHelper.getBlockCovariance(sortedSampleCovariance, sortedClustering)
#     allJointTraces, allClusterTraces, noiseTrace = KLapproximation.precalcuateRelevantTraces(sortedClustering, PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allBlockSampleCovs, noisePrecisionMode, allClusterPrecisionModes)
#     
#     posteriorNuNoise = float(p + 1)
#     
#     allPosteriorNuClusters = float(p + n + 1) * numpy.ones(len(allClusterSizes))
#     
#     for j in xrange(len(allClusterSizes)):
#         clusterSize = allClusterSizes[j]
#         priorNuCluster = float(clusterSize + 1)
#         allPosteriorNuClusters[j] = float(n + priorNuCluster)
#         
#     # posteriorNuNoise = 1001000.0
#     # allPosteriorNuClusters = 41.0 * numpy.ones(len(allClusterSizes))
#     
#     # print "posteriorNuNoise = ", posteriorNuNoise
#     
#     klDivApproxPrevious = KLapproximation.evalKLApproximation(allClusterSizes, allClusterPrecisionModes, allJointTraces, allClusterTraces, noiseTrace, n, p, posteriorNuNoise, allPosteriorNuClusters)
#     print "klDivApprox (before optimization) = ", klDivApproxPrevious
#     
#     infoStr = str(posteriorNuNoise) + "," + str(allPosteriorNuClusters)
#     print infoStr + " klDivApprox = " + str(klDivApproxPrevious)
#     
#     return posteriorNuNoise, allPosteriorNuClusters
# 
# 
# 
#     SEARCH_INTERVAL = (float(p + 1), float(100 * posteriorNuNoise))
#     
#     # SEARCH_INTERVAL = (float(p + 1), float(100 * (p + n)))
#     
#     # print "SEARCH_INTERVAL = ", SEARCH_INTERVAL
#     # assert(False)
#     
#     
#     for i in xrange(1000):
#         print str(i) + " iteration - KL-diveregence minimization "
#         
#         def funcPosteriorNuNoise(posteriorNuNoise_local):
#             klDivApprox = KLapproximation.evalKLApproximation(allClusterSizes, allClusterPrecisionModes, allJointTraces, allClusterTraces, noiseTrace, n, p, posteriorNuNoise_local[0], allPosteriorNuClusters)
#             return klDivApprox
#     
#         result = scipy.optimize.minimize(funcPosteriorNuNoise, x0 = posteriorNuNoise, method = 'TNC', jac = False, bounds = [SEARCH_INTERVAL])
#         posteriorNuNoise = result.x[0]
#         #     print result.success
#         #     print result.status
#         #     print result.message
#         #     print result.nit
#         
#         
# #         for j in xrange(len(allClusterSizes)):
# #             def funcPosteriorNuClusters(posteriorNuOneCluster_local):
# #                 allPosteriorNuClusters_local = numpy.copy(allPosteriorNuClusters)
# #                 allPosteriorNuClusters_local[j] = posteriorNuOneCluster_local
# #                 klDivApprox = KLapproximation.evalKLApproximation(allClusterSizes, allClusterPrecisionModes, allJointTraces, allClusterTraces, noiseTrace, n, p, posteriorNuNoise, allPosteriorNuClusters_local)
# #                 return klDivApprox
# #             # result = scipy.optimize.minimize_scalar(funcPosteriorNuClusters, bracket = SEARCH_INTERVAL, bounds = SEARCH_INTERVAL, method = 'Bounded')
# #             result = scipy.optimize.minimize(funcPosteriorNuClusters, x0 = allPosteriorNuClusters[j], method = 'TNC', jac = False, bounds = [SEARCH_INTERVAL])
# #             allPosteriorNuClusters[j] = result.x[0]
#         
#         
#         # update all simultanously
#         def funcPosteriorNuClusters(posteriorNuOneCluster_local):
#             allPosteriorNuClusters_local = posteriorNuOneCluster_local * numpy.ones(len(allClusterSizes))
#             klDivApprox = KLapproximation.evalKLApproximation(allClusterSizes, allClusterPrecisionModes, allJointTraces, allClusterTraces, noiseTrace, n, p, posteriorNuNoise, allPosteriorNuClusters_local)
#             return klDivApprox
#         result = scipy.optimize.minimize(funcPosteriorNuClusters, x0 = allPosteriorNuClusters[0], method = 'TNC', jac = False, bounds = [SEARCH_INTERVAL])
#         allPosteriorNuClusters = result.x[0] * numpy.ones(len(allClusterSizes))
#         
#         klDivApprox = KLapproximation.evalKLApproximation(allClusterSizes, allClusterPrecisionModes, allJointTraces, allClusterTraces, noiseTrace, n, p, posteriorNuNoise, allPosteriorNuClusters)
#         print "klDivApprox = ", klDivApprox
#         
#         if (numpy.abs(klDivApproxPrevious - klDivApprox) < 0.0001):
#             break
#         else:
#             assert(klDivApprox < klDivApproxPrevious)
#             klDivApproxPrevious = klDivApprox
#     
#         
#     infoStr = str(posteriorNuNoise) + "," + str(allPosteriorNuClusters)
#     print infoStr + " klDivApprox = " + str(klDivApprox)
#     return posteriorNuNoise, allPosteriorNuClusters



# checked
def optimizePosteriorNus(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, sortedClustering, noisePrecisionMode, allClusterPrecisionModes):
    
    p = sortedSampleCovariance.shape[0]
    allBlockSampleCovs = idcHelper.getBlockCovariance(sortedSampleCovariance, sortedClustering)
    allJointTraces, allClusterTraces, noiseTrace = KLapproximation.precalcuateRelevantTraces(sortedClustering, PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allBlockSampleCovs, noisePrecisionMode, allClusterPrecisionModes, 1.0)
    
    def funcPosteriorNuNoiseScalar(posteriorNu_local):
        priorNuNoise = float(p + 1)
        return KLapproximation.oneSigmaPart(priorNuNoise, n, p, noiseTrace, posteriorNu_local, 0)
    
    SEARCH_INTERVAL = (float(p + 1), float(100 * (p + n + 1)))
    result = scipy.optimize.minimize_scalar(funcPosteriorNuNoiseScalar, bounds = SEARCH_INTERVAL, method = 'bounded')
    posteriorNuNoise = result.x
    
    totalKL = funcPosteriorNuNoiseScalar(posteriorNuNoise)
    
    allPosteriorNuClusters = numpy.zeros(len(allClusterSizes))
        
    for j in xrange(len(allClusterSizes)):
        clusterSize = allClusterSizes[j]
        assert(clusterSize == allClusterPrecisionModes[j].shape[0])
        priorNuCluster = float(clusterSize + 1)
        
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




# checked
# def getLogMarginalAverage(PRIOR_SCALE_FOR_CLUSTER, n, originalSampleCovariance, clustering, SOLVER):
#     assert(PRIOR_SCALE_FOR_CLUSTER == 1.0)
#     assert(SOLVER == "proposed-EXACT")
#     assert(n >= 10)
#     
#     allNoiseScales = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0]
#     
#     totalLogMarginal = 0.0
#     for PRIOR_SCALE_FOR_NOISE in allNoiseScales:
#         print "PRIOR_SCALE_FOR_NOISE = ", PRIOR_SCALE_FOR_NOISE
#         logMarginalApproximation = getLogMarginalPrincipled(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, originalSampleCovariance, clustering, SOLVER)
#         totalLogMarginal += logMarginalApproximation
#     
#     avgLogMarginalApproximation = totalLogMarginal / float(len(allNoiseScales))
#     return avgLogMarginalApproximation

