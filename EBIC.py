
import numpy
import idcHelper # checked
import baselines

import shared.clusteringHelper as clusteringHelper
import sklearn.metrics
import sklearn.cluster

import betaModel
from ChibEstimator import ChibEstimator

import marginalHelper
import time
import multiprocessing


# def getBlockCovarianceNew(dataVectorsAllTrain, clusterAssignments):
#     idsInClusters = getIdsOfEachCluster(clusterAssignments)
#     
#     allBlockCovs = []
#     for varIdsInCurrentCluster in idsInClusters:
#         dataVectorsAllTrainBlock = dataVectorsAllTrain[:, varIdsInCurrentCluster]
#         blockCov = numpy.asmatrix(numpy.cov(dataVectorsAllTrainBlock.transpose(), rowvar=True, bias=True))
#         allBlockCovs.append(numpy.asmatrix(blockCov))
#         
#     return allBlockCovs

# counts the number of undirected edges, according to definition of edge in paper
def getNumberOfEdges(allBlockCovs):
    nrEdges = 0
    for blockCov in allBlockCovs:
        p = blockCov.shape[0]
        assert(p >= 1)
        nrEdges += 0.5 * (p * p - p)
    return nrEdges

# def test():
#     M = numpy.asarray([[2.0, 0.8, 0.6, 0.3], [0.8, 1.5, 0.5, 0.2], [0.6, 0.5, 1.2, 0.9], [0.3, 0.2, 0.9, 3.0]])
#     clusterAssignments = numpy.asarray([2, 1, 2, 2])
#     # clusterAssignments = hiddenVarIds
#     allCovs = getBlockCovariance(M, clusterAssignments)
#     
#     print "allCovs:"
#     for z, blockCov in enumerate(allCovs):
#         print "block ", z
#         print blockCov.shape[0], blockCov.shape[1]
#     
#     
# test()


# always use it since otherwise for some data sets the  resulting matrix is not positive definite.
def getNumericalStableInverseCovariance(cov):
    smoothingParam = 0.001 # for numerical stability
    covSize = cov.shape[0]
    invCov = numpy.linalg.inv(cov + smoothingParam * numpy.eye(covSize))
    invCov = numpy.asmatrix(invCov)
    return invCov

# covariance matrix S
# sample size n
def getUnpenalizedLogLikelihood(allBlockCovs, n):
    
    totalLL = 0.0
    for blockCov in allBlockCovs:
        invBlockCov = getNumericalStableInverseCovariance(blockCov)
        totalLL += idcHelper.getLogLikelihood(invBlockCov, blockCov)
    
    return (float(n) / 2.0) * totalLL


    
    

# checked
# "Extended Bayesian Information Criteria for Gaussian Graphical Models", NIPS, 2011
def getEBIC(S, n, clusterAssignments, gamma):
    p = S.shape[0]
    # if p >= n:
    #     print "********* WARNING: p >= n ***********"
    
    allBlockCovs = idcHelper.getBlockCovariance(S, clusterAssignments)
    nrEdges = getNumberOfEdges(allBlockCovs)
    ln = getUnpenalizedLogLikelihood(allBlockCovs, n)
    
    EBICcriteria = - 2.0 * ln + nrEdges * numpy.log(n) + 4.0 * nrEdges * gamma * numpy.log(p)
    return EBICcriteria




def getSilhouetteScore(precM, clusterAssignments):
    
    # note: !!
    # sklearn.metrics.silhouette_score the best is -1.0 and the worst value is 1.0
    if numpy.sum(clusterAssignments) == clusterAssignments.shape[0]:
        # all in one cluster assignment
        return 1.0
    
    if clusterAssignments.shape[0] <= 12:
        # print "WARNING: IGNORE getSilhouetteScore"
        return 1.0
    
    precMabs = numpy.abs(precM)    
    return -1.0 * sklearn.metrics.silhouette_score(precMabs, clusterAssignments, metric='precomputed')


def getCalinskiHarabazIndex(precM, clusterAssignments):
    
    if numpy.sum(clusterAssignments) == clusterAssignments.shape[0]:
        # all in one cluster assignment
        return 0.0
    
    if clusterAssignments.shape[0] <= 12:
        # print "WARNING: IGNORE getCalinskiHarabazIndex"
        return 0.0
    
    p = precM.shape[0]
    precMabs = numpy.abs(precM)
    
    withinClusterSum = 0.0
    withinClusterElems = 0.0
    
    outsideClusterSum = 0.0
    outsideClusterElems = 0.0
    
    for i in xrange(p):
        for j in xrange(i+1, p):
            if clusterAssignments[i] == clusterAssignments[j]:
                withinClusterSum += precMabs[i,j]
                withinClusterElems += 1.0
            else:
                outsideClusterSum += precMabs[i,j]
                outsideClusterElems += 1.0
    
    withinClusterMean = withinClusterSum / withinClusterElems
    outsideClusterMean = outsideClusterSum / outsideClusterElems
    
    withinClusterSqrSum = 0.0
    outsideClusterSqrSum = 0.0
    
    for i in xrange(p):
        for j in xrange(i+1, p):
            if clusterAssignments[i] == clusterAssignments[j]:
                withinClusterSqrSum += (withinClusterMean - precMabs[i,j]) ** 2
            else:
                outsideClusterSqrSum += (outsideClusterMean - precMabs[i,j]) ** 2
            
    withinClusterVar = withinClusterSqrSum / withinClusterElems
    outsideClusterVar = outsideClusterSqrSum / outsideClusterElems

    # print "withinClusterVar = ", withinClusterVar
    # print "outsideClusterVar = ", outsideClusterVar
    # assert(withinClusterVar > 0.0 and outsideClusterVar > 0.0)
    assert(outsideClusterVar > 0.0)
    
    return -1.0 * (withinClusterVar / outsideClusterVar)


def test():
    M = numpy.asarray([[2.0, 0.8, 0.06, 0.03], [0.8, 1.5, 0.05, 0.02], [0.06, 0.05, 1.2, 0.9], [0.03, 0.02, 0.9, 3.0]])
    clusterAssignments = numpy.asarray([1,1,0,0])
    print "M = "
    print M
    print "clusterAssignments = "
    print clusterAssignments
    print "CalinskiHarabazIndex = ", getCalinskiHarabazIndex(M, clusterAssignments)
    return

# test()


def getAIC(S, n, clusterAssignments):
    
    allBlockCovs = idcHelper.getBlockCovariance(S, clusterAssignments)
    nrEdges = getNumberOfEdges(allBlockCovs)
    ln = getUnpenalizedLogLikelihood(allBlockCovs, n)
    
    EBICcriteria = - 2.0 * ln + 2.0 * nrEdges
    return EBICcriteria


def containsClustering(allCollectedClusterings, newClustering):
    
    for existingClustering in allCollectedClusterings:
        if sklearn.metrics.adjusted_rand_score(existingClustering, newClustering) == 1.0:
            return True
    
    return False
    



def calculateMarginalLikelihoodProposed(allPassedData):
    
    try:
            
        allRuntimesQueue, PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampledCovTrain, SOLVER, beta, df, approximationMethod, M, clusterAssignments, clusteringId, estimatorType, kappa  = allPassedData
        
        startTimeOneClustering = time.time()
            
        print "proccess cluster number ", clusteringId
        print "number of samples = ", n
        print "clusterAssignments = "
        idcHelper.showVector(clusterAssignments)
        
        assert(PRIOR_SCALE_FOR_NOISE >= 0.01 and PRIOR_SCALE_FOR_NOISE <= 200)
        
        if approximationMethod == "variational":
            marginalLikelihood, _, _ = betaModel.getLogMarginalPrincipledNew(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampledCovTrain, clusterAssignments, SOLVER, beta, df)
        else:
            assert(approximationMethod == "MCMC")
            if estimatorType == "ChibEstimator":
                chib = ChibEstimator(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampledCovTrain, clusterAssignments, beta, df, kappa, str(clusteringId))
            else:
                assert(False)
                # assert(estimatorType == "ChibEstimator2Block")
                # chib = ChibEstimator2Block(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampledCovTrain, clusterAssignments, beta, df, str(clusteringId))
            BURN_IN_ITERATIONS = int(M * 0.1)
            marginalLikelihood = chib.estimateLogMarginal(M, BURN_IN_ITERATIONS)
                
        runtimeThisClustering = (time.time() - startTimeOneClustering) / 60.0
        print "finished - runtime for this clustering (min) = ", runtimeThisClustering
        allRuntimesQueue.put(runtimeThisClustering)
    
    except (KeyboardInterrupt):
        print "! GOT KEYBOARD INTERRUPT OR AN EXCEPTION !"
        allRuntimesQueue.task_done()
        print "task done for queue"
        
    return marginalLikelihood


def getBaselineScores(allCollectedClusterings, sampledCovTrain, n, scoreName):
    allScores = numpy.zeros(len(allCollectedClusterings))
    
    smoothingParam = 0.001 # for numerical stability
    sampledCovTrainInv = numpy.linalg.inv(sampledCovTrain + smoothingParam * numpy.eye(sampledCovTrain.shape[0]))
        
    for i, clusterAssignments in enumerate(allCollectedClusterings):
        if scoreName == "AIC":
            allScores[i] = getAIC(sampledCovTrain, n, clusterAssignments)
        elif scoreName == "BIC":
            allScores[i] = getEBIC(sampledCovTrain, n, clusterAssignments, 0.0)
        elif scoreName == "EBIC_0.5":
            allScores[i] = getEBIC(sampledCovTrain, n, clusterAssignments, 0.5)
        elif scoreName == "EBIC_1.0":
            allScores[i] = getEBIC(sampledCovTrain, n, clusterAssignments, 1.0)
        elif scoreName == "CalinskiHarabazIndex":
            allScores[i] = getCalinskiHarabazIndex(sampledCovTrainInv, clusterAssignments)
    
    return -1.0 * allScores

            
def evalWithEBIC(dataVectorsAllTrain, sampledCovTrain, n, p, allLambdaValues, allClustersCandidatesNrs, methodNamePenaltyType, spectralVariant, hiddenVarIds, USE_PROPOSED_MARGINAL, minNrVariables, SOLVER, debugInfo, PRIOR_SCALE_FOR_NOISE, ADD_TRUE_CLUSTERING, clusteringMethod, PRIOR_SCALE_FOR_CLUSTER, INCLUDES_ALL_IN_ONE, ONLY_TRUE_CLUSTERING, beta, df, approximationMethod, M, NUMBER_OF_CPUS, estimatorType, kappa):
    
    # see "Extended Bayesian Information Criteria for Gaussian Graphical Models", NIPS, 
    # gamma = 0.0
    
    nrClusterNrCandidates = allClustersCandidatesNrs.shape[0]
    
    
    bestSil = float("inf")
    bestClusteringSil = None # checked
    
    bestCHI = float("inf")
    bestClusteringCHI = None # checked
    
    bestAIC = float("inf") 
    bestClusteringAIC = None # checked
    
    bestNegMarginalProp = float("inf")
    bestClusteringNegMarginalProp = None # checked
    
    bestNegMarginal = float("inf")
    bestClusteringNegMarginal = None # checked
    
    bestNegMarginalWT = float("inf")
    bestClusteringNegMarginalWT = None # checked
    
    bestEBIC05 = float("inf")
    bestClusteringEBIC05 = None # checked
    
    bestEBIC1 = float("inf")
    bestClusteringEBIC1 = None # checked
    
    bestEBIC = float("inf")
    bestClusteringEBIC = None # checked
    
    if methodNamePenaltyType == "BigQuic":
        print "[ Start running BigQuic"
        assert(False)
        # allX = quic.BigQuic(dataVectorsNormalized, allLambdaValues)
        print "Finished running BigQuic ]"
    elif methodNamePenaltyType == "allL1ExceptDiagonalADMM" or methodNamePenaltyType == "allL1ExceptDiagonalADMM2":
        initMatrix = (numpy.zeros((p,p)), numpy.zeros((p,p)))
    elif methodNamePenaltyType == "tikhonov" or methodNamePenaltyType == "tikhonov2":
        eigVals, eigVecs = numpy.linalg.eigh(sampledCovTrain)
        initMatrix = (numpy.asmatrix(eigVecs), eigVals)
    else:
        initMatrix = numpy.eye(p)
    
    
    allCollectedClusterings = []
    
    if ONLY_TRUE_CLUSTERING:
        allCollectedClusterings.append(hiddenVarIds)
    else:
        if ADD_TRUE_CLUSTERING:
            allCollectedClusterings.append(hiddenVarIds)
        
        if INCLUDES_ALL_IN_ONE:
            allInOneCluster = numpy.ones(hiddenVarIds.shape[0], dtype = int)
            allCollectedClusterings.append(allInOneCluster)
        
        
        if clusteringMethod == "spectral":
            for lambdaCandidateId in xrange(len(allLambdaValues)):
                lambdaValue = allLambdaValues[lambdaCandidateId]
                print "test lambda value = ", lambdaValue
                if methodNamePenaltyType == "BigQuic":
                    assert(False)
                    # currentX = allX[lambdaCandidateId]
                else:
                    currentX, initMatrix = baselines.getClusteringGraphicalLassoVanilla(sampledCovTrain, lambdaValue, methodNamePenaltyType, initMatrix)
                   
                # print "number of edges = ", statHelper.countEdges(currentX)
                
                # print "final learned precision matrix (baseline) = "
                # idcHelper.showMatrix(currentX)
            
                if methodNamePenaltyType == "lapReg" or methodNamePenaltyType == "lapRegApprox" or methodNamePenaltyType == "allL2fast" or methodNamePenaltyType == "allL1ExceptDiagonalADMM2" or  methodNamePenaltyType == "tikhonov2":
                    L = idcHelper.getLaplacianSquare(currentX, spectralVariant)
                else:
                    L = idcHelper.getLaplacianAbs(currentX, spectralVariant)
                
                eigValsL, eigVecsL = numpy.linalg.eigh(L)
                 
                for clusterCandidateId in xrange(nrClusterNrCandidates):
                    numberOfClusters = allClustersCandidatesNrs[clusterCandidateId]
                    clusterAssignments = idcHelper.getClusteringFast(eigVecsL, numberOfClusters)
                    
                    # print "clusterAssignments = ", clusterAssignments
                    assert(numpy.min(clusterAssignments) == 1)
                    
                    if not containsClustering(allCollectedClusterings, clusterAssignments):
                        allCollectedClusterings.append(clusterAssignments)
                        
        elif clusteringMethod == "spectralWithoutReg":
            
            X = getNumericalStableInverseCovariance(sampledCovTrain)
            L = idcHelper.getLaplacianAbs(X, spectralVariant)
            eigValsL, eigVecsL = numpy.linalg.eigh(L)
            for clusterCandidateId in xrange(nrClusterNrCandidates):
                    numberOfClusters = allClustersCandidatesNrs[clusterCandidateId]
                    clusterAssignments = idcHelper.getClusteringFast(eigVecsL, numberOfClusters)
                    assert(numpy.min(clusterAssignments) == 1)
                    if not containsClustering(allCollectedClusterings, clusterAssignments):
                        allCollectedClusterings.append(clusterAssignments)
                
        elif clusteringMethod == "single" or clusteringMethod == "average":
            
            for clusterCandidateId in xrange(nrClusterNrCandidates):
                numberOfClusters = allClustersCandidatesNrs[clusterCandidateId]
                clusterAssignments = baselines.hierarchicalClustering(sampledCovTrain, numberOfClusters, clusteringMethod)
                if not containsClustering(allCollectedClusterings, clusterAssignments):
                    allCollectedClusterings.append(clusterAssignments)
        
        elif clusteringMethod == "kmeans":
            
            for clusterCandidateId in xrange(nrClusterNrCandidates):
                numberOfClusters = allClustersCandidatesNrs[clusterCandidateId]
                kMeansClustering = sklearn.cluster.KMeans(n_clusters=numberOfClusters)
                kMeansClustering.fit(dataVectorsAllTrain.transpose())
                clusterAssignments = idcHelper.letLabelsStartWith1(kMeansClustering.labels_)
                if not containsClustering(allCollectedClusterings, clusterAssignments):
                    allCollectedClusterings.append(clusterAssignments)
            
        elif clusteringMethod == "allInOne":
            
            # spectral clustering
            assert(methodNamePenaltyType == "allL1ExceptDiagonalADMM")
            for lambdaCandidateId in xrange(len(allLambdaValues)):
                lambdaValue = allLambdaValues[lambdaCandidateId]
                currentX, initMatrix = baselines.getClusteringGraphicalLassoVanilla(sampledCovTrain, lambdaValue, methodNamePenaltyType, initMatrix)
                L = idcHelper.getLaplacianAbs(currentX, spectralVariant)
                eigValsL, eigVecsL = numpy.linalg.eigh(L)
                 
                for clusterCandidateId in xrange(nrClusterNrCandidates):
                    numberOfClusters = allClustersCandidatesNrs[clusterCandidateId]
                    clusterAssignments = idcHelper.getClusteringFast(eigVecsL, numberOfClusters)
                    if not containsClustering(allCollectedClusterings, clusterAssignments):
                        allCollectedClusterings.append(clusterAssignments)
           
            # single
            for clusterCandidateId in xrange(nrClusterNrCandidates):
                numberOfClusters = allClustersCandidatesNrs[clusterCandidateId]
                clusterAssignments = baselines.hierarchicalClustering(sampledCovTrain, numberOfClusters, "single")
                if not containsClustering(allCollectedClusterings, clusterAssignments):
                    allCollectedClusterings.append(clusterAssignments)
            
            # average
            for clusterCandidateId in xrange(nrClusterNrCandidates):
                numberOfClusters = allClustersCandidatesNrs[clusterCandidateId]
                clusterAssignments = baselines.hierarchicalClustering(sampledCovTrain, numberOfClusters, "average")
                if not containsClustering(allCollectedClusterings, clusterAssignments):
                    allCollectedClusterings.append(clusterAssignments)
            
            # k-means
            for clusterCandidateId in xrange(nrClusterNrCandidates):
                numberOfClusters = allClustersCandidatesNrs[clusterCandidateId]
                kMeansClustering = sklearn.cluster.KMeans(n_clusters=numberOfClusters)
                kMeansClustering.fit(dataVectorsAllTrain.transpose())
                clusterAssignments = idcHelper.letLabelsStartWith1(kMeansClustering.labels_)
                if not containsClustering(allCollectedClusterings, clusterAssignments):
                    allCollectedClusterings.append(clusterAssignments)
                    
        else:
            assert(False)            
    
    
    print "total number of clusterings = ", len(allCollectedClusterings)
    
    
    if estimatorType == "ChibEstimatorDiagnosis":
        import random
        random.seed(423525)
        random.shuffle(allCollectedClusterings)
    
    
    avgRuntimeProposed = None
    allMarginalLikelihoodsProposedMethod = None
    allScoresBIC = numpy.zeros(len(allCollectedClusterings))
    allANMIs = numpy.zeros(len(allCollectedClusterings))
    allRS = numpy.zeros(len(allCollectedClusterings))
    
    
    if USE_PROPOSED_MARGINAL:
        assert(minNrVariables == "all")
        
        if estimatorType == "ChibEstimatorDiagnosis":
            print "KAPPA = ", kappa
            clusteringId = 0
            chib = ChibEstimator(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampledCovTrain, clusterAssignments, beta, df, kappa, str(clusteringId))
            BURN_IN_ITERATIONS = int(M * 0.1)
            chib.estimateLogMarginalDiagnosis(M, BURN_IN_ITERATIONS, str(n) + "n_" + str(sampledCovTrain.shape[0]) + "p_" + str(M) + "MCMCsamples" + "_" + debugInfo)
            assert(False)
            
        
        m = multiprocessing.Manager()
        allRuntimesQueue = m.Queue()
        
        dataForProcessingEachClustering = []
        for clusteringId in xrange(len(allCollectedClusterings)):
            
            # only for testing:
            # if clusteringId != 10:
            #     continue
            
            clusterAssignments = allCollectedClusterings[clusteringId]
            allPassedData = (allRuntimesQueue, PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sampledCovTrain, SOLVER, beta, df, approximationMethod, M, clusterAssignments, clusteringId, estimatorType, kappa)
            dataForProcessingEachClustering.append(allPassedData)
        
        try:
            print "NUMBER_OF_CPUS = ", NUMBER_OF_CPUS
            pool = multiprocessing.Pool(processes=NUMBER_OF_CPUS)
            allMarginalLikelihoodsProposedMethod = pool.map(calculateMarginalLikelihoodProposed, dataForProcessingEachClustering)
        except (KeyboardInterrupt):
            print "main process exiting.."
            pool.terminate()
            pool.join()
            assert(False)
            

        allMarginalLikelihoodsProposedMethod = numpy.asarray(allMarginalLikelihoodsProposedMethod)
        bestId = numpy.argmax(allMarginalLikelihoodsProposedMethod)
        
        bestNegMarginalProp = -1.0 * allMarginalLikelihoodsProposedMethod[bestId]
        bestClusteringNegMarginalProp = numpy.copy(allCollectedClusterings[bestId])
        
        allRuntimes = numpy.zeros(len(allCollectedClusterings))
        for clusteringId in xrange(len(allCollectedClusterings)):
            allRuntimes[clusteringId] = allRuntimesQueue.get()
            assert(allRuntimes[clusteringId] > 0.0)

        avgRuntimeProposed = numpy.mean(allRuntimes)
        
    
    allMarginalLikelihoodsBaseline = numpy.zeros(len(allCollectedClusterings))
    
    for i, clusterAssignments in enumerate(allCollectedClusterings):
        
        currentModelNegMarginal = marginalHelper.getNegLogMarginalLikelihoodNew(sampledCovTrain, n, clusterAssignments, df)
        allMarginalLikelihoodsBaseline[i] = -1.0 * currentModelNegMarginal
        if currentModelNegMarginal <= bestNegMarginal:
            bestNegMarginal = currentModelNegMarginal
            bestClusteringNegMarginal = numpy.copy(clusterAssignments)
        
        currentModelNegMarginalWT = -1.0 # marginalHelper.getNegLogMarginalLikelihoodWithThresholding(sampledCovTrain, n, clusterAssignments, 1.0, uniformNoiseStdOnPrecForThresholdingMethod)
        if currentModelNegMarginalWT <= bestNegMarginalWT:
            bestNegMarginalWT = currentModelNegMarginalWT
            bestClusteringNegMarginalWT = numpy.copy(clusterAssignments)
        
        currentModelAIC = getAIC(sampledCovTrain, n, clusterAssignments)
        if currentModelAIC <= bestAIC:
            bestAIC = currentModelAIC
            bestClusteringAIC = numpy.copy(clusterAssignments)
        
        smoothingParam = 0.001 # for numerical stability
        sampledCovTrainInv = numpy.linalg.inv(sampledCovTrain + smoothingParam * numpy.eye(sampledCovTrain.shape[0]))
        
        currentSil = getSilhouetteScore(sampledCovTrainInv, clusterAssignments)
        if currentSil <= bestSil:
            bestSil = currentSil
            bestClusteringSil = numpy.copy(clusterAssignments)
             
        currentCHI = getCalinskiHarabazIndex(sampledCovTrainInv, clusterAssignments)
        if currentCHI <= bestCHI:
            bestCHI = currentCHI
            bestClusteringCHI = numpy.copy(clusterAssignments)    
                     
        currentModelEBIC = getEBIC(sampledCovTrain, n, clusterAssignments, 0.0)
        allScoresBIC[i] = currentModelEBIC
        if currentModelEBIC <= bestEBIC:
            bestEBIC = currentModelEBIC
            bestClusteringEBIC = numpy.copy(clusterAssignments)
                
        currentModelEBIC05 = getEBIC(sampledCovTrain, n, clusterAssignments, 0.5)
        if currentModelEBIC05 <= bestEBIC05:
            bestEBIC05 = currentModelEBIC05
            bestClusteringEBIC05 = numpy.copy(clusterAssignments)
            
        currentModelEBIC1 = getEBIC(sampledCovTrain, n, clusterAssignments, 1.0)
        if currentModelEBIC1 <= bestEBIC1:
            bestEBIC1 = currentModelEBIC1
            bestClusteringEBIC1 = numpy.copy(clusterAssignments)
        
        _, ANMI, _ = clusteringHelper.getRSandNMIwithAdjustment(hiddenVarIds, clusterAssignments)
        allANMIs[i] = ANMI
        allRS[i] = clusteringHelper.rand_score(hiddenVarIds, clusterAssignments)
        
            
    
    # bestFlattendId = numpy.argmin(allEBICs)
    # bestIdPair = numpy.unravel_index(bestFlattendId, dims = (nrLambdaCandidates, nrClusterNrCandidates))
    # bestLambda = allLambdaValues[bestIdPair[0]]
    # bestClusterNr = allClustersCandidatesNrs[bestIdPair[1]]
    # allEBICsFlat = allEBICs.flatten()
    # allANMIsFlat = allANMIs.flatten()
    # bicANMIcorr = idcHelper.getCorrelation(allEBICsFlat, allANMIsFlat)
    
    bestANMI = numpy.max(allANMIs)
    bestRS = numpy.max(allRS)
    
    return bestNegMarginalProp, bestClusteringEBIC, bestClusteringNegMarginal, bestClusteringNegMarginalWT, bestClusteringNegMarginalProp, bestClusteringEBIC05, bestClusteringEBIC1, bestClusteringAIC, bestClusteringCHI, bestClusteringSil,bestANMI, bestRS, allCollectedClusterings, allMarginalLikelihoodsProposedMethod, allMarginalLikelihoodsBaseline, avgRuntimeProposed


