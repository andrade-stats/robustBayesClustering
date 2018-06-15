
import numpy
import marginalHelper
import marginalLikelihood_approx
import betaModel
import scipy.stats
import idcHelper
import scipy.misc
import time

class ChibEstimator:
    
    def initialize(self,  kappa):
        
        self.samplingDFparameters = []
        self.scaleParametersMatricies = []
        
        samplingDF = self.beta * kappa * self.n + self.p + 1 + self.df
        self.samplingDFparameters.append(samplingDF)
        self.scaleParametersMatricies.append(numpy.linalg.inv(self.noisePrecisionMode) * (samplingDF + self.p + 1))
    
        for clusterId in xrange(self.numberOfClusters):
            clusterPrecisionMode = self.allClusterPrecisionModes[clusterId]
            clusterSize = clusterPrecisionMode.shape[0]
            samplingDF = (1.0 - self.beta) * self.n * kappa + clusterSize + 1 + self.df
            self.samplingDFparameters.append(samplingDF)
            self.scaleParametersMatricies.append(numpy.linalg.inv(clusterPrecisionMode) * (samplingDF + clusterSize + 1))
        
        return
    
    # checked
    def __init__(self, PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, originalSampleCovariance, clustering, beta, df, kappa, debugInfo):
        assert(beta >= 0.0)
        assert(kappa >= 1.0)
        
        sortedSampleCovariance, allClusterSizes, sortedClustering = marginalLikelihood_approx.getPermutedCovarianceMatrixAndClusterSizes(originalSampleCovariance, clustering)
        noisePrecisionMode, allClusterPrecisionModes, objValue = betaModel.findPosteriorMode3Block(PRIOR_SCALE_FOR_CLUSTER, PRIOR_SCALE_FOR_NOISE, n, sortedSampleCovariance, allClusterSizes, beta, df)
    
        self.noisePrecisionMode = noisePrecisionMode
        self.allClusterPrecisionModes = allClusterPrecisionModes
        
        self.PRIOR_SCALE_FOR_CLUSTER = PRIOR_SCALE_FOR_CLUSTER
        self.PRIOR_SCALE_FOR_NOISE = PRIOR_SCALE_FOR_NOISE
        self.n = n
        self.sortedSampleCovariance = sortedSampleCovariance
        self.beta = beta
        self.df = df
        self.p = sortedSampleCovariance.shape[0]
        self.numberOfClusters = len(allClusterPrecisionModes)
        self.debugInfo = debugInfo
        
        self.initialize(kappa)
        return
    
    
    
    # checked
    # k = 0: means noise matrix
    # k > 0: indiciates the cluster id
    def proposeCovarianceMatrix(self, matrixId):
        sampledCovarianceMatrix =  scipy.stats.invwishart.rvs(self.samplingDFparameters[matrixId], self.scaleParametersMatricies[matrixId], size=1)
        return numpy.asmatrix(sampledCovarianceMatrix)
    
    # checked
    def proposalLogProbCov(self, proposedCovarianceMatrix, matrixId):
        return idcHelper.getInverseWishartLogPDF(self.samplingDFparameters[matrixId], self.scaleParametersMatricies[matrixId], proposedCovarianceMatrix)
        
    def getJointLogProb(self, noisePrecision, allClusterPrecisions, newPrecisionMatrix, matrixId):
        
        if matrixId == 0:
            fixedNoisePrecision = newPrecisionMatrix
            fixedAllClusterPrecisions = list(allClusterPrecisions)
        else:
            fixedNoisePrecision = noisePrecision
            fixedAllClusterPrecisions = list(allClusterPrecisions)
            fixedAllClusterPrecisions[matrixId - 1] = newPrecisionMatrix
        
        return marginalHelper.getJointLogProb(self.PRIOR_SCALE_FOR_CLUSTER, self.PRIOR_SCALE_FOR_NOISE, self.n, self.sortedSampleCovariance, fixedNoisePrecision, fixedAllClusterPrecisions, self.beta, self.df)
        
        
    # checked
    def getAcceptanceProb(self, allPrecisionMatricies, proposedPrecisionMatrix, currentPrecisionMatrix, matrixId):
        
        noisePrecision = allPrecisionMatricies[0]
        allClusterPrecisions = allPrecisionMatricies[1:len(allPrecisionMatricies)]
        
        logNominator = self.getJointLogProb(noisePrecision, allClusterPrecisions, proposedPrecisionMatrix, matrixId)
        logNominator += self.proposalLogProbCov(numpy.linalg.inv(currentPrecisionMatrix), matrixId)
        
        logDenominator = self.getJointLogProb(noisePrecision, allClusterPrecisions, currentPrecisionMatrix, matrixId)
        logDenominator += self.proposalLogProbCov(numpy.linalg.inv(proposedPrecisionMatrix), matrixId)
        
        ratio = logNominator - logDenominator
        
        if ratio > 0.0:
            return 1.0
        else:
            return numpy.exp(ratio)
        
    
    # checked
    def getMHsampleFromConditional(self, allPrecisionMatricies, matrixId):
        
        burnInIt = 0
        sampleSize = 1
        currentPrecisionMatrix = numpy.linalg.inv(self.proposeCovarianceMatrix(matrixId))
        
        samples = []
        
        acceptedStatesBeforeBurnIn = 0
        acceptedStatesAfterBurnIn = 0
        
        for i in xrange(burnInIt + sampleSize):
            proposedPrecisionMatrix = numpy.linalg.inv(self.proposeCovarianceMatrix(matrixId))
        
            alpha = self.getAcceptanceProb(allPrecisionMatricies, proposedPrecisionMatrix, currentPrecisionMatrix, matrixId)
            
            if numpy.random.uniform() < alpha:
                # print "accept"
                currentPrecisionMatrix = numpy.copy(proposedPrecisionMatrix)
                
                if i >= burnInIt:
                    acceptedStatesAfterBurnIn += 1
                else:
                    acceptedStatesBeforeBurnIn += 1
                
            if i >= burnInIt:
                samples.append(currentPrecisionMatrix)
        
        assert(len(samples) == sampleSize)
        assert(acceptedStatesAfterBurnIn == 1 or acceptedStatesAfterBurnIn == 0)
        
        return samples[0], acceptedStatesAfterBurnIn    
        
    
    
#     def estimateLogMarginalWithRestart(self, M, BURN_IN_ITERATIONS):
#         returnValue = self.estimateLogMarginalOriginal(M, BURN_IN_ITERATIONS)
#        
#         if returnValue is not None:
#             return returnValue
#         else:
#             # restart with a higher kappa
#             kappa = 10.0
#             print "RESTART WITH NEW KAPPA IS NECESSARY, NEW KAPPA = ", kappa
#             self.initialize(kappa)
#             returnValue = self.estimateLogMarginalOriginal(M, BURN_IN_ITERATIONS)
#             if returnValue is not None:
#                 return returnValue
#             else:
#                 print "CANNOT RECOVER ANYMORE: FAILED AGAIN WITH KAPPA = ", kappa
#                 assert(False)
#                 return None
            
    
    # double checked
    # M = number of samples for nominator and denominator
    def estimateLogMarginal(self, M, BURN_IN_ITERATIONS):
        
        # self.ro = marginalHelper.setupR()
        
        print "M = ", M
        print "BURN_IN_ITERATIONS = ", BURN_IN_ITERATIONS
        
        startTimeAllMCMC = time.time()
        
        nrBlocks = self.numberOfClusters + 1
        
        allPrecisionModes = []
        allPrecisionModes.append(self.noisePrecisionMode)
        allPrecisionModes.extend(self.allClusterPrecisionModes)
        
        allCurrentPrecisionMatricies = []
        for matrixId in xrange(nrBlocks): 
            allCurrentPrecisionMatricies.append( numpy.linalg.inv(self.proposeCovarianceMatrix(matrixId)) )
        
        samplesFromPosteriorConditionedOnModes = []
              
        for conditionedBeforeMatrixId in xrange(nrBlocks):
            allCurrentPrecisionMatricies[0:conditionedBeforeMatrixId] = allPrecisionModes[0:conditionedBeforeMatrixId]
            collectedPosteriorSamples = []
            
            totalAcceptanceCount = 0.0
            totalIterationsAfterBurnIn = 0
        
            for it in xrange(BURN_IN_ITERATIONS + M):
                for matrixId in xrange(conditionedBeforeMatrixId, nrBlocks, 1):
                    newPrecisionMatrix, acceptedCount = self.getMHsampleFromConditional(allCurrentPrecisionMatricies, matrixId)
                    allCurrentPrecisionMatricies[matrixId] = newPrecisionMatrix
                    if it >= BURN_IN_ITERATIONS:
                        totalAcceptanceCount += acceptedCount
                        totalIterationsAfterBurnIn += 1
                if it >= BURN_IN_ITERATIONS:
                    # conditionedBeforeMatrixId corresponds to i in paper
                    # we now have a sample from p(theta_{>= i} | X, theta_{< i})
                    sampledPrecisionMatrices = allCurrentPrecisionMatricies[conditionedBeforeMatrixId:nrBlocks]
                    collectedPosteriorSamples.append(sampledPrecisionMatrices)
            
            # assert(totalIterationsAfterBurnIn == M)
            avgAcceptanceRate = float(totalAcceptanceCount) / float(totalIterationsAfterBurnIn)
            print "Average acceptance rate = ", avgAcceptanceRate
          
            assert(len(collectedPosteriorSamples) == M)
            samplesFromPosteriorConditionedOnModes.append(collectedPosteriorSamples)
        
        
        print "Finished: Got all samples from posterior."
        
        logMarginal = 0.0
        
        for conditionedBeforeMatrixId in xrange(nrBlocks):
            
            # conditionedBeforeMatrixId corresponds to i in paper
            # retrieve all samples from p(theta_{>= i} | X, theta_{< i})
            samplesFromPosteriorConditionedOnThetaBeforei = samplesFromPosteriorConditionedOnModes[conditionedBeforeMatrixId]
            
            assert(len(samplesFromPosteriorConditionedOnThetaBeforei) == M)
            assert(len(samplesFromPosteriorConditionedOnThetaBeforei[0]) == nrBlocks - conditionedBeforeMatrixId)
            
            allLogTermsInNominator = numpy.zeros(M)
            
            for m in xrange(M):
                allPrecisionMatricies = list(allPrecisionModes)
                allPrecisionMatricies[conditionedBeforeMatrixId:nrBlocks] = samplesFromPosteriorConditionedOnThetaBeforei[m]
                
                proposedPrecisionMatrix = allPrecisionModes[conditionedBeforeMatrixId]
                currentPrecisionMatrix = allPrecisionMatricies[conditionedBeforeMatrixId]
                alpha = self.getAcceptanceProb(allPrecisionMatricies, proposedPrecisionMatrix, currentPrecisionMatrix, conditionedBeforeMatrixId)
                assert(alpha >= 0.0 and alpha <= 1.0)
                allLogTermsInNominator[m] = numpy.log(alpha) + self.proposalLogProbCov(numpy.linalg.inv(allPrecisionModes[conditionedBeforeMatrixId]), conditionedBeforeMatrixId)
                
            logNominatorEstimate = scipy.misc.logsumexp(allLogTermsInNominator) - numpy.log(M)
            if (scipy.misc.logsumexp(logNominatorEstimate) == -numpy.inf):
                print "ALL ENTRIES ARE -inf in logNominatorEstimate !!!"
                print "conditionedBeforeMatrixId = ", conditionedBeforeMatrixId
                print "self.debugInfo = ", self.debugInfo
            assert(scipy.misc.logsumexp(logNominatorEstimate) > -numpy.inf)
            
            logMarginal -= logNominatorEstimate
            
            
            allLogTermsInDenominator = numpy.zeros(M)
            
            for m in xrange(M):
                allPrecisionMatricies = list(allPrecisionModes)
                
                if conditionedBeforeMatrixId < nrBlocks - 1:
                    # conditionedBeforeMatrixId corresponds to i in paper
                    # retrieve all samples from p(theta_{>= i+1} | X, theta_{< i+1})
                    samplesFromPosteriorConditionedOnThetaBeforeOrEquali = samplesFromPosteriorConditionedOnModes[conditionedBeforeMatrixId + 1]
                    assert(len(allPrecisionMatricies[(conditionedBeforeMatrixId+1):nrBlocks]) == len(samplesFromPosteriorConditionedOnThetaBeforeOrEquali[m]))
                    allPrecisionMatricies[(conditionedBeforeMatrixId+1):nrBlocks] = samplesFromPosteriorConditionedOnThetaBeforeOrEquali[m]
                
                proposedPrecisionMatrix = numpy.linalg.inv(self.proposeCovarianceMatrix(conditionedBeforeMatrixId))
                currentPrecisionMatrix = allPrecisionModes[conditionedBeforeMatrixId]
                alpha = self.getAcceptanceProb(allPrecisionMatricies, proposedPrecisionMatrix, currentPrecisionMatrix, conditionedBeforeMatrixId)
                assert(alpha >= 0.0 and alpha <= 1.0)
                allLogTermsInDenominator[m] = numpy.log(alpha)
            
            if (scipy.misc.logsumexp(allLogTermsInDenominator) == -numpy.inf):
                print "ALL ENTRIES ARE -inf in allLogTermsInDenominator !!!"
                print "conditionedBeforeMatrixId = ", conditionedBeforeMatrixId
                print "self.debugInfo = ", self.debugInfo
            assert(scipy.misc.logsumexp(allLogTermsInDenominator) > -numpy.inf)
            logDenominatorEstimate = scipy.misc.logsumexp(allLogTermsInDenominator) - numpy.log(M)
            logMarginal += logDenominatorEstimate
            
        
        logMarginal += marginalHelper.getJointLogProb(self.PRIOR_SCALE_FOR_CLUSTER, self.PRIOR_SCALE_FOR_NOISE, self.n, self.sortedSampleCovariance, self.noisePrecisionMode, self.allClusterPrecisionModes, self.beta, self.df)
        
        print "logMarginal = ", logMarginal
        assert(logMarginal > -numpy.inf and numpy.exp(logMarginal) <= 1.0)
        print "Runtime in minutes = ", (time.time() - startTimeAllMCMC) / 60.0
        
        return logMarginal
    
    
    # overdispersed
    def proposeCovarianceMatrixOverdispersed(self, matrixId):
        p = self.scaleParametersMatricies[matrixId].shape[0]
        sampledCovarianceMatrix =  scipy.stats.invwishart.rvs(p+1, numpy.eye(p), size=1)
        return numpy.asmatrix(sampledCovarianceMatrix)
    
    
    
    
    # helper method for "estimateLogMarginalDiagnosis"
    @staticmethod
    def getAsOneVector(allSymMatrices):
        
        allInOneVec = numpy.zeros(0)
        
        for symMatrix in allSymMatrices:
            upperTriangularIndicies = numpy.triu_indices(symMatrix.shape[0])
            allInOneVec = numpy.append(allInOneVec, symMatrix[upperTriangularIndicies])
        
        return allInOneVec
            
    # to test with gelman-rubin diagnositic 
    def estimateLogMarginalDiagnosis(self, M, BURN_IN_ITERATIONS, infoStr):
        
        print "M = ", M
        print "BURN_IN_ITERATIONS = ", BURN_IN_ITERATIONS
        
        nrBlocks = self.numberOfClusters + 1
        
        allPrecisionModes = []
        allPrecisionModes.append(self.noisePrecisionMode)
        allPrecisionModes.extend(self.allClusterPrecisionModes)
        
        NUMBER_OF_CHAINS = 2
        for chainId in xrange(NUMBER_OF_CHAINS):
        
            print "RUN CHAIN", chainId
        
            vec = ChibEstimator.getAsOneVector(allPrecisionModes)
            print vec.shape
            parameterDim = ChibEstimator.getAsOneVector(allPrecisionModes).shape[0]
            print "parameterDim = ", parameterDim
            collectedSamplesOneChain = numpy.zeros(shape = (M,  parameterDim))
            
            allCurrentPrecisionMatricies = []
            for matrixId in xrange(nrBlocks): 
                allCurrentPrecisionMatricies.append( numpy.linalg.inv(self.proposeCovarianceMatrix(matrixId)) )
            
            
            totalAcceptanceCount = 0.0
            totalIterationsAfterBurnIn = 0
        
            sampleId = 0
            for it in xrange(BURN_IN_ITERATIONS + M):
                for matrixId in xrange(0, nrBlocks, 1):
                    newPrecisionMatrix, acceptedCount = self.getMHsampleFromConditional(allCurrentPrecisionMatricies, matrixId)
                    allCurrentPrecisionMatricies[matrixId] = newPrecisionMatrix
                    if it >= BURN_IN_ITERATIONS:
                        totalAcceptanceCount += acceptedCount
                        totalIterationsAfterBurnIn += 1
                if it >= BURN_IN_ITERATIONS:
                    # we now have a sample from p(theta_1, ...,theta_{number of clusters + 1}  | X)
                    sampledPrecisionMatrices = allCurrentPrecisionMatricies[0:nrBlocks]
                    collectedSamplesOneChain[sampleId] = ChibEstimator.getAsOneVector(sampledPrecisionMatrices)
                    sampleId += 1
                
            assert(sampleId == M)  
            
            avgAcceptanceRate = float(totalAcceptanceCount) / float(totalIterationsAfterBurnIn)
            print "Average acceptance rate = ", avgAcceptanceRate
            
            #/Users/danielandrade/workspace/StanTest/
            # numpy.savetxt("/Users/danielandrade/workspace/StanTest/samples/collectedSamplesForOutput_" + infoStr + "_" + str(chainId) + "chainId", collectedSamplesOneChain, delimiter=',')
            numpy.savetxt("/home/daniel/workspace/StanTest/samples/collectedSamplesForOutput_" + infoStr + "_" + str(chainId) + "chainId", collectedSamplesOneChain, delimiter=',')
        
        print "Saved diagnosis samples from all chains, info = ", infoStr
        print "FINISHED DIAGNOSIS SUCCESSFULLY"
        return
    
        
#     # to test with gelman diagnosis 
#     def estimateLogMarginalDiagnosisOld(self, M, BURN_IN_ITERATIONS):
#         
#         startTimeAllMCMC = time.time()
#         
#         nrBlocks = self.numberOfClusters + 1
#         
#         allPrecisionModes = []
#         allPrecisionModes.append(self.noisePrecisionMode)
#         allPrecisionModes.extend(self.allClusterPrecisionModes)
#         
#         
#         NUMBER_OF_CHAINS = 2
#         
#         # SAVE_MATRIX_ID = 0 means noise covariance matrix
#         # SAVE_MATRIX_ID = 1 means first block of cluster covariance matrix
#         
#         for SAVE_MATRIX_ID in [0,1]:
#             upperTriangularIndicies = numpy.triu_indices(allPrecisionModes[SAVE_MATRIX_ID].shape[0])
#             parameterDim = upperTriangularIndicies[0].shape[0]
#             
#             
#             for chainId in xrange(NUMBER_OF_CHAINS):
#                 
#                 collectedSamplesOneChain = numpy.zeros(shape = (M,  parameterDim))
#                 
#                 allCurrentPrecisionMatricies = []
#                 for matrixId in xrange(nrBlocks): 
#                     allCurrentPrecisionMatricies.append( numpy.linalg.inv(self.proposeCovarianceMatrixOverdispersed(matrixId)) )
#                 
#                 
#                 totalAcceptanceCount = 0.0
#                 totalIterationsAfterBurnIn = 0.0
#                 
#                 allAcceptanceCounts = numpy.zeros(nrBlocks)
#                 
#                 sampleId = 0
#                 
#                 for conditionedBeforeMatrixId in xrange(1):
#                     allCurrentPrecisionMatricies[0:conditionedBeforeMatrixId] = allPrecisionModes[0:conditionedBeforeMatrixId]
#                     
#                     i = 0
#                     while(sampleId < M):
#                         # print "iteration ", i
#                         for matrixId in xrange(conditionedBeforeMatrixId, nrBlocks, 1):
#                             newPrecisionMatrix, acceptedCount = self.getMHsampleFromConditional(allCurrentPrecisionMatricies, matrixId)
#                             allCurrentPrecisionMatricies[matrixId] = newPrecisionMatrix
#                             if i >= BURN_IN_ITERATIONS:
#                                 allAcceptanceCounts[matrixId] += acceptedCount
#                                 totalAcceptanceCount += acceptedCount
#                                 totalIterationsAfterBurnIn += 1.0
#                         if i >= BURN_IN_ITERATIONS:
#                             collectedSamplesOneChain[sampleId] = allCurrentPrecisionMatricies[SAVE_MATRIX_ID][upperTriangularIndicies]
#                             sampleId += 1
#                         i += 1
#                     
#                     
#                 numpy.savetxt("../../samples/collectedSamplesForOutput_" + str(self.p) +"p_" + str(SAVE_MATRIX_ID) + "matrixId" + "_" + str(chainId) + "chainId", collectedSamplesOneChain, delimiter=',')
#             
#               
#             avgAcceptanceRate = float(totalAcceptanceCount) / float(totalIterationsAfterBurnIn)
#             print "Runtime in minutes = ", (time.time() - startTimeAllMCMC) / 60.0
#             print "Average acceptance rate = ", avgAcceptanceRate
#             print "allAcceptanceCounts = "
#             print allAcceptanceCounts
#         return
        