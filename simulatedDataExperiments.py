import numpy
import time
import sys

import idcHelper
import shared.clusteringHelper as clusteringHelper

import baselines
import simulation
import EBIC

# Arguments
# 1: number of clusters
# 2: number of variables
# 3: "balanced" or "unbalanced" clusters
# 4: "invWishart" or "uniform" distribution of clusters
# 5: noise level >= 0
# 6: "variational" or "MCMC" approximation
# 7: beta 
# 8: number of cpus

# Example:
# /opt/intel/intelpython2/bin/python -m simulatedDataExperiments 4 40 balanced uniform 0.0 variational 0.02 30


# MCMC parameters:
NUMBER_OF_MCMC_SAMPLES = 10000
KAPPA = 10.0 # recommended 10.0
MCMCestimatorType = "ChibEstimator" # should be "ChibEstimator"


USE_PROPOSED_MARGINAL = True  # should be "True"
INCLUDES_ALL_IN_ONE = False  # should be "False"
ONLY_TRUE_CLUSTERING = False # should be "False"
ADD_TRUE_CLUSTERING = False # should be "False"
SPECTRAL_CLUSTERING_VARIANT = "noNormSC" # should be "noNormSC"
minNrVariables = "all" # should be "all"
lambdaRange = "allLambda" # should be "allLambda"
df = 0.0 # should be "0.0"
SOLVER = "ADMM-3BLOCK" # recommended
PRIOR_SCALE_FOR_NOISE = 1.0 # recommened 1.0
PRIOR_SCALE_FOR_CLUSTER = 1.0 # recommened 1.0
methodName = "allL1ExceptDiagonalADMM" # recommended
clusteringMethod = "spectral"

NUMBER_OF_REPETIONS = 5 # is set to 5 in our experiment

assert(len(sys.argv) == 9)
NUMBER_OF_CLUSTERS = int(sys.argv[1])
NUMBER_OF_VARIABLES = int(sys.argv[2])
clusterSizeDist = sys.argv[3]
sampleType = sys.argv[4]
assert(clusterSizeDist == "balanced" or clusterSizeDist == "unbalanced")
assert(sampleType == "uniform" or sampleType == "invWishart")

noiseLevel = float(sys.argv[5])

if noiseLevel == 0.0:
    addNoiseToWhat = "noNoise"
    noiseType = None
else:
    addNoiseToWhat = "prec"
    noiseType = sampleType

approximationMethod = sys.argv[6]
beta = float(sys.argv[7])
NUMBER_OF_CPUS = int(sys.argv[8])

assert(approximationMethod == "variational" or approximationMethod == "MCMC")
assert(beta >= 0.0 and beta <= 1.0)
assert(NUMBER_OF_CPUS >= 1)
 
if NUMBER_OF_VARIABLES == 12:
    allSampleCounts = [12, 120, 1200, 1200000]
elif NUMBER_OF_VARIABLES == 40:
    allSampleCounts = [20, 40, 400, 4000, 40000, 4000000]
elif NUMBER_OF_VARIABLES == 100:
    allSampleCounts = [50, 100, 1000, 1000000]
elif NUMBER_OF_VARIABLES == 120:
    allSampleCounts = [60, 120, 1200, 12000, 1200000]
elif NUMBER_OF_VARIABLES == 400:
    allSampleCounts = [200, 400, 4000, 40000, 2000000]
else:
    assert(False)

debugInfo = sampleType
print "sampleType = ", sampleType
print "NUMBER_OF_CPUS = ", NUMBER_OF_CPUS


collectedStatsANMIs_Sil = []
collectedStatsANMIs_CHI = []
collectedStatsANMIs_AIC = []
collectedStatsANMIs_NegMarginal = []
collectedStatsANMIs_NegMarginalWT = []
collectedStatsANMIs_NegMarginalProp = []
collectedStatsANMIs_EBIC05 = []
collectedStatsANMIs_EBIC1 = []
collectedStatsANMIs = []


collectedStatsRS_Sil = []
collectedStatsRS_CHI = []
collectedStatsRS_AIC = []
collectedStatsRS_NegMarginal = []
collectedStatsRS_NegMarginalWT = []
collectedStatsRS_NegMarginalProp = []
collectedStatsRS_EBIC05 = []
collectedStatsRS_EBIC1 = []
collectedStatsRS = []

collectedClusterNrsWithHomogenity = []
collectedLambdas = []
collectedBIC_selectedModel = []
collectedBIC_correctModel = []

collectedBIC_bestANMImodel = []
collectedAllBicANMIcorrs = []
collectedAllHypothesesSizes = []
collectedAllBestANMI = []
collectedAllBestRS = []

collectedStatsRS_CGL = []
collectedStatsANMIs_CGL = []

collectedStatsRS_DPVC = []
collectedStatsANMIs_DPVC = []

bestNegMarginalProp = None 

totalAvgRuntimeProposed = 0.0
totalIterations = 0.0

print "KAPPA = ", KAPPA

for currentOuterIteration in xrange(len(allSampleCounts)):
    
    RANDOM_GENERATOR_SEED = 9899832
    numpy.random.seed(RANDOM_GENERATOR_SEED)
    
    
    print "NUMBER_OF_CLUSTERS = ", NUMBER_OF_CLUSTERS
    print "NUMBER_OF_VARIABLES = ", NUMBER_OF_VARIABLES

    NUMBER_OF_SAMPLES = allSampleCounts[currentOuterIteration]
    
    assert(NUMBER_OF_VARIABLES >= 4)
    assert(NUMBER_OF_SAMPLES >= 10)
    assert(NUMBER_OF_CLUSTERS >= 2)
    
    if NUMBER_OF_VARIABLES == 12:
        allClustersCandidatesNrs = numpy.asarray([2,3,4,5,6,7,8,9,10,11])
    else:
        allClustersCandidatesNrs = numpy.asarray([2,3,4,5,6,7,8,9,10,11,12,13,14,15])

    
    print "METHOD DETAILS:"
    print "methodName = ", methodName
    print "SPECTRAL_CLUSTERING_VARIANT = ", SPECTRAL_CLUSTERING_VARIANT 
    print "FULL DATA PROPERTIES: "
    print "clusterSizeDist = ", clusterSizeDist
    print "sampleType = ", sampleType
    
    resultsRS_CGL  = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsNMI_CGL  = numpy.zeros(NUMBER_OF_REPETIONS)
    
    resultsNMI_Sil = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsNMI_CHI = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsNMI_AIC = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsNMI_NegMarginal = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsNMI_NegMarginalWT = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsNMI_NegMarginalProp = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsNMI_EBIC05 = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsNMI_EBIC1 = numpy.zeros(NUMBER_OF_REPETIONS)
    
    resultsRS_Sil = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsRS_CHI = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsRS_AIC = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsRS_NegMarginal = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsRS_NegMarginalWT = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsRS_NegMarginalProp = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsRS_EBIC05 = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsRS_EBIC1 = numpy.zeros(NUMBER_OF_REPETIONS)
    
    resultsRS = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsNMI = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsHom = numpy.zeros(NUMBER_OF_REPETIONS)
    allBestLambdas = numpy.zeros(NUMBER_OF_REPETIONS)
    allBestNrClusters = numpy.zeros(NUMBER_OF_REPETIONS)
    allRuntimes = numpy.zeros(NUMBER_OF_REPETIONS)
    allRuntimesInSeconds = numpy.zeros(NUMBER_OF_REPETIONS)
    
    allSelectedModelEBICs = numpy.zeros(NUMBER_OF_REPETIONS)
    allCorrectModelEBICs = numpy.zeros(NUMBER_OF_REPETIONS)
    allFoundCorrectClusteringCounts = numpy.zeros(NUMBER_OF_REPETIONS)
    
    allHypothesesSizes = numpy.zeros(NUMBER_OF_REPETIONS)
    allBestANMI = numpy.zeros(NUMBER_OF_REPETIONS)
    allBestRS = numpy.zeros(NUMBER_OF_REPETIONS)
    
    resultsRS_DPVC = numpy.zeros(NUMBER_OF_REPETIONS)
    resultsNMI_DPVC = numpy.zeros(NUMBER_OF_REPETIONS)
    
    startTime = time.time()
    
    for iterationId in xrange(NUMBER_OF_REPETIONS):
        
        print "*************** ITERATION = " + str(iterationId) + " ******************************"
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = simulation.createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, clusterSizeDist, sampleType, addNoiseToWhat, noiseType, noiseLevel)
        assert(dataVectorsAllOriginal.shape[1] == len(hiddenVarIds))
        
        startTimeOneIt = time.time()
        
        dataVectorsAllTrain = dataVectorsAllOriginal
        sampledCovAllTrain = numpy.asmatrix(numpy.cov(dataVectorsAllTrain.transpose(), rowvar=True, bias=True))
        
        
        if methodName == "SLC" or methodName == "ALC":
            linkageType = None
            if methodName == "SLC":
                linkageType = "single"
            else: 
                linkageType = "average"
            
            # apply method in "The cluster graphical lasso for improved estimation of Gaussian graphical models", 2015
            bestClusterNrCGL = baselines.mseClusterEval(sampledCovAllTrain, allClustersCandidatesNrs, linkageType)
            clusterAssignmentsCGL = baselines.hierarchicalClustering(sampledCovAllTrain, bestClusterNrCGL, linkageType)
            resultsRS_CGL[iterationId], resultsNMI_CGL[iterationId], _ = clusteringHelper.getRSandNMIwithAdjustment(hiddenVarIds, clusterAssignmentsCGL)
            continue
        
        else:
            
            if lambdaRange == "allLambda":
                
                if len(hiddenVarIds) <= 100:
                    allLambdaValues = [0.0001, 0.0005]
                    for i in xrange(10):
                        allLambdaValues += [0.001 * (i + 1)]
                else:
                    allLambdaValues = [0.01, 0.05]
                    for i in xrange(10):
                        allLambdaValues += [0.1 * (i + 1)]
                    
                # print "allLambdaValues = ", allLambdaValues
                # assert(False)
                
            elif lambdaRange == "allLambdaExtended":
                allLambdaValues = [0.005, 0.006, 0.007, 0.008, 0.009]
                for i in xrange(100):
                    allLambdaValues += [0.01 * (i + 1)]
                for i in xrange(10):
                    allLambdaValues += [1.0 + 0.1 * (i + 1)]
                    
                # print "allLambdaValues = "
                # print allLambdaValues
                # assert(False)
                
            else:
                assert(False)
                
            allLambdaValues.reverse()
            
            
            n = dataVectorsAllTrain.shape[0]
            p = dataVectorsAllTrain.shape[1]
            
            assert(lambdaRange == "allLambda" or lambdaRange == "allLambdaExtended")
            assert(SOLVER == "ADMM-3BLOCK" or SOLVER == "ADMM-2BLOCK" or SOLVER == "proposed-EXACT" or SOLVER == "proposed-CG" or SOLVER == "CVXPY"  or SOLVER == "proposedWithBFGS_dual" or SOLVER == "compareCVXPYandProposed")
            assert(minNrVariables == "all")
            assert(PRIOR_SCALE_FOR_CLUSTER == 1.0)
            assert(beta >= -1.0 and beta <= 1.0)

            bestNegMarginalProp, clusterAssignmentsEBIC, clusterAssignmentsNegMarginal, clusterAssignmentsNegMarginalWT, clusterAssignmentsNegMarginalProp, clusterAssignmentsEBIC05, clusterAssignmentsEBIC1, clusterAssignmentsAIC, clusterAssignmentsCHI, clusterAssignmentsSil, bestANMI, bestRS, allCollectedClusterings, allMarginalLikelihoodsProposedMethod, allMarginalLikelihoodsBaseline, avgRuntimeProposed = EBIC.evalWithEBIC(dataVectorsAllTrain, sampledCovAllTrain, n, p, allLambdaValues, allClustersCandidatesNrs, methodName, SPECTRAL_CLUSTERING_VARIANT, hiddenVarIds, USE_PROPOSED_MARGINAL, minNrVariables, SOLVER, debugInfo, PRIOR_SCALE_FOR_NOISE, ADD_TRUE_CLUSTERING, clusteringMethod, PRIOR_SCALE_FOR_CLUSTER, INCLUDES_ALL_IN_ONE, ONLY_TRUE_CLUSTERING, beta, df, approximationMethod, NUMBER_OF_MCMC_SAMPLES, NUMBER_OF_CPUS, MCMCestimatorType, KAPPA)
            allBestLambdas[iterationId] = -1
        
        selectedModelEBIC = EBIC.getEBIC(sampledCovAllTrain, dataVectorsAllTrain.shape[0], clusterAssignmentsEBIC, 0.0)
        correctModelEBIC = EBIC.getEBIC(sampledCovAllTrain, dataVectorsAllTrain.shape[0], hiddenVarIds, 0.0)
        
        allSelectedModelEBICs[iterationId] = selectedModelEBIC
        allCorrectModelEBICs[iterationId] = correctModelEBIC
        allHypothesesSizes[iterationId] = len(allCollectedClusterings)
        allBestANMI[iterationId] = bestANMI
        allBestRS[iterationId] = bestRS
            
        resultsRS[iterationId], resultsNMI[iterationId], resultsHom[iterationId] = clusteringHelper.getRSandNMIwithAdjustment(hiddenVarIds, clusterAssignmentsEBIC)
        resultsRS_NegMarginal[iterationId], resultsNMI_NegMarginal[iterationId], _ = clusteringHelper.getRSandNMIwithAdjustment(hiddenVarIds, clusterAssignmentsNegMarginal)
        resultsRS_NegMarginalWT[iterationId], resultsNMI_NegMarginalWT[iterationId], _ = clusteringHelper.getRSandNMIwithAdjustment(hiddenVarIds, clusterAssignmentsNegMarginalWT)
        resultsRS_NegMarginalProp[iterationId], resultsNMI_NegMarginalProp[iterationId], _ = clusteringHelper.getRSandNMIwithAdjustment(hiddenVarIds, clusterAssignmentsNegMarginalProp)
        resultsRS_EBIC05[iterationId], resultsNMI_EBIC05[iterationId], _ = clusteringHelper.getRSandNMIwithAdjustment(hiddenVarIds, clusterAssignmentsEBIC05)
        resultsRS_EBIC1[iterationId], resultsNMI_EBIC1[iterationId], _ = clusteringHelper.getRSandNMIwithAdjustment(hiddenVarIds, clusterAssignmentsEBIC1)
        resultsRS_AIC[iterationId], resultsNMI_AIC[iterationId], _ = clusteringHelper.getRSandNMIwithAdjustment(hiddenVarIds, clusterAssignmentsAIC)
        
        resultsRS_CHI[iterationId], resultsNMI_CHI[iterationId], _ = clusteringHelper.getRSandNMIwithAdjustment(hiddenVarIds, clusterAssignmentsCHI)
        resultsRS_Sil[iterationId], resultsNMI_Sil[iterationId], _ = clusteringHelper.getRSandNMIwithAdjustment(hiddenVarIds, clusterAssignmentsSil)
        
        allBestNrClusters[iterationId] = -1 
        allRuntimes[iterationId] = (time.time() - startTimeOneIt) / 60.0
        allRuntimesInSeconds[iterationId] = (time.time() - startTimeOneIt)
        
    
    collectedStatsRS_DPVC.append(idcHelper.getAvgAndStd(resultsRS_DPVC))
    collectedStatsANMIs_DPVC.append(idcHelper.getAvgAndStd(resultsNMI_DPVC))
    
    collectedStatsRS_CGL.append(idcHelper.getAvgAndStd(resultsRS_CGL))
    collectedStatsANMIs_CGL.append(idcHelper.getAvgAndStd(resultsNMI_CGL))
    
    collectedStatsRS_Sil.append(idcHelper.getAvgAndStd(resultsRS_Sil))
    collectedStatsRS_CHI.append(idcHelper.getAvgAndStd(resultsRS_CHI))
    collectedStatsRS_AIC.append(idcHelper.getAvgAndStd(resultsRS_AIC))
    
    collectedStatsRS_NegMarginal.append(idcHelper.getAvgAndStd(resultsRS_NegMarginal))
    collectedStatsRS_NegMarginalWT.append(idcHelper.getAvgAndStd(resultsRS_NegMarginalWT))
    collectedStatsRS_NegMarginalProp.append(idcHelper.getAvgAndStd(resultsRS_NegMarginalProp))
    collectedStatsRS_EBIC05.append(idcHelper.getAvgAndStd(resultsRS_EBIC05))
    collectedStatsRS_EBIC1.append(idcHelper.getAvgAndStd(resultsRS_EBIC1))
    collectedStatsRS.append(idcHelper.getAvgAndStd(resultsRS))
    
    
    collectedStatsANMIs_Sil.append(idcHelper.getAvgAndStd(resultsNMI_Sil))
    collectedStatsANMIs_CHI.append(idcHelper.getAvgAndStd(resultsNMI_CHI))
    collectedStatsANMIs_AIC.append(idcHelper.getAvgAndStd(resultsNMI_AIC))
    
    collectedStatsANMIs_NegMarginal.append(idcHelper.getAvgAndStd(resultsNMI_NegMarginal))
    collectedStatsANMIs_NegMarginalWT.append(idcHelper.getAvgAndStd(resultsNMI_NegMarginalWT))
    collectedStatsANMIs_NegMarginalProp.append(idcHelper.getAvgAndStd(resultsNMI_NegMarginalProp))
    collectedStatsANMIs_EBIC05.append(idcHelper.getAvgAndStd(resultsNMI_EBIC05))
    collectedStatsANMIs_EBIC1.append(idcHelper.getAvgAndStd(resultsNMI_EBIC1))
    collectedStatsANMIs.append(idcHelper.getAvgAndStd(resultsNMI))
    
    collectedClusterNrsWithHomogenity.append(idcHelper.getAvg(allBestNrClusters) + " (" + idcHelper.getAvg(resultsHom) + ") ")
    minLambda = str(numpy.min(allBestLambdas))
    maxLambda = str(numpy.max(allBestLambdas))
    collectedLambdas.append("[" + minLambda + "," + maxLambda + "]")
    
    collectedBIC_selectedModel.append(idcHelper.getAvg(allSelectedModelEBICs))
    collectedBIC_correctModel.append(idcHelper.getAvg(allCorrectModelEBICs))
    collectedAllHypothesesSizes.append(idcHelper.getAvgAndStd(allHypothesesSizes))
    collectedAllBestANMI.append(idcHelper.getAvgAndStd(allBestANMI))
    collectedAllBestRS.append(idcHelper.getAvgAndStd(allBestRS))
        
    duration = (time.time() - startTime) / 60.0


print ""
print "************** Evaluation Setting **************"
print "NUMBER_OF_REPETIONS = " + str(NUMBER_OF_REPETIONS)

if methodName == "allL1ExceptDiagonalADMM":
    print "minNrVariables (used by proposed method) = ", minNrVariables
    print "PRIOR_SCALE_FOR_NOISE = " + str(PRIOR_SCALE_FOR_NOISE)
    print "PRIOR_SCALE_FOR_CLUSTER = " + str(PRIOR_SCALE_FOR_CLUSTER)
    print "beta = " + str(beta)
    print "df = " + str(df)


print sampleType
print "clusterSizeDist = ", clusterSizeDist
print "NUMBER_OF_CLUSTERS = ", NUMBER_OF_CLUSTERS
print "addNoiseToWhat = ", addNoiseToWhat
if addNoiseToWhat != "noNoise":
    print "noiseType = ", noiseType
    print "noiseLevel = ", noiseLevel
print "p\t" + str(NUMBER_OF_VARIABLES)
print "n\t" + " & ".join([str(n) for n in allSampleCounts])
    
    
def getResultRep(resultsRS, resultsANMI):
    assert(len(resultsRS) == len(resultsANMI))
    return " & ".join(resultsANMI)
    
print "clusteringMethod = ", clusteringMethod
print "NUMBER_OF_CPUS = ", NUMBER_OF_CPUS

if totalIterations > 0.0:
    print "average runtime for evaluating one clustering with proposed method (min) = ", (totalAvgRuntimeProposed / float(totalIterations))

if approximationMethod == "MCMC":
    print "MCMCestimatorType = ", MCMCestimatorType
    print "NUMBER_OF_MCMC_SAMPLES = ", NUMBER_OF_MCMC_SAMPLES
    print "KAPPA = ", KAPPA
print "max number of clusters considered = ", allClustersCandidatesNrs[len(allClustersCandidatesNrs) - 1]


print ""
print "************** Evaluation Summary (ANMI scores) **************"

if methodName == "SLC" or methodName == "ALC":
    print "ARI & ANMI"
    print "CGL (" + methodName + ") & \t" + getResultRep(collectedStatsRS_CGL, collectedStatsANMIs_CGL)  + " \\\\ "
else:
    print "Proposed ($\\beta = " + str(beta) + "$) & \t" + getResultRep(collectedStatsRS_NegMarginalProp, collectedStatsANMIs_NegMarginalProp) + " \\\\ "

print "Inverse Wishart prior         & \t" + getResultRep(collectedStatsRS_NegMarginal, collectedStatsANMIs_NegMarginal) + " \\\\ "
print "EBIC ($\\gamma = 0$)      & \t" + getResultRep(collectedStatsRS, collectedStatsANMIs) + " \\\\ "
print "EBIC ($\\gamma = 0.5$)            & \t" + getResultRep(collectedStatsRS_EBIC05, collectedStatsANMIs_EBIC05) + " \\\\ "
print "EBIC ($\\gamma = 1.0$)            & \t" + getResultRep(collectedStatsRS_EBIC1, collectedStatsANMIs_EBIC1) + " \\\\ "
print "AIC & \t" + getResultRep(collectedStatsRS_AIC, collectedStatsANMIs_AIC) + " \\\\ "
print "Calinski-Harabaz Index & \t" + getResultRep(collectedStatsRS_CHI, collectedStatsANMIs_CHI) + " \\\\ "
print "\\midrule  \\\\ "
print "Oracle model selection and size of hypotheses space:"
print "\\cmidrule{2-8}"
print "\\multirow{2}{*}{" + clusteringMethod + "} " + " & ANMI " + " & \t" + getResultRep(collectedAllBestRS, collectedAllBestANMI)  + " \\\\ "
print  " & $|\mathscr{C}|$ & " + " & ".join(collectedAllHypothesesSizes) + " \\\\ "


