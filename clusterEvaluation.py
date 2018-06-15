import numpy
import sys
import shared.statHelper as statHelper
import EBIC
import idcHelper

# Arguments
# 1: data file in csv format (each row is one sample)
# 2: normalization of variables ("normalize") or not "none"
# 3: maximal number of clusters that should be considered (e.g. if 10, then consider number of clusters 2,3,4,5,...,10)
# 6: "variational" or "MCMC" approximation of the proposed method
# 7: beta (recommended value 0.02)
# 8: number of cpus

# EXAMPLE:
# python -m clusterEvaluation test.csv normalize 10 variational 0.02 4



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


assert(len(sys.argv) == 7)

FILENAME = sys.argv[1]
dataVectorsAllOriginal = numpy.genfromtxt(FILENAME, delimiter=',')

DATA_TRANSFORMATION = sys.argv[2]

MAX_NUMBER_OF_CLUSTERS =  int(sys.argv[3])
assert(MAX_NUMBER_OF_CLUSTERS >= 2)
allClustersCandidatesNrs =  numpy.arange(2,MAX_NUMBER_OF_CLUSTERS+1, 1)

approximationMethod = sys.argv[4]
beta = float(sys.argv[5])
NUMBER_OF_CPUS = int(sys.argv[6])

assert(approximationMethod == "variational" or approximationMethod == "MCMC")
assert(beta >= 0.0 and beta <= 1.0)
assert(NUMBER_OF_CPUS >= 1)


if DATA_TRANSFORMATION == "center":
    dataVectorsAllOriginal = statHelper.centerData(dataVectorsAllOriginal)
elif DATA_TRANSFORMATION == "normalize":
    dataVectorsAllOriginal = statHelper.normalizeData(dataVectorsAllOriginal)
else:
    assert(DATA_TRANSFORMATION == "none")


print "allClustersCandidatesNrs = ", allClustersCandidatesNrs

RANDOM_GENERATOR_SEED = 9899832
numpy.random.seed(RANDOM_GENERATOR_SEED)


dataVectorsAllTrain = dataVectorsAllOriginal
sampledCovAllTrain = numpy.asmatrix(numpy.cov(dataVectorsAllTrain.transpose(), rowvar=True, bias=True))

hiddenVarIds = numpy.ones(dataVectorsAllOriginal.shape[1], dtype = numpy.int)

if methodName == "sampleCov":
    allLambdaValues = [1.0]
elif methodName == "tikhonov" or methodName == "tikhonov2" or methodName == "lapRegApprox":
    allLambdaValues = [0.01, 0.05]
    for i in xrange(10):
        allLambdaValues += [0.1 * (i + 1)]
    for i in xrange(99):
        allLambdaValues += [1.0 * (i + 2)]                    
elif lambdaRange == "allLambda":
    
    if len(hiddenVarIds) <= 100:
        allLambdaValues = [0.0001, 0.0005]
        for i in xrange(10):
            allLambdaValues += [0.001 * (i + 1)]
    else:
        allLambdaValues = [0.01, 0.05]
        for i in xrange(10):
            allLambdaValues += [0.1 * (i + 1)]

elif lambdaRange == "allLambdaExtended":
    allLambdaValues = [0.005, 0.006, 0.007, 0.008, 0.009]
    for i in xrange(100):
        allLambdaValues += [0.01 * (i + 1)]
    for i in xrange(10):
        allLambdaValues += [1.0 + 0.1 * (i + 1)]
        
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

bestNegMarginalProp, clusterAssignmentsEBIC, clusterAssignmentsNegMarginal, clusterAssignmentsNegMarginalWT, clusterAssignmentsNegMarginalProp, clusterAssignmentsEBIC05, clusterAssignmentsEBIC1, clusterAssignmentsAIC, clusterAssignmentsCHI, clusterAssignmentsSil, bestANMI, bestRS, allCollectedClusterings, allMarginalLikelihoodsProposedMethod, allMarginalLikelihoodsBaseline, avgRuntimeProposed = EBIC.evalWithEBIC(dataVectorsAllTrain, sampledCovAllTrain, n, p, allLambdaValues, allClustersCandidatesNrs, methodName, SPECTRAL_CLUSTERING_VARIANT, hiddenVarIds, USE_PROPOSED_MARGINAL, minNrVariables, SOLVER, "", PRIOR_SCALE_FOR_NOISE, ADD_TRUE_CLUSTERING, clusteringMethod, PRIOR_SCALE_FOR_CLUSTER, INCLUDES_ALL_IN_ONE, ONLY_TRUE_CLUSTERING, beta, df, approximationMethod, NUMBER_OF_MCMC_SAMPLES, NUMBER_OF_CPUS, MCMCestimatorType, KAPPA)

assert(SOLVER == "ADMM-3BLOCK")
assert(PRIOR_SCALE_FOR_NOISE == 1.0)
assert(PRIOR_SCALE_FOR_CLUSTER == 1.0) 
    
    
def saveRanking(allCollectedClusterings, allScores, filename):
    
    allSortedClusteringIds = numpy.argsort(-1.0 * allScores)
    
    with open(filename, "w") as f:
        for i in allSortedClusteringIds:
            f.write(str(allScores[i]))
            f.write("\t")
            f.write(idcHelper.getVectorAsStr(allCollectedClusterings[i]))
            f.write("\n")
    
    print "saved to ", filename
    return

for scoreName in ["AIC", "BIC", "EBIC_0.5", "EBIC_1.0", "CalinskiHarabazIndex"]:
    allScores = EBIC.getBaselineScores(allCollectedClusterings, sampledCovAllTrain, n, scoreName)
    saveRanking(allCollectedClusterings, allScores, FILENAME + "_" + scoreName)
    
saveRanking(allCollectedClusterings, allMarginalLikelihoodsBaseline, FILENAME + "_" + "basicInverseWishartPrior")
saveRanking(allCollectedClusterings, allMarginalLikelihoodsProposedMethod, FILENAME + "_" + "robustBayes")

print "--- SUCCESSFULLY SAVED ALL CLUSTERINGS AND SCORES OF PROPOSED METHOD AND BASLINE ---"
                    
                



