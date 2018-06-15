import numpy
import collections
import sklearn.metrics
import sklearn.datasets
import statHelper
import scipy.io



# checked
def getClustering(assignment):
    clustering = collections.defaultdict(set)
    for objId in xrange(len(assignment)):
        classId = assignment[objId]
        assert(int(classId) == classId)
        classId = int(classId)
        clustering[classId].add(objId)
    
    return clustering

# implementation according to formula in IR book (Manning), checked
def getPurity(referenceAssignment, predictedAssignment):
    assert(len(referenceAssignment) == len(predictedAssignment))
    assert(len(referenceAssignment) >= 2)
    assert(numpy.min(referenceAssignment) == 1)
    numberOfClasses = numpy.max(referenceAssignment)
    
    totalCorrectNumber = 0
    clusteringResult = getClustering(predictedAssignment)
    for k in clusteringResult:
        classCountInCurrentCluster = numpy.zeros(numberOfClasses + 1) # classId start with 1 !!!
        # print "predClusterId = " + str(k) + ", elements: " + str(clusteringResult[k])
        for objId in clusteringResult[k]:
            classId = referenceAssignment[objId]
            assert(classId >= 1 and classId <= numberOfClasses)
            classCountInCurrentCluster[classId] += 1
        # print numpy.max(classCountInCurrentCluster)
        totalCorrectNumber += numpy.max(classCountInCurrentCluster)
    
    # print " totalCorrectNumber = ", totalCorrectNumber
    # print "len(predictedAssignment) = ", len(predictedAssignment)
    return float(totalCorrectNumber) / float(len(predictedAssignment))
 
    
def testClusterEvaluation():
    # 1 = triangle, 2 = circle, 3 = cross
    labels_true = [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    labels_pred = [3, 3, 3, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 3, 3]
    print "example from IR book:"
    showResult(labels_true, labels_pred)
    print "simple example 1:"
    showResult([1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3])
    print "simple example 2:"
    showResult([1, 1, 2, 2, 3, 3], [1, 1, 1, 2, 3, 3])
    return 

def showResult(labels_true, labels_pred):
    purity = getPurity(labels_true, labels_pred)
    print "purity = ", purity
    
    nmi = sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred)
    print "normalized mutual information = ", nmi
    
    adjustedNMI = sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred)
    print "(adjusted for chance) normalized mutual information = ", adjustedNMI
    
    print str(round(nmi, 2)) + "/" + str(round(purity, 2))
    return

def getLabelRatios(assignment, numberOfClusters):
    observedCounts = statHelper.getCounts(assignment, numberOfClusters)
    totalCounts = numpy.sum(observedCounts)
    assert(totalCounts == len(assignment))
    assert(totalCounts > 0)
    ratios = observedCounts.astype(numpy.float32) / totalCounts
    return ratios
    
def showResultShort(labels_true, labels_pred):
    purity = getPurity(labels_true, labels_pred)
    nmi = sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred)
    rs = rand_score(labels_true, labels_pred)
    ars = sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
    print "rs/ars/nmi/purity = " + str(round(rs, 2)) + " & " + str(round(ars, 2)) + " & " + str(round(nmi, 2)) + " & " + str(round(purity, 2))
    return

def getRSandNMIwithAdjustment(labels_true, labels_pred):
    if labels_pred is None:
        return -1, -1, -1
    
    ars = sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
    anmi = sklearn.metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    hom = sklearn.metrics.homogeneity_score(labels_true, labels_pred)
    return ars, anmi, hom



#  adapted from source for "sklearn.metrics.adjusted_rand_score"
# see https://github.com/scikit-learn/scikit-learn/blob/a5ab948/sklearn/metrics/cluster/supervised.py#L113
def rand_score(labels_true, labels_pred):
    labels_true, labels_pred = sklearn.metrics.cluster.supervised.check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    n_classes = numpy.unique(labels_true).shape[0]
    n_clusters = numpy.unique(labels_pred).shape[0]
 
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (n_classes == n_clusters == 1 or
            n_classes == n_clusters == 0 or
            n_classes == n_clusters == n_samples):
        return 1.0
 
    # corresponds to "a + b" in Wikipedia: https://en.wikipedia.org/wiki/Rand_index
    sameBehaviourPairsCount = 0
    for i in xrange(0, n_samples):
        for j in xrange(i + 1, n_samples):
            if ((labels_true[i] == labels_true[j]) and (labels_pred[i] == labels_pred[j])) or ((labels_true[i] != labels_true[j]) and (labels_pred[i] != labels_pred[j])):
                sameBehaviourPairsCount += 1
                
    return float(sameBehaviourPairsCount) / float(sklearn.metrics.cluster.supervised.comb2(n_samples))
    


def sampleFromNoiseMixtureModel(mixtureProbs, modelMeans, modelSigmas, noiseSigmas, noiseProbs):
    clusterId = numpy.random.choice(len(mixtureProbs), p=mixtureProbs)
    noiseOrNot = numpy.random.choice(2, p=[noiseProbs[clusterId], 1.0 - noiseProbs[clusterId]])
    assert(noiseOrNot == 0 or noiseOrNot == 1)
    assert(len(noiseSigmas) == len(modelSigmas))
    
    if noiseOrNot == 0:
        # sample from noise model
        covariate = numpy.random.normal(loc = modelMeans[clusterId, :], scale = noiseSigmas[clusterId])
    else:
        # sample from cluster model
        covariate = numpy.random.normal(loc = modelMeans[clusterId, :], scale = modelSigmas[clusterId])
        
    return (clusterId + 1), covariate


def createNoiseClusterDataSamples(NUMBER_OF_CLUSTERS):
    
    assert(NUMBER_OF_CLUSTERS == 3)
    
    NUMBER_OF_OBSERVATIONS = 1000
    
    DATA_DIMENSION = 1
    
    mixtureProbs = [ 1.0/3.0,  1.0/3.0, 1.0/3.0]
    modelSigmas = [1.0, 1.0, 1.0]
    noiseSigmas = [15.0, 3.0, 1.0]
    noiseProbs = [0.3, 0.3, 0.0]
    
    # NUMBER_OF_CLUSTERS = len(mixtureProbs)
    modelMeans = numpy.zeros(shape = (NUMBER_OF_CLUSTERS, DATA_DIMENSION))
    modelMeans[0, :] = -5.0
    modelMeans[1, :] = 0.0
    modelMeans[2, :] = 10.0
    
    hiddenDataIds = numpy.zeros(NUMBER_OF_OBSERVATIONS, dtype = numpy.int_)
    dataVectors = numpy.zeros(shape = (NUMBER_OF_OBSERVATIONS, DATA_DIMENSION))
    
    for i in xrange(NUMBER_OF_OBSERVATIONS):
        hiddenDataIds[i], dataVectors[i, :] = sampleFromNoiseMixtureModel(mixtureProbs, modelMeans, modelSigmas, noiseSigmas, noiseProbs)
        assert(hiddenDataIds[i] >= 1 and hiddenDataIds[i] <= NUMBER_OF_CLUSTERS)
    
    print "finished data creation successfully"
    return dataVectors, hiddenDataIds


# load Wisconsin Breast Cancer from UCI
# http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#breast-cancer
def loadBreastCancerData(filename = "../datasets/breast-cancer.txt"):
    NUMBER_OF_CLUSTERS = 2
    dataVectors, hiddenDataIds = sklearn.datasets.load_svmlight_file(filename)
    hiddenDataIds[hiddenDataIds == 2.0] = 1
    hiddenDataIds[hiddenDataIds == 4.0] = 2
    hiddenDataIds = hiddenDataIds.astype(dtype = numpy.int_)
    dataVectors = dataVectors.toarray() # convert from sparse matrix to normal numpy array
    return dataVectors, hiddenDataIds, NUMBER_OF_CLUSTERS

def loadIrisData():
    iris = sklearn.datasets.load_iris()
    dataVectors = iris.data
    hiddenDataIds = iris.target + 1
    NUMBER_OF_CLUSTERS = numpy.max(hiddenDataIds)
    return dataVectors, hiddenDataIds, NUMBER_OF_CLUSTERS

# load letter data set
# http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#letter
def loadLetterData(filename = "../datasets/letter.scale.val"):
    NUMBER_OF_CLUSTERS = 26
    dataVectors, hiddenDataIds = sklearn.datasets.load_svmlight_file(filename)
    hiddenDataIds = hiddenDataIds.astype(dtype = numpy.int_)
    dataVectors = dataVectors.toarray() # convert from sparse matrix to normal numpy array
    assert(numpy.min(hiddenDataIds) == 1 and numpy.max(hiddenDataIds) == NUMBER_OF_CLUSTERS)
    return dataVectors, hiddenDataIds, NUMBER_OF_CLUSTERS


# saves in format required by "Warped Mixture Models"
def saveDataNormalizedInMat(dataVectors, hiddenDataIds, filename):
    dataVectorsNormalized = statHelper.normalizeData(dataVectors)
    y = numpy.copy(hiddenDataIds)
    y = y.reshape(hiddenDataIds.shape[0],1)
    y = y.astype(numpy.double)
    assert(y.min() == 1)
    scipy.io.savemat(filename, {"X":dataVectorsNormalized, "y":y})
    print "y shape = "
    print y.shape
    print "NUMBER_OF_CLUSTERS = " + str(hiddenDataIds.max())
    print "saved data to " + filename
    
    