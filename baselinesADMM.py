import numpy
import idcHelper

def fastGraphicalLassoExceptDiagonal(sampledCov, lambdaValue, initAux):
    
    RELATIVE_ERROR_CRITERIA = 0.00001
    print "RELATIVE_ERROR_CRITERIA = ", RELATIVE_ERROR_CRITERIA
    
    rho = 1.0
    
    # warm start:
    currentZ = initAux[0]
    currentU = initAux[1]
    previousObjValue = float("inf")
    
    # cold start:
    # p = sampledCov.shape[0]
    # currentZ = numpy.zeros((p,p))
    # currentU = numpy.zeros((p,p))
    
    MAX_ITERATION = 100
    iterationCount = 0
    
    for i in xrange(MAX_ITERATION):
        iterationCount = i
        currentX = getOptimalX(sampledCov, currentZ, currentU, rho)
        currentZ = getOptimalZ(currentX, currentU, lambdaValue, rho)
        currentU = currentU + currentX - currentZ
        
        currentObjValue = getObjValueL1ExceptDiagonal(sampledCov, lambdaValue, currentX)
        # print "current obj value = ", currentObjValue
        
        if i > 0:
            stoppingCriteria = numpy.abs(previousObjValue - currentObjValue) / numpy.abs(previousObjValue)
            if stoppingCriteria < RELATIVE_ERROR_CRITERIA:
                break
        
        # assert(currentObjValue <= previousObjValue)
        previousObjValue = currentObjValue
        
    
    print "ADMM method used iterations = ", iterationCount    
    print "ADMM obj value = ", previousObjValue
    return currentX, (currentZ, currentU)

# checked
def getOptimalX(sampledCov, currentZ, currentU, p):
    rightHandSide = p * (currentZ - currentU) - sampledCov
    bestX = idcHelper.getFastDiagonalSolution(rightHandSide, p)
    return bestX

# tested
def softThresholding(v, t):
    assert(t > 0)
    if numpy.abs(v) <= t:
        return 0.0
    else:
        return v - numpy.sign(v) * t

# tested, same as in ADMM book page 47, except that we do not penalize the diagonal elements
def getOptimalZ(currentX, currentU, lambdaValue, p):
    bestZ = currentX + currentU
    for i in xrange(0, bestZ.shape[0], 1):
        for j in xrange(i + 1, bestZ.shape[0], 1):
            # apply soft thresholding
            bestZ[i, j] = softThresholding(bestZ[i, j], lambdaValue / p)
            bestZ[j, i] = bestZ[i, j]
            
    return bestZ

# checked
def getObjValueL1ExceptDiagonal(S, lambdaValue, X):
    assert(type(S) == numpy.matrixlib.defmatrix.matrix)
    assert(type(X) == numpy.matrixlib.defmatrix.matrix)
    
    n = S.shape[0]
    diagZeroMatrix = numpy.asmatrix(numpy.ones((n,n)) - numpy.eye(n))
    
    reg = numpy.multiply(diagZeroMatrix, X)
    reg = numpy.sum(numpy.abs(reg))
    
    return -1 * idcHelper.getLogLikelihood(X, S) + lambdaValue * reg


def test():
    currentX = numpy.asarray([[0.05, -0.8, 0.6, 0.3], [-0.8, 1.0, 0.05, 0.2], [0.6, 0.05, 0.05, 0.1], [0.3, 0.2, 0.1, 1.0]])
    currentU = numpy.zeros((4,4))
    lambdaValue = 1.0
    p = 10.0
    print "currentX:"
    print currentX
    print "soft-thresholded currentX:"
    print getOptimalZ(currentX, currentU, lambdaValue, p)
    # print softThresholding(v = -0.11, t = 0.1)