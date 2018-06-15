

Contains the code for variable clustering and clustering evaluation according to 
"Robust Bayesian Model Selection for Variable Clustering with the Gaussian Graphical Model"

required python packages: numpy, scipy, tabulate

In order to cluster your data use "clusterEvaluation.py".

EXAMPLE:
python -m clusterEvaluation test.csv normalize 10 variational 0.02 4

Arguments  
1: data file in csv format (each row is one sample)  
2: normalization of variables ("normalize") or not "none"  
3: maximal number of clusters that should be considered (e.g. if 10, then consider number of clusters 2,3,4,5,...,10)  
6: "variational" or "MCMC" approximation of the proposed method  
7: beta (recommended value 0.02)  
8: number of cpus

Assumes that the data is saved as a csv file with  
number of rows = number of samples,  
number of columns = number of variables


outputs all clusterings found and the marginal likelihoods / scores for the proposed method and the following other evaluation metricies:
AIC
BIC
EBIC
basic inverse Wishart prior
Calinski-Harabaz index
robustBayes (proposed method)

for more parameters like the number of MCMC samples see beginning of file "clusterEvaluation.py".

In order to reproduce the simulation experiments use "simulatedDataExperiments.py" as described in the beginning of "simulatedDataExperiments.py".
