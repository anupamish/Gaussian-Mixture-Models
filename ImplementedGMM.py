#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568
import numpy as np
np.random.seed(0)

def main():
	numberOfIteartionsForEM = 10
	numberOfGaussians = 3
	k=3
	data = np.genfromtxt(sys.argv[1],delimiter=",")
	Ric = np.transpose(np.random.rand(k,data.shape[0]))

	gloabalCoVariance = np.random.rand(k,data.shape[1],data.shape[1]) # k x Dim x Dim
	globalMean = np.random.rand(k,data.shape[1])
	gloabalAmplitude = np.random.rand(k,1)

	rowSum = np.sum(Ric,axis=1,keepdims=1)
	# rowSum.shape

	Ric = Ric/rowSum 
	# This is a N x k matrix where each row represents one cluster 
	# - soft assignment values of N points to cluster

	flag = True
	prevRic = np.copy(Ric)
	i=0
	while flag :
		i+=1
		print("Iteration ",i)

		# print ("Mean : ", globalMean,"\n")    
		# print ("CoVariance\n: ")
		# print (gloabalCoVariance,"\n\n")
		globalMean, gloabalCoVariance, globalAmplitude = BtoA(Ric)
		prevRic = Ric.copy()
		Ric = AtoB(globalMean,gloabalCoVariance,globalAmplitude)
		if np.allclose(prevRic,Ric,atol=0.000001):
			flag=False

	print("-------------------------------MEAN-------------------------------------")
	for i in range(0,numberOfGaussians):
		print("MEAN: {} \n {}".format(i+1,globalMean[i]))

	print("-------------------------------COVARIANCE-------------------------------------")
	for i in range(0,numberOfGaussians):
		print("COVARIANCE: {} \n {}".format(i+1,gloabalCoVariance[i]))
	print("-------------------------------AMPLITUDE-------------------------------------")
	for i in range(0,numberOfGaussians):
		print("AMPLITUDE: {} \n {}".format(i+1,globalAmplitude[i]))
  
 

def BtoA(Ric):
    RicTranspose = np.transpose(Ric)
#     This has shape k x N with each row holding soft prob assignments for all point to a particular cluster C.
    meanMatrix = (np.dot(RicTranspose,data))/np.sum(RicTranspose,axis=1).reshape(-1,1)
#     print(meanMatrix)
     # This is a k x N matrix which is multiplied to N x Dim data
#(and divided by sum of Weights for each guassian) to get K X D means
#     meanMatrix has a shape of K x D =(3 x 2) .. for each gaussian k we have a mean vector
    for i in range(0,k):
        curGausMean = meanMatrix[i]
        DTranspose = data- curGausMean
#         Dtranspose shape is 150 x 2
        D = np.transpose(DTranspose)
        weights = Ric[:,i] # get weights for the current cluster
        weights = weights.reshape(-1,1);
        DTranspose = (DTranspose * weights)/sum(weights)
        gloabalCoVariance[i] = np.dot(D,DTranspose)
    globalAmplitude = np.sum(Ric,axis=0)/data.shape[0]
    return meanMatrix, gloabalCoVariance, globalAmplitude;
#     Update the mean/coVariance matrices for k gaussians

def PDF(mean,covariance,amplitude,curDataPoint):
    determinant = np.linalg.det(covariance)
    if determinant==0:
        print("Here")
    multiplier = pow(1/(pow((2 * np.pi),len(mean)) * determinant),0.5) # len(mean) == dimension
    XminusUTranspose = (curDataPoint- mean).reshape(data.shape[1],-1)
    XminusU = np.transpose(XminusUTranspose)
    # print(covariance)
    inverse = np.linalg.inv(covariance)
    numerator = float((np.dot(np.dot(XminusU,inverse),XminusUTranspose) * -1)/2)
    expo = np.exp(numerator)
    # print("PDF Value:",multiplier*expo,"Amplitude: ",amplitude)
    return (amplitude * multiplier * expo)

def AtoB(globalMean,gloabalCoVariance,globalAmplitude):
    Ric = np.zeros((data.shape[0],k))
    for i in range(0,data.shape[0]):
        sum=0
        curDataPoint = data[i]
        curPointPDFs = np.random.rand(1,k)
        for j in range(0,k):
            curMean = globalMean[j]
            curAmplitude = globalAmplitude[j]
            curCovariance = gloabalCoVariance[j]
            curPointPDFs[0][j] = PDF(curMean,curCovariance,curAmplitude,curDataPoint)
        sum = np.sum(curPointPDFs,axis=1)
        curPointRic = curPointPDFs/sum
        Ric[i] = curPointRic
    return Ric

if __name__ == '__main__':
    main()
# print(np.sum(Ric,axis=1))





