#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568

import sys
import pandas as pd
from sklearn import mixture

def main():
    test_data = sys.argv[1]
    data = pd.read_csv(test_data, header=None)
    points = mixture.GaussianMixture(n_components=3,covariance_type='full')
    points.fit(data)   
    print ("Amplitudes :\n",points.weights_)
    print ("\nMeans: \n",points.means_)
    print ("\nCovariances: \n",points.covariances_)    

if __name__ == '__main__':
    main()