
'''

Training script for Udacity Project 3
Sandeep Pawar
Ver 1
Date Jan 25, 2021

'''
import numpy as np
import pandas as pd
import argparse
from azureml.core.run import Run


from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.impute import SimpleImputer

from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import KernelPCA, PCA
from sklearn.svm import SVC

seed = 123


import warnings
warnings.filterwarnings("ignore")

run = Run.get_context()

def load_data(dataframe):
    
    # Load data with all the columns from the source
    # x is teh training data
    # y is the label for the training data
    
    x = dataframe.drop('y', axis=1)
    
    y = dataframe['y'] 
    
    return x, y 

x, y = load_data(dataframe)

def main():
    
    parser = argparse.ArgumentParser()
    
    weights = {-1:0.9334, 1:(1-0.9334)}

    parser.add_argument('--impute', type=str, default='median', help="Imputation Strategy")
    parser.add_argument('--kernel', type=str, default="rbf", help="Kernel for SVC")
    parser.add_argument('--gamma', type=str, default='auto', help="Gamma value")
    parser.add_argument('--penalty', type=float, default=1, help="Penalty")

    args = parser.parse_args()

    run.log("Imputation:", str(args.impute))
    run.log("kernel:", str(args.kernel))
    run.log("Gamma:", str(args.gamma))
    run.log("penalty:", str(args.penalty))
    
    clf = Pipeline([
    ('imputer', SimpleImputer(strategy=args.impute)), # To impute missing values
    ('threshold', VarianceThreshold(0.01)), # Remove near-constant features
    ('scaler', StandardScaler()),
    ('PCA', PCA(n_components = 200)),
    ('classification', (SVC(class_weight = weights, 
                           random_state=seed, 
                           kernel = args.kernel, 
                           C = args.penalty,
                           gamma = args.gamma, 
                           probability =True)))
    ])
        
    clf.fit(x, y)    
    
    score = cross_val_score(clf, X=x, y=y, cv=5, scoring = 'roc_auc')
        
    
    run.log("Mean_AUC", np.float( score.mean()))

    #Serialize the model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(clf, 'outputs/hyperDrive_{}_{}'.format(args.kernel,args.gamma, args.penalty))

if __name__ == '__main__':
    main()
