from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier

import numpy as np  
import pandas as pd  

def svm_handler(X_train, y_train):
    '''
    Input: 
        - X_train: The (X) training feature matrix. 
        - y_train: The (Y) training target matrix.
        
    Output: 
        svm: An SVM object. 
    '''
    svm = SVC(gamma = 'auto', kernel='linear')
    svm.fit(X_train, y_train)
    return svm 


def predict_score(model, X_test, y_test):
    '''
    Input: 
        - model: An ML model (e.g., ANN, SVM, etc.) 
        - X_test: The (X) testing feature matrix 
        - y_test: The (Y) testing target matrix 
        
    Output:
        - The accuracy of the model rounded to 4 decimals 
    '''
    prediction = model.predict(X_test)
    
    # Calculate the absolute errors
    errors = abs(prediction - y_test)
    
    # Calculate mean absolute percentage error
    mean_APE = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mean_APE)
    
    return round(accuracy, 4)


def logistic_handler(X_train, y_train):
    '''
    Input: 
        - X_train: The (X) training feature matrix. 
        - y_train: The (Y) training target matrix.
        
    Output: 
        logreg: A Logistic Regression object. 
    '''
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    return logreg 

def neural_network_handler(X_train, y_train, solver, alpha, hidden_layers, random_state):
    '''
    Input: 
        - X_train: The (X) training feature matrix. 
        - y_train: The (Y) training target matrix.
        - solver: The algorithm used to update the parameters.
        - alpha: The regularization term. This is used to avoid overfitting.
        - hidden_layers: The number of neurons in each hidden layer of the ANN
        - random_state: Determines random number generation for weights and bias initialization. 
        
    Output: 
        svm: An ANN object. 
    '''
    neural_network = MLPClassifier(solver=solver, alpha=alpha,hidden_layer_sizes=hidden_layers, random_state=random_state)
    neural_network.fit(X_train, y_train)
    return neural_network

    