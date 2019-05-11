import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn import linear_model  
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
#import tensorflow as tf

from sklearn.metrics import accuracy_score 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model, datasets, metrics


from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import NearestNeighbors
#from sklearn.neighbors import LocalOutlierFactor

import pandas as pd
import csv

from scipy import fftpack 
from scipy.fftpack import fft, ifft


from math import sqrt


f = open(r"C:\Users\syedt\Documents\Thesis\hrsampledata\fitbit3.csv")


reader = list(csv.reader(f))

time_stamp=[]

heart_rate=[]

steps=[]

gsr=[]

temp=[]

t=[]



heart_target_label=[]


time_stamp_string=[]

heart_rate_string=[]

steps_string=[]

gsr_string=[]

temp_string=[]

heart_target=[]

t_string=[]




for i in reader[1:]:
    
#Store the Time String from fitbit CSV file and the corresponding HR String into a list
    
    time_stamp_string.append(i[0])

    heart_rate_string.append(i[1])
    
    steps_string.append(i[2])
    
    gsr_string.append(i[3])
    
    temp_string.append(i[5])
        
    t_string.append(i[6])
            


#Detect where the readings had NA for HR and remove that data from list

for k in range(len(heart_rate_string)):  
    
#    if  time_stamp_string[k]=='NA':
        
        
#         time_stamp_string[k] =  time_stamp_string[k].replace('NA', '')
#    

    if heart_rate_string[k]=='NA':
        
        
        heart_rate_string[k] = heart_rate_string[k].replace('NA', '')
        
        
    if steps_string[k]=='NA':
        
        steps_string[k]=steps_string[k].replace('NA','')
        
    if gsr_string[k]=='NA':
        
        gsr_string[k]=gsr_string[k].replace('NA','')
        
        
    if temp_string[k]=='NA':
        
        temp_string[k]=temp_string[k].replace('NA','')
        

    if t_string[k]=='NA':
        
        t_string[k]=t_string[k].replace('NA','')
                
        
    
        
#Clear out empty data in the list
        
#time_stamp_string =list(filter(None, time_stamp_string))       
        
heart_rate_string= list(filter(None, heart_rate_string))

steps_string=list(filter(None, steps_string))

gsr_string=list(filter(None, gsr_string))

temp_string=list(filter(None, temp_string))

t_string=list(filter(None, t_string))



for m in range(len(gsr_string)):    
#    
##Convert String to Integer    
    
    
#    time_stamp.append(int(time_stamp_string[m])) 
    heart_rate.append(int(heart_rate_string[m]))
    steps.append(float(steps_string[m]))  
    gsr.append(float(gsr_string[m]))
    temp.append(float(temp_string[m]))
    t.append(float(t_string[m]))
    
time_stamp=[]

heart_rate_high=[]

steps_high=[]

gsr_high=[]

temp_high=[]

t_high=[]


plt.plot(t, heart_rate, label='c')
from scipy import signal
#
#heart_rate=np.fft.fft(heart_rate)




heart_rate =np.asarray(heart_rate)
heart_rate =heart_rate/60

print ('********** BEfore Process : This is the Heart Rate High********************')
print (heart_rate)

fs =1   # Sampling frequency

w =  (fs / 2) # Normalize the frequency
b, a = signal.butter(3, w, 'low',analog=False)
output_heart = signal.filtfilt(b, a, heart_rate)
plt.plot(t, output_heart, label='filtered')
plt.legend()
plt.show()






#
#plt.plot(t, gsr, label='c')
#fs = 1  # Sampling frequency
#
#fc = 3  # Cut-off frequency of the filter
#w = (fs / 2) # Normalize the frequency
#c, d = signal.butter(5, w, 'low')
#output_gsr = signal.filtfilt(c, d, np.fft.fft(gsr))
#plt.plot(t, output_gsr, label='filtered')
#plt.legend()
#plt.show()
#
#
##
#plt.plot(t, temp, label='c')
#fs =1  # Sampling frequency
#
#fc = 3  # Cut-off frequency of the filter
#w =  (fs / 2) # Normalize the frequency
#e, f = signal.butter(5, w, 'low')
#output_temp = signal.filtfilt(e, f, np.fft.fft(temp))
#plt.plot(t, output_temp, label='filtered')
#plt.legend()
#plt.show()
#
#


output_heart= np.int16(output_heart*60)


for f in range(len(output_heart)):    
    
#    if  heart_rate[f]>130 and heart_rate[f]<150:
        

#         if  abs(orange[f])<abs(np.mean(orange)):
             
#             if  abs(bannana[f])<abs(np.mean(bannana)):
    
        
###        
            heart_rate_high.append(output_heart[f])
            
            steps_high.append(steps[f])

            gsr_high.append(gsr[f])

            temp_high.append(temp[f])
            
            t_high.append(t[f])
        


#Map the data
#time_stamp=list(map(lambda x: [x],time_stamp))         
heart_rate_high=list(map(lambda x: [x],heart_rate_high))    
steps_high=list(map(lambda y: [y],steps_high))   
gsr_high=list(map(lambda z: [z],gsr_high))   
temp_high=list(map(lambda z: [z],temp_high)) 
t_high=list(map(lambda z: [z],t_high)) 



#Group the data




#time_stamp=np.array(time_stamp)
heart_rate_high= np.squeeze(np.array(heart_rate_high))
steps_high=np.squeeze(np.array(steps_high))
temp_high=np.squeeze(np.array(temp_high))
gsr_high=np.squeeze(np.array(gsr_high))
t_high=np.squeeze(np.array(t_high))


#heart_rate_high = heart_rate_high.reshape(len(heart_rate_high),1)

gsr_high = gsr_high.reshape(len(gsr_high),1)
t_high = t_high.reshape(len(t_high),1)

#gsr_high = abs(np.fft.ifft(np.asarray(gsr_high)))
#temp_high = abs(np.fft.ifft(np.asarray(temp_high)))
#heart_rate_high =np.int16(heart_rate_high*60)
#
print ('********** ############ : This is the Heart Rate High*###########*******************')
print (heart_rate_high)

#
data =np.column_stack((gsr_high,heart_rate_high))
data=data.reshape(len(gsr_high),2)



#Here we split our data



train, test,train_labels, test_labels = train_test_split(data,temp_high,test_size=0.33,random_state=10)

#
##
#train= np.array(train)
#scaler.fit(train)
#train = scaler.transform(train)
#test = scaler.transform(test)



 
    
##Create the Decision Tree Classifier
#
#lof=LocalOutlierFactor( n_neighbors=35, contamination=outliers_fraction)
#lof.fit(train)
#
#
#
#tree_regressor = DecisionTreeRegressor(random_state=0,max_depth=10)
#
#regr_2_adaboost = AdaBoostRegressor(tree_regressor, n_estimators=800)
#
#
#
#tree_regressor.fit(train, train_labels)
#regr_1_decisiontree_prediction=tree_regressor.predict(test)
#
#
#regr_2_adaboost.fit(train, train_labels)
#
#regr_2_adaboost_prediction=regr_2_adaboost.predict(test)
#
#RSME_adaboost=mean_squared_error(test_labels, regr_2_adaboost_prediction)
#
#score_adaboost_regressor=cross_val_score(regr_2_adaboost, test, test_labels,cv=10)
#
#print('this is the score using adaboost regressor')
#print(score_adaboost_regressor.mean())

#
## Plot the results
#plt.figure()
#plt.scatter(test, test_labels, c="k")
#
#plt.plot(test, regr_2_adaboost_prediction, c="r", linewidth=1)
#plt.xlabel("data")
#plt.ylabel("target")
#plt.title("Boosted Decision Tree Regression")
#plt.legend()
#plt.show()



print('this is the score using Kneighbours regressor')

neigh = KNeighborsRegressor()
neigh.fit(train, train_labels) 

score_K=cross_val_score(neigh, test, test_labels)

print(score_K.mean())



print('this is the score using Gradient Boosting regressor')

est = GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, max_depth=12, random_state=0, loss='huber').fit(train, train_labels)

est.fit(train, train_labels.ravel())
score_GBoost=cross_val_score(est, test, test_labels.ravel())


print(score_GBoost.mean())




#print('this is the score using Logistic algorithm')
#
#logreg = linear_model. LogisticRegression()
#
#logreg.fit(train, train_labels)
#logreg_score=cross_val_score(logreg,  test, test_labels)
#print(logreg_score)


print('this is the score using Bayesian Ridge Regression')

BRidge = linear_model.BayesianRidge(n_iter=500,alpha_1=1e-06, alpha_2=1e-06,tol=1.e-8)
BRidge.fit(train, train_labels)
score_BRidge=cross_val_score(BRidge,  test, test_labels)

print(score_BRidge.mean())

BRidge_prediction=BRidge.predict(test)

RSME_BRidge=mean_squared_error(test_labels, BRidge_prediction)
print(sqrt(RSME_BRidge))



#print('this is the score using Lasso Lars Regression')
#
#lasso = linear_model.LassoLars()
#lasso.fit(train, train_labels)
#score_lasso=cross_val_score(BRidge,  test, test_labels)
#print(score_lasso.mean())
#
#
#lasso_prediction=lasso.predict(test)
#
#RSME_lasso=mean_squared_error(test_labels, lasso_prediction)
#print(sqrt(RSME_lasso))
#
#
#
#print('this is the score using SGD Regression')
#stochastic = linear_model.SGDRegressor()
#stochastic.fit(train, train_labels)
#score_stochastic=cross_val_score(stochastic,  test, test_labels)
#print(score_stochastic.mean())
#
#stochastic_prediction=stochastic.predict(test)
#
#RSME_stochastic=mean_squared_error(test_labels, stochastic_prediction)
#print(sqrt(RSME_stochastic))




#
#print('this is the score using Random Forest Regressor algorithm')
#
#train, train_labels = make_regression(n_features=4, n_informative=500,  random_state=0, shuffle=False)
#randomforest = RandomForestRegressor(max_depth=12, random_state=0)
#randomforest.fit(train,train_labels)
#score_randomforest=cross_val_score(randomforest, test, test_labels)
#
#print(score_randomforest.mean())
#
#
#randomforest_prediction=randomforest.predict(test)
##
#RSME_randomforest=mean_squared_error(test_labels, randomforest_prediction)
#print(sqrt(RSME_randomforest))
#
#
#print('this is the score using bagging regressor algorithm')
#
#clf_bagging=BaggingRegressor(n_estimators=300).fit(train, train_labels)
#
#
#clf_bagging.fit(train, train_labels)
#
#score_bagging=cross_val_score(clf_bagging, test, test_labels)
#print(score_bagging.mean())
#
#bagging_prediction=clf_bagging.predict(test)
#
#RSME_bagging=mean_squared_error(test_labels, bagging_prediction)
#print(sqrt(RSME_bagging))
#
##print("Logistic regression using RBM features:\n%s\n" % (metrics.classification_report( test_labels,clf_bagging.predict(test))))
#
#
##from sklearn.neighbors.kde import KernelDensity
##kde = KernelDensity(kernel='gaussian', bandwidth=0.0001).fit(train, train_labels)
##score_kde=cross_val_score(kde, test, test_labels)
##
##print('this is the score using Unsupervised Learning Kernel Density algorithm')
##print(score_kde)      
#
#
#
print('this is the score using MLP algorithm')

clf_MLP = MLPRegressor(solver='lbfgs',learning_rate_init=0.01, alpha=0.001,hidden_layer_sizes=(60,30,21,9,5))
clf_MLP.fit(train, train_labels)  

MLP_prediction=clf_MLP.predict(test)

score_MLP=cross_val_score(clf_MLP, test, test_labels)

print("Neural Network")

print(score_MLP.mean())
#
#
#
RSME_MLP=mean_squared_error(test_labels, MLP_prediction)
print(sqrt(RSME_MLP))


#
#
#
### Models we will use
##logistic = linear_model.LogisticRegression()
##rbm = BernoulliRBM(random_state=0, verbose=True)
##classifier_bernoulli = Pipeline(steps=[('rbm', rbm), ('logistic', BRidge)])
##rbm.learning_rate = 0.00001
##rbm.n_iter = 10
### More components tend to give better prediction performance, but larger
### fitting time
##rbm.n_components = 1000
##logistic.C = 6000.0
### Training RBM-Logistic Pipeline
##classifier_bernoulli.fit(train, train_labels)
##score_bernoulli=cross_val_score(classifier_bernoulli,
##print("Logistic regression using RBM features:"+ (mean_squared_error( test_labels,classifier_bernoulli.predict(test))))
##
##
##
##
#
#
#
##
##print('this is the score using Convolutional Neural Network algorithm')
##from sknn.mlp import Classifier, Convolution, Layer
##cnn = Classifier(layers=[Convolution("Rectifier", channels=8, kernel_shape=(10,10)), Layer("Softmax")],learning_rate=0.002, n_iter=5)
##cnn.fit(train,train_labels)
##score_cnn=cross_val_score(cnn, test, test_labels)
##print(score_cnn)
#
#
#
#
##
##print('this is the score using SVR algorithm')
##
##

#
#
#
