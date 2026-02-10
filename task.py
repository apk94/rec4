import json
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from sklearn import linear_model, tree, metrics
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Task(object):
    def __init__(self, bike_df, bank_df):
        np.random.seed(31415)
        self.bike_df = bike_df.sample(1000).copy()
        self.bank_df = bank_df.copy()

    def t1(self):
        train = self.bike_df.iloc[0:900]
        train_x = train[['weekday']].values
        train_y = train[['cnt']].values

        test = self.bike_df.iloc[900:]
        test_x = test[['weekday']].values
        test_y = test[['cnt']].values
       
        regr = linear_model.LinearRegression()  
        regr.fit(train_x, train_y)
        predict_y = regr.predict(test_x)
        
        meansq_error = np.mean((predict_y - test_y) ** 2)
        print ("Mean squared error: %.2f" % meansq_error)
        return meansq_error

    def t2_1(self):
        train = self.bike_df.iloc[0:900]
        train_x = train[['season', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'temp_feels', 'hum', 'windspeed']].values
        train_y = train[['cnt']].values

        test = self.bike_df.iloc[900:]
        test_x = test[['season', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'temp_feels', 'hum', 'windspeed']].values
        test_y = test[['cnt']].values
        
        regr = linear_model.LinearRegression()  
        regr.fit(train_x, train_y)
        predict_y = regr.predict(test_x)
        
        meansq_error = np.mean((predict_y - test_y) ** 2)
        print ("Mean squared error: %.2f" % meansq_error)
        return meansq_error

    ##task2.2: Task 1 shows the MSE for weekdays only, but task 2.1 shows all of the attributes. 
    ##The MSE for weekdays is significantly higher than the MSE for all attributes. I think it's
    ##better to use all attributes so it is a more accurate description.
 
    def t3(self):
        self.bank_df['sex'] = self.bank_df['sex'].replace(['FEMALE', 'MALE'], [1, 2])
        self.bank_df['region'] = self.bank_df['region'].replace(['INNER_CITY', 'TOWN', 'RURAL', 'SUBURBAN'], [1, 2, 3, 4])
        self.bank_df['married'] = self.bank_df['married'].replace(['YES', 'NO'], [1, 2])
        self.bank_df['mortgage'] = self.bank_df['mortgage'].replace(['YES', 'NO'], [1, 2])  # Fixed quote

        train = self.bank_df.iloc[:500]
        train_x = train[['sex', 'region', 'married']].values
        train_y = train[['mortgage']].values

        test = self.bank_df.iloc[500:]
        test_x = test[['sex', 'region', 'married']].values
        test_y = test[['mortgage']].values  # Fixed: was train, should be test
        
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(train_x, train_y)

        predict_y = clf.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, predict_y)
        return accuracy
        
if __name__ == "__main__":
    t = Task(pd.read_csv('http://labrinidis.cs.pitt.edu/cs1656/data/bike_share.csv'), pd.read_csv('http://labrinidis.cs.pitt.edu/cs1656/data/bank-data.csv'))
    print("---------- Task 1 ----------")
    print(t.t1())
    print("--------- Task 2.1 ---------")
    print(t.t2_1())
    print("---------- Task 3 ----------")
    print(t.t3())





