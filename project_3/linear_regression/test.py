import numpy as np
import pandas 
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier

# load the iris datasets
input_filename = "turnstile_data_master_with_weather.csv"
dataframe = pandas.read_csv(input_filename)

   
'''
features = dataframe[['Hour', 'maxpressurei', 'maxdewpti', 'mindewpti', 'minpressurei',
                      'meandewpti', 'meanpressurei','fog', 'rain', 'meanwindspdi', 'mintempi',
                      'meantempi', 'maxtempi','precipi', 'thunder']]
'''


features = dataframe[['rain', 'precipi', 'Hour', 'meantempi']]
           
dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
features = features.join(dummy_units)
    
# Values
values = dataframe['ENTRIESn_hourly']
   
# Get numpy arrays
#features_array = features.values
#values_array = values.values



# create a base classifier used to evaluate a subset of attributes
#model = SGDRegressor()
#model = LogisticRegression()
#results = model.fit(features, values)

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(features , values)
# display the relative importance of each attribute
print(model.feature_importances_)

'''
# create the RFE model and select 3 attributes
rfe = RFE(model, 18)
rfe = rfe.fit(features , values)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
'''
