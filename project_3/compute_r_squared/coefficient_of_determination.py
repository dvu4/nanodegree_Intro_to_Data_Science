import pandas as pd
import numpy as np

from prediction import predictions


def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced 
    # predictions.
    # 
    # Numpy has a couple of functions -- np.mean() and np.sum() --
    # that you might find useful, but you don't have to use them.

    # YOUR CODE GOES HERE
    '''
    data : y
    prediction : f
    average of data : y_bar
    r_squared = 1 - sum(y-f)**2/sum(y-y_bar)**2
    '''

    data_avg = np.mean(data)
    print data_avg
    print np.sum((data-predictions)**2)
    print np.sum((data-data_avg)**2)
    
    #Total sum of squares (SST)
    SST = np.sum((data-data_avg)**2)
    #regression sum of squares (SSR)
    SSR = np.sum((data-predictions)**2)
    
    r_squared = 1 - SSR/SST

    return r_squared


if __name__ == "__main__":
    input_filename = "turnstile_data_master_with_weather.csv"
    turnstile_master = pd.read_csv(input_filename)
    predicted_values = predictions(turnstile_master)
    r_squared = compute_r_squared(turnstile_master['ENTRIESn_hourly'], predicted_values)
    print "R-Squared is :",r_squared
