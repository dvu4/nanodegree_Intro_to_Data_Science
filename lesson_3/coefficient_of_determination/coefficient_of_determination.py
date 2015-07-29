from prediction import predictions

import pandas as pd
import numpy as np

def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced 
    # predictions.
    # 
    # Numpy has a couple of functions -- np.mean() and np.sum() --
    # that you might find useful, but you don't have to use them.


    # YOUR CODE GOES HERE
    '''
    data y
    prediction f
    average of data y_bar
    r_squared = 1 - sum(y-f)**2/sum(y-y_bar)**2
    '''

    data_avg = np.mean(data)
    print data_avg
    print np.sum((data-predictions)**2)
    print np.sum((data-data_avg)**2)
    r_squared = 1 - np.sum((data-predictions)**2)/np.sum((data-data_avg)**2)
    return r_squared


if __name__ == "__main__":
    input_filename = "turnstile_data_master_with_weather.csv"
    turnstile_master = pd.read_csv(input_filename)
    predictions = predictions(turnstile_master)
    r_squared = compute_r_squared(turnstile_master['ENTRIESn_hourly'], predictions)
    print r_squared
