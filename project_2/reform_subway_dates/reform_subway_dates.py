import datetime
import time
import pandas as pd

def reformat_subway_dates(date):
    '''
    The dates in our subway data are formatted in the format month-day-year.
    The dates in our weather underground data are formatted year-month-date.
    
    In order to join these two data sets together, we'll want the dates formatted
    the same way.  Write a function that takes as its input a date in the MTA Subway
    data format, and returns a date in the weather underground format.
    
    Hint: 
    There is a useful function in the datetime library called strptime. 
    More info can be seen here:
    http://docs.python.org/2/library/datetime.html#datetime.datetime.strptime
    '''
    
    #print date    
    date_t = time.strptime(date, "%m-%d-%y")
    #print date_t[:]
    #print date_t[0]#year
    #print date_t[1]#month
    #print date_t[2]#date
    date_formatted = datetime.datetime(date_t[0],date_t[1],date_t[2]).strftime("%Y-%m-%d")
    #print date_formatted
    return date_formatted

if __name__ == "__main__":
    input_filename = "turnstile_data_master_subset_time_to_hours.csv"
    output_filename = "output.csv"

    turnstile_master = pd.read_csv(input_filename)
    student_df = turnstile_master.copy(deep=True)
    student_df['DATEn'] = student_df['DATEn'].map(reformat_subway_dates)
    student_df.to_csv(output_filename)
