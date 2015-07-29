from pandas import *
from ggplot import *
from datetime import *

def plot_weather_data(turnstile_weather):
    '''
    You are passed in a dataframe called turnstile_weather. 
    Use turnstile_weather along with ggplot to make a data visualization
    focused on the MTA and weather data we used in assignment #3.  
    You should feel free to implement something that we discussed in class 
    (e.g., scatterplots, line plots, or histograms) or attempt to implement
    something more advanced if you'd like.  

    Here are some suggestions for things to investigate and illustrate:
     * Ridership by time of day or day of week
     * How ridership varies based on Subway station
     * Which stations have more exits or entries at different times of day

    If you'd like to learn more about ggplot and its capabilities, take
    a look at the documentation at:
    https://pypi.python.org/pypi/ggplot/
     
    You can check out:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
     
    To see all the columns and data points included in the turnstile_weather 
    dataframe. 
     
    However, due to the limitation of our Amazon EC2 server, we are giving you about 1/3
    of the actual data in the turnstile_weather dataframe
    '''
    
from pandas import *
from ggplot import *
from datetime import *


def plot_weather_data(turnstile_weather):
    '''
    You are passed in a dataframe called turnstile_weather. 
    Use turnstile_weather along with ggplot to make a data visualization
    focused on the MTA and weather data we used in assignment #3.  
    You should feel free to implement something that we discussed in class 
    (e.g., scatterplots, line plots, or histograms) or attempt to implement
    something more advanced if you'd like.  

    Here are some suggestions for things to investigate and illustrate:
     * Ridership by time of day or day of week
     * How ridership varies based on Subway station (UNIT)
     * Which stations have more exits or entries at different times of day
       (You can use UNIT as a proxy for subway station.)

    If you'd like to learn more about ggplot and its capabilities, take
    a look at the documentation at:
    https://pypi.python.org/pypi/ggplot/
     
    You can check out:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
     
    To see all the columns and data points included in the turnstile_weather 
    dataframe. 
     
    However, due to the limitation of our Amazon EC2 server, we are giving you a random
    subset, about 1/3 of the actual data in the turnstile_weather dataframe.
    '''
    
    pandas.options.mode.chained_assignment = None
    df = turnstile_weather
    #print df.head()
    print df.ix[:10]
    
    temp = pandas.DatetimeIndex(df['DATEn'])
    df['day_of_week'] = temp.weekday
    
    newdf1 = df[['Hour', 'ENTRIESn_hourly']].groupby('Hour', as_index = False).sum()
    newdf2 = df[['UNIT', 'ENTRIESn_hourly']].groupby('UNIT', as_index = False).sum()
    newdf3 = df[['Hour', 'ENTRIESn_hourly']].groupby('Hour', as_index = False).sum()
    newdf4 = df[['UNIT', 'EXITSn_hourly']].groupby('UNIT', as_index = False).sum()    
    print newdf4

    
    newdf = df[['day_of_week', 'ENTRIESn_hourly']].groupby('day_of_week', as_index = False).sum()
   

    days = {0:'0-Monday',1:'1-Tuesday',2:'2-Wednesday',3:'3-Thursday',4:'4-Friday',5:'5-Satursday',6:'6-Sunday'}
    
    newdf['day_of_week'] = newdf['day_of_week'].apply(lambda x: days[x])
    
    plot = ggplot(newdf,aes(x='day_of_week')) \
          + geom_bar(aes(weight='ENTRIESn_hourly'),fill='green') \
          + ggtitle('NYC Subway ridership by day of week') + xlab('Day') + ylab('Entries')
         
    plot = ggplot(newdf1,aes(x='Hour')) \
          + geom_bar(aes(weight='ENTRIESn_hourly'),fill='green') \
          + ggtitle('NYC Subway ridership by time of day') + xlab('Time') + ylab('Entries')   
    plot = ggplot(newdf2,aes(x='UNIT')) \
          + geom_bar(aes(weight='ENTRIESn_hourly'),fill='green') \
          + ggtitle('NYC Subway ridership by Subway station') + xlab('Subway station') + ylab('Entries') 
    plot = ggplot(newdf3,aes(x='Hour')) \
          + geom_bar(aes(weight='ENTRIESn_hourly'),fill='green',binwidth = 1) \
          + ggtitle('NYC Subway ridership by time') + xlab('Hour') + ylab('Entries') 
    plot = ggplot(newdf4,aes(x='UNIT')) \
          + geom_bar(aes(weight='EXITSn_hourly'),fill='purple') \
          + ggtitle('NYC Subway ridership by Subway station') + xlab('Subway station') + ylab('Exits') 
    plot = ggplot(newdf4,aes(x='UNIT')) \
          + geom_bar(aes(weight='EXITSn_hourly'),fill='purple') \
          + geom_bar(newdf2, aes( weight='ENTRIESn_hourly'),fill='green') \
          + ggtitle('NYC Subway ridership by Subway station') + xlab('Subway station') + ylab('Exits') 
    '''
    plot = ggplot(newdf,aes(x='day_of_week')) \
          + geom_histogram(color='green') \
          + ggtitle('NYC Subway ridership by day of week') + xlab('Day') + ylab('Entries')        
 
    plot = ggplot(newdf,aes(x='ENTRIESn_hourly',color = 'UNIT')) \
          + geom_density() \
          + ggtitle('NYC Subway ridership by day of week') + xlab('Day') + ylab('Entries')  
    '''        
    return plot


    '''
    pandas.options.mode.chained_assignment = None
    df = turnstile_weather
    entriesByDayOfMonth = df[['DATEn', 'ENTRIESn_hourly']] \
        .groupby('DATEn', as_index=False).sum()
    entriesByDayOfMonth['Day'] = [datetime.strptime(x, '%Y-%m-%d') \
                                      .strftime('%w %A') \
                                  for x in entriesByDayOfMonth['DATEn']]
    entriesByDay = entriesByDayOfMonth[['Day', 'ENTRIESn_hourly']]\
        .groupby('Day', as_index=False).sum()
    plot = ggplot(entriesByDay, aes(x='Day')) \
           + geom_bar(aes(weight='ENTRIESn_hourly'), fill='blue') \
           + ggtitle('NYC Subway ridership by day of week') + xlab('Day') + ylab('Entries')
    
    return plot
    '''

    '''
    pandas.options.mode.chained_assignment = None
    df = turnstile_weather
    entriesByDayOfMonth = df[['DATEn', 'ENTRIESn_hourly']] \
        .groupby('DATEn', as_index=False).mean()
    entriesByDayOfMonth['Day'] = [datetime.strptime(x, '%Y-%m-%d') \
                                      .strftime('%w %A') \
                                  for x in entriesByDayOfMonth['DATEn']]
    entriesByDay = entriesByDayOfMonth[['Day', 'ENTRIESn_hourly']]\
        .groupby('Day', as_index=False).mean()
        
    plot = ggplot(entriesByDay, aes('Day', 'ENTRIESn_hourly')) + \
        geom_bar(fill = 'steelblue', stat='bar') + \
        ggtitle("NYC Subway ridership by day of week") + \
        xlab('Day') + ylab('Entries')
        
    return plot
    '''

    
if __name__ == "__main__":
    image = "plot.png"
    with open(image, "wb") as f:
        input_filename = 'turnstile_data_master_with_weather.csv'
        turnstile_weather = pandas.read_csv(input_filename)
        turnstile_weather['datetime'] = turnstile_weather['DATEn'] + ' ' + turnstile_weather['TIMEn']
        gg =  plot_weather_data(turnstile_weather)
        ggsave(f, gg)
