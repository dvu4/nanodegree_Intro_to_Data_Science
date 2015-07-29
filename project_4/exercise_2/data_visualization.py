from pandas import *
from ggplot import *

def plot_weather_data(turnstile_weather):
    ''' 
    Use ggplot to make another data visualization focused on the MTA and weather
    data we used in assignment #3. You should make a type of visualization different
    than you did in exercise #1, and try to use the data in a different way (e.g., if you
    made a lineplot concerning ridership and time of day in exercise #1, maybe look at weather
    and try to make a histogram in exercise #2). 
    
    You should feel free to implement something that we discussed in class
    (e.g., scatterplots, line plots, or histograms) or attempt to implement 
    something more advanced if you'd like.  Here are some suggestions for things
    to investigate and illustrate:
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

    print turnstile_weather.ix[:10]

    df = turnstile_weather[['rain', 'ENTRIESn_hourly']].groupby('rain', as_index = False).sum()


    map_dict = {1: "rain", 0: "no rain"}
    turnstile_master["weather"] = turnstile_master["rain"].map(map_dict)
    plot = ggplot(turnstile_master, aes(x='ENTRIESn_hourly', color = 'weather')) \
        + geom_bar(aes(weight='ENTRIESn_hourly'),fill= '') \
        + ggtitle('NYC Subway ridership by weather') + xlab('Entries') + ylab('Frequency')

    #print plot

    
    plot = ggplot(turnstile_weather, aes(x='ENTRIESn_hourly', color = 'rain')) \
           + geom_bar(aes(weight='ENTRIESn_hourly'), fill='green') + geom_density()\
           + ggtitle('NYC Subway ridership by day of week') + xlab('Day') + ylab('Entries')
    plot = ggplot(turnstile_weather, aes(x='ENTRIESn_hourly', color = 'fog')) \
           + geom_bar(aes(weight='ENTRIESn_hourly'), fill='orange') \
           + ggtitle('NYC Subway ridership by day of week') + xlab('Day') + ylab('Entries')
            
    plot = ggplot(df,aes(x='rain')) \
          + geom_bar(aes(weight='ENTRIESn_hourly'),fill='orange',binwidth = 0.5) \
          + ggtitle('NYC Subway ridership by rain') + xlab('rain') + ylab('Entries') 
    return plot


    '''
    plot = ggplot(turnstile_weather, aes('EXITSn_hourly', 'ENTRIESn_hourly')) + stat_smooth(span=.15, color='black', se=True)+ geom_point(color='lightblue') + ggtitle("MTA Entries By The Hour!") + xlab('Exits') + ylab('Entries')
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
