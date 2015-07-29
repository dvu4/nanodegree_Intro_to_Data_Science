import numpy
import scipy.stats
import pandas
import scipy.special


def compare_averages(filename):
    """
    Performs a t-test on two sets of baseball data (left-handed and right-handed hitters).

    You will be given a csv file that has three columns.  A player's
    name, handedness (L for lefthanded or R for righthanded) and their
    career batting average (called 'avg'). You can look at the csv
    file via the following link:
    https://www.dropbox.com/s/xcn0u2uxm8c4n6l/baseball_data.csv
    
    Write a function that will read that the csv file into a pandas data frame,
    and run Welch's t-test on the two cohorts defined by handedness.
    
    One cohort should be a data frame of right-handed batters. And the other
    cohort should be a data frame of left-handed batters.
    
    We have included the scipy.stats library to help you write
    or implement Welch's t-test:
    http://docs.scipy.org/doc/scipy/reference/stats.html
    
    With a significance level of 95%, if there is no difference
    between the two cohorts, return a tuple consisting of
    True, and then the tuple returned by scipy.stats.ttest.  
    
    If there is a difference, return a tuple consisting of
    False, and then the tuple returned by scipy.stats.ttest.
    
    For example, the tuple that you return may look like:
    (True, (9.93570222, 0.000023))
    """
    ###http://stackoverflow.com/questions/354883/how-do-you-return-multiple-values-in-python

    # Read data into pandaframe
    data = pandas.read_csv(filename)
    #print data

    #Split data set into two data frames - RIGHT-HANDED AND LEFT-HANDED BATTTER
    sample1 = data[data["handedness"] == "R"]
    #print sample1
    sample2 = data[data["handedness"] == "L"]
    #print sample2

    # Perform Welch's t-test
    t_stat, p_val = scipy.stats.ttest_ind(sample1["avg"], sample2["avg"], equal_var=False)
    #print t_stat, p_val
    #print p_val >= 0.05


    #Perform t-test by computing the descriptive statistics of two samples

    s1_mean = sample1["avg"].mean()
    s1_var  = sample1["avg"].var(ddof=1)
    s1_n= sample1["avg"].size
    s1_dof = s1_n - 1

    s2_mean = sample2["avg"].mean()
    s2_var  = sample2["avg"].var(ddof=1)
    s2_n = sample2["avg"].size
    s2_dof = s2_n - 1

    # Compute Welch's t-test using the descriptive statistics.
    tf = (s1_mean - s2_mean) / numpy.sqrt(s1_var/s1_n + s2_var/s2_n)
    dof = (s1_var/s1_n + s2_var/s2_n)**2 / (s1_var**2/(s1_n**2*s1_dof) + s2_var**2/(s2_n**2*s2_dof))
    pf = 2*scipy.special.stdtr(dof, -numpy.abs(tf))
     
    print (t_stat, p_val)
    print (tf,pf)
    assert  round(t_stat,5) == round(tf,5)
    assert  round(p_val,5) == round(pf,5)
    
    return (p_val >= 0.05 ,(t_stat, p_val))

if __name__ == '__main__':
    filename = "baseball_data.csv"
    result = compare_averages(filename)
    print result
