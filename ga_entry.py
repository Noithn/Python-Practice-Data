import os
from numpy.core.defchararray import count, index
from numpy.core.numeric import True_
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import datetime as dt

from collections import Counter
from pandas.core.algorithms import factorize
from matplotlib.pyplot import figure
from scipy import stats
from seaborn import palettes
###importing
data = pd.read_csv('DSI_kickstarterscrape_dataset.csv')

###grabbing necessary data, removing null values
#proj = data.loc[:, "project id"]
pledge = data.loc[:,"pledged"]
back = data.loc[:, "backers"]
data.dropna(subset=['pledged'], inplace=True)
#nullCheck = pd.isnull(pledge).values.sum()
##This dropped out all null values
#print(len(data))

###changing pledged to int from float
df = pd.DataFrame(data)
df['category']=df['category'].replace(['Film &amp; Video'], 'Film & Video')
df1 = df['pledged'] = df['pledged'].astype("int64")
df2 = df['backers'] = df['backers'].astype("int64")
df3 = df['duration'] = df['duration'].astype("int64")
filt_backers_df = df2[((df2 -df2.mean()) / df.backers.std()).abs() < .085]
###Mean calc
def m(dataset):
    return sum(dataset) / len(dataset)
#print(m(df1))
#4980.749678963979 is the mean, provided that we did not include duplicates

#this creates a pivot table, but was not sufficient for proper data handling, and was not used to create visualizations.
#c_backers = df.pivot_table(columns=['backers'], aggfunc='size')
#print(min(df2), max(df2))
#frequency = df['backers'].value_counts()
#print(frequency)


df.info()
#print(df['funded date'])

df['Funding by Month'] = df['funded date'].apply(lambda x: dt.datetime.strptime(x, '%a, %d %b %Y %X -%f').strftime('%B'))
df['Funding by Day'] = df['funded date'].apply(lambda x: dt.datetime.strptime(x, '%a, %d %b %Y %X -%f').strftime('%a'))


#When looking at success to failures, this was the method used to properly handle the data viz.
#status_bool = {'successful': True, 'failed': False, 'suspended': False, 'canceled': False}
#df['status_bool'] = df['status'].map(status_bool)
#print(df['status_bool'].count(0))

#print(df['category'].unique().tolist())
#While the binary of success/'other' was usable, I began exploring further to see if it could be broken down for better case by case basis.
s_n = {'successful': 0, 'failed': 1, 'suspended': 2, 'canceled': 3, 'live': 4}
df['s_n'] =df['status'].map(s_n)
#ds_arr = np.array(df['s_n'])
#print(ds_arr.count(0))

frequency = df['s_n'].value_counts()
print(frequency)

##Checking normality.
#dn_test = np.array(df['duration'])
#k,p = stats.mstats.normaltest(dn_test)
#if p < 0.05:
#    print('Pvalue =', p, '\nZscore=', k, '\nDistribuation non-normal')
#else:
#    print('Pvalue =', p, '\nZscore=', k, '\nDistrbution normal')
##

###Check for duplicate projects prior to mean
#print(proj.duplicated(keep = False))

#Playing with status,
#s_n = {'successful': 0, 'failed': 1, 'suspended': 2, 'canceled': 3, 'live': 4}
#df['s_n'] = df['status'].map(s_n)

##Checking distribution of duration
#plt.hist(df['duration'], density= True)
#plt.xlabel('Time')
#plt.ylabel('Projects')
#plt.title('Duration')
#sns.kdeplot(df['duration'])
#plt.title('Duration')

##Creating a plot of backers
#plt.hist(filt_backers_df['backers'])
#plt.xlabel('Backers')
#plt.ylabel('Freq of Projects')
#plt.title('Plotting # of Backers')
##Histogram, minus the outliers
#plt.hist(filt_backers_df)
#plt.xlabel('Backers')
#plt.ylabel('Freq of Projects')
#plt.title('Removing outliers - Backers')

##Creating a boxplot for better viewing
#sns.kdeplot(filt_backers_df)
#plt.xlabel('Backers')
#plt.title('Project Backers - Removed Outliers')

##Slides for presentation
#Pie Chart
plt.pie(frequency)
plt.ylabel('Status')
plt.title('Project Status')
plt.legend(df['status'],loc=3)
#print(df['s_n'].count(0))
#print(df['status'].count('successful'))
#plt.pie(df['status_bool'])
#plt.ylabel('Status')
#plt.title('Project Status')



#Pledges to total Duration
#sns.factorplot(x='status', y='goal', data=df, kind='bar', ci=None)
#plt.xlabel('Success')
#plt.ylabel('Goals')
#plt.title('Goal to Success')

#Pledges by Category
#sns.factorplot(x='category', y='pledged', data=df, kind='bar', ci=None)
#plt.xlabel('Category of Kickstarters')
#plt.ylabel('Pledges')
#plt.title('Breakdown by Category')

#sns.factorplot(x='Funding by Month', y='status_bool', data=df, kind='bar', ci=None)
#plt.xlabel('Funded On:')
#plt.ylabel('Success')
plt.show()