import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import sys

data = pd.read_csv('GEFCOM.txt', delimiter="\s+", index_col=False, header=None, names=['YYYYMMDD', 'HH', 'zonal_price', 'system load', 'zonal_load', 'day-of-the-week'])
data_np = np.loadtxt('GEFCOM.txt', delimiter='\t', usecols=list(range(6)))
data['YYYYMMDD'] = data['YYYYMMDD'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
indicies = []
for i in data.index:
    indicies.append(datetime.strftime(data.loc[i, 'YYYYMMDD'],'%d/%m %Y'))

fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
axs[0].plot(data.zonal_price)
axs[0].set_xticks(ticks=[0, 4344, 8760, 13128, 17544, 21888])
axs[0].set_xticklabels(labels=['01/01 2011', '01/07 2011', '01/01 2012', '01/07 2012', '01/01 2013', '01/07 2013'])
axs[0].tick_params(axis='x', labelbottom=True)
axs[1].plot(data.zonal_load)
axs[1].set_xticks(ticks=[0, 4344, 8760, 13128, 17544, 21888])
axs[1].set_xticklabels(labels=['01/01 2011', '01/07 2011', '01/01 2012', '01/07 2012', '01/01 2013', '01/07 2013'])

plt.show()
#print(data)

data['naive'] = data['zonal_price'].shift(24)
#print(data)

#train_data = data[data['YYYYMMDD'] < '2011-06-30']
#test_data = data[data['YYYYMMDD'] >= '2011-06-30']

#longer calibration window
train_data = data[data['YYYYMMDD'] < '2013-01-01']
test_data = data[data['YYYYMMDD'] >= '2013-01-01']

#With longer calibration window the MAE for these two approaches are lower.

days_train = len(train_data)
days_test = len(test_data)
print(days_train)
print(days_test)
real_train = data_np[:days_train, 2]
real_test = data_np[-days_test:, 2]

#seperate hours
ar1 = np.zeros((int(days_test/24), 24))
days_train_ones = int(days_train / 24 - 1)
print(days_train_ones)
days_test_ones = int(days_test/24)
print(days_test_ones)

for hour in range(24):
    # y - labels for training (dependent variable)
    # x - inputs for training (independent variables)
    # xf - inputs for the test
    y = real_train[hour::24]
    #print(len(y))
    x = np.stack([np.ones((days_train_ones,)), y[:-1]])
    xf = np.stack([np.ones((days_test_ones,)), data_np[hour::24, 2][days_train_ones:-1]])
    y = y[1:]
    betas = np.linalg.lstsq(x.T, y, rcond=None)[0]
    pred = np.dot(betas, xf)
    ar1[:, hour] = pred
    #print([hour, np.mean(np.abs(pred - real_test[hour::24])), betas]) # added row

ar1 = np.reshape(ar1, (ar1.shape[0] * ar1.shape[1], ))
print(['Seperate hours MAE', np.mean(np.abs(ar1 - real_test))])

#single model
'''
y = real_train
x = np.stack([np.ones((len(y),)), np.arange(len(y))])
# GM np.arange here? Matrix x should contain data from y, like in line 58.
xf = np.stack([np.ones((days_test,)), np.arange(len(y), len(y) + days_test)])
y = y.reshape(-1, 1)
betas = np.linalg.lstsq(x.T, y, rcond=None)[0]
pred = np.dot(xf.T, betas)
ar2 = pred.flatten()


ar2 = np.zeros((int(days_test/24), 24))
for hour in range(24):
    # y - labels for training (dependent variable)
    # x - inputs for training (independent variables)
    # xf - inputs for the test
    y = real_train[hour::24]
    #print(len(y))
    x = np.stack([np.ones((days_train_ones-6,)), y[6:-1]])
    xf = np.stack([np.ones((days_test_ones,)), data_np[hour::24, 2][days_train_ones:-1]])
    y = y[7:]
    betas = np.linalg.lstsq(x.T, y, rcond=None)[0]
    pred = np.dot(betas, xf)
    ar2[:, hour] = pred
    #print([hour, np.mean(np.abs(pred - real_test[hour::24])), betas]) # added row

ar2 = np.reshape(ar2, (ar2.shape[0] * ar2.shape[1], ))
'''

# GM
real_test = test_data['zonal_price'].to_numpy() # true labels for test data
ar2 = np.zeros_like(real_test) # predefine output matrix
Ndays = len(ar2) // 24 #divide
firstind = 24 # the first index for our y_train (we omit the first day, as we can't create a training sample
lastind = np.argwhere(data_np[:, 0] == 20130101).squeeze()[0] # we need to select the first one
for day in range(Ndays):
    y = data_np[firstind + 24*day:lastind + 24*day, 2] # zonal price is column with index 2
    x = data_np[firstind + 24*(day-1):lastind + 24*day, 2] # y shifted by 1 day back, longer by 1 day (we will use the added samples for xf)
    x = np.stack([x, np.ones_like(x)]).T # intercept added, .T means transpose to matrix
    xf = x[-24:, :] # extracting the xf, selects the last 24 samples of x, which will be used as the input to predict the next day's prices
    x = x[:-24, :] # remove xf from x, removes the last 24 samples of x, which will be used to train the model
    for h in range(24): #loop iterates over each hour in the day
        betas = np.linalg.lstsq(x[h::24], y[h::24], rcond=None)[0] # hourly data selected here
        '''x[h::24] selects every 24th element of x, starting from index h. 
        This creates a new array containing every h-th hourly zonal price in x,
         starting from the h-th hour of the day.'''
        ar2[day*24 + h] = np.dot(xf[h::24], betas)
print(['Single model MAE', np.mean(np.abs(ar2 - real_test))])

#rolling calibration window
real_test = test_data['zonal_price'].values
ar2 = np.zeros((int(days_test/24), 24))

for hour in range(24):
    for i in range(days_test_ones):
        # y - labels for training (dependent variable)
        # x - inputs for training (independent variables)
        # xf - inputs for the test
        train_data = train_data[train_data['HH'] == hour]
        test_datum = test_data.iloc[i]

        y = train_data['zonal_price'].values
        x = np.stack([np.ones((len(y),)), np.arange(len(y))])
        xf = np.array([1, len(y)]).reshape(-1, 1)
        y = y.reshape(-1, 1)
        betas = np.linalg.lstsq(x.T, y, rcond=None)[0]
        pred = np.dot(xf.T, betas)
        ar2[i, hour] = pred
        train_data = pd.concat([train_data, test_datum])

ar2 = np.reshape(ar2, (ar2.shape[0] * ar2.shape[1],))
print(['Separate hours rolling window MAE', np.mean(np.abs(ar2 - real_test))])

