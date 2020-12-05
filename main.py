import pandas as pd
import lightgbm as lgbm
import warnings
warnings.filterwarnings("ignore")

# read train data file and holiday table
data_df = pd.read_csv("train.csv") 
holiday_df = pd.read_csv("holiday.csv") 

#Data preprocessing for train data
data_df = data_df.set_index('id')
data_df['date'] = pd.to_datetime((data_df['date']),format='%d/%m/%Y %H:%M')
data_df['hour'] = pd.DatetimeIndex(data_df['date']).hour
data_df['year'] = pd.DatetimeIndex(data_df['date']).year
data_df['month'] = pd.DatetimeIndex(data_df['date']).month
data_df['day'] = pd.DatetimeIndex(data_df['date']).day
data_df['weekday'] = (data_df['date'].dt.dayofweek) 
data_df['workingday'] = (data_df['date'].dt.dayofweek<5).astype(int)


#Import weather condition data from https://www.worldweatheronline.com/
weather_df = pd.read_csv('hkweather.csv')
weather_df['date_time'] = pd.to_datetime((weather_df['date_time']),format='%Y-%m-%d')
weather_df = weather_df[['date_time','cloudcover','humidity', 'tempC','visibility','winddirDegree','windspeedKmph','WindChillC']]
weather_df['year'] =  pd.DatetimeIndex(weather_df['date_time']).year
weather_df['month'] = pd.DatetimeIndex(weather_df['date_time']).month
weather_df['day'] = pd.DatetimeIndex(weather_df['date_time']).day
weather_df = weather_df.drop(columns=['date_time'], axis=1);

# join train data with holiday table
holiday_df['holiday'] = 1
data_df = pd.merge(data_df, holiday_df, how='left', on=['year', 'month','day'])
data_df = pd.merge(data_df, weather_df, how='left', on=['year', 'month','day'])
data_df['holiday'] = data_df['holiday'].fillna(0)

#Deal with cat variables
dummies_month = pd.get_dummies(data_df['month'], prefix= 'month')
dummies_weekday=pd.get_dummies(data_df['weekday'],prefix='weekday')
data_df=pd.concat([data_df,dummies_month,dummies_weekday],axis=1)


#Drop unnecessary variables
x = data_df.drop(columns=['speed','date','weekday','day','month'], axis=1)
y = data_df['speed']

# set parameters and build model
param = {'max_depth': 12, 'num_trees':200, 'num_leaves': 110, 'objective':'regression', 'learning_rate':0.1}
data_train_lgbm = lgbm.Dataset(x, y)
model = lgbm.train(param, data_train_lgbm)


# Get test data
test_df = pd.read_csv("test.csv")
submission = pd.read_csv('SampleSubmission.csv')
submission['id'] = test_df['id']

#Data preprocessing for test data
test_df = test_df.set_index('id')
test_df['date'] = pd.to_datetime((test_df['date']),format='%d/%m/%Y %H:%M')
test_df['hour'] = pd.DatetimeIndex(test_df['date']).hour
test_df['year'] = pd.DatetimeIndex(test_df['date']).year
test_df['month'] = pd.DatetimeIndex(test_df['date']).month
test_df['day'] = pd.DatetimeIndex(test_df['date']).day
test_df['weekday'] = (test_df['date'].dt.dayofweek) 
test_df['workingday'] = (test_df['date'].dt.dayofweek<5).astype(int)

# join test data with holiday table
holiday_df['holiday'] = 1
test_df = pd.merge(test_df, holiday_df, how='left', on=['year', 'month','day'])
test_df = pd.merge(test_df, weather_df, how='left', on=['year', 'month','day'])
test_df['holiday'] = test_df['holiday'].fillna(0)

# Dummies for cat variables
dummies_month = pd.get_dummies(test_df['month'], prefix= 'month')
dummies_weekday=pd.get_dummies(test_df['weekday'],prefix='weekday')
test_df=pd.concat([test_df,dummies_month,dummies_weekday],axis=1)

#Drop unnecessary variables
test_x = test_df.drop(columns=['date','weekday','day','month'], axis=1)

#predicting on the test set and creating submission file
predict = model.predict(test_x)
submission['speed'] = predict
submission.to_csv('result_final.csv',index=False)
