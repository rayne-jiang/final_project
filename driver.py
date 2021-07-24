import pandas as pd
from datetime import datetime
df = pd.read_csv('June_2019_D28 Driver Timesheet Report .csv')

df['End Of Duty'] = pd.to_datetime(df['End Of Duty'], format='%d/%m/%Y %H:%M')
df['StartOfDuty'] = pd.to_datetime(df['StartOfDuty'], format='%d/%m/%Y %H:%M')

df['working_hour'] = df['End Of Duty'] - df['StartOfDuty']

# 04:
Category = df['Category'].unique()
map_category = {'04, Shunting Staff': 4, '01, Full Time Drivers':1,'07, National Express Drivers':7, '17, Engineering Staff': 17,'16, Office Staff':16, '05, Casual Drivers':5 }
df['Category_number'] = df['Category'].map(map_category)

# deal with nan number in Columns "Drive, Other, POA, Rest"
drive_col = ["Drive", "Other", "POA", "Rest"]

for col in drive_col:
    # delete outlier
    df = df.drop(df[df[col] > '24:00:00'].index)
    df[col] = df[col].fillna('00:00')

    # df[col] = df[col].str.split(':')



# record_hour = []
# for index, row in df.iterrows():
#     hour = 0
#     minute = 0
#     for col in drive_col:
#         col_hour, col_minute = df[col].str.split(':')
#         col_hour
# # df['Record_Hour'] = df['Drive'] + df['Other'] + df['POA'] + df['Rest']
# # df



# select full time driver

full_time_driver = df[df['Category_number']==1]