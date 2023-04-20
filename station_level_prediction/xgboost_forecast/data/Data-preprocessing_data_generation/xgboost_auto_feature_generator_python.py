import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import sys
import os
import tsfresh as tsf

os.chdir('/home/optimusprime/Desktop/peeterson/github/DeepAR_demand_prediction/station_level_prediction/xgboost_forecast/data/demand_data')
raw_ds = pd.read_csv("Xgboost_raw_dataset.csv")

raw_ds


"""
TALLY the outflow
"""

# outflow
end_60 = np.where(raw_ds["60_min_clstr_outflow"].isnull() == True)[0][0] ## remove NAN values
end_10 = len(raw_ds) #np.where(raw_ds["10_min_outf_130"].isnull() == True)[0][0]

end = np.minimum(6*end_60,end_10)
end_60 = int(np.floor(end/6))
end_10 = end_60*6
print("end_10",end_10)
end_10_outf = end_10 

chosen_clstr_outf = raw_ds.loc[:end_60-1]["60_min_clstr_outflow"]
chosen_clstr_outf
tot_idx_10_min_outflow = end_10
tot_idx_10_min_outflow

ten_min_stn_outf = raw_ds.loc[:tot_idx_10_min_outflow-1][["10_min_outf_130","10_min_outf_184","10_min_outf_293","10_min_outf_350" ]]
tot_idx_10_min_outflow


ten_min_stn_outf # all staions demand df
agg_stns_outf = [] # aggregated staitons

for i in range(0,tot_idx_10_min_outflow-4,6): # max = 13038
    tot = sum(ten_min_stn_outf.loc[i:i+5].sum())
    agg_stns_outf.append(tot)

min_len = np.minimum(len(agg_stns_outf),len(chosen_clstr_outf))
print(min_len)

agg_stats = np.array(agg_stns_outf[:min_len])
chosen_clstr_outf = np.array(chosen_clstr_outf[:min_len])

# plt.plot(agg_stats-chosen_clstr_outf)
# plt.show()
print(np.where(agg_stats-chosen_clstr_outf<0))
print(np.where(agg_stats-chosen_clstr_outf>0))



"""
TALLY the inflow
"""

# inflow
end_60 = np.where(raw_ds["60_min_clstr_inflow"].isnull() == True)[0][0] ## remove NAN values
end_10 = len(raw_ds) #np.where(raw_ds["10_min_inf_130"].isnull() == True)[0][0] 

end = np.minimum(6*end_60,end_10)
end_60 = int(np.floor(end/6))
end_10 = end_60*6
print("end_10",end_10)
end_10_inf = end_10

chosen_clstr_inf = raw_ds.loc[:end_60-1]["60_min_clstr_inflow"]
chosen_clstr_inf
tot_idx_10_min_inflow = end_10
tot_idx_10_min_inflow

ten_min_stn_inf = raw_ds.loc[:tot_idx_10_min_inflow-1][["10_min_inf_130","10_min_inf_184","10_min_inf_293","10_min_inf_350" ]]
tot_idx_10_min_inflow


ten_min_stn_inf # all staions demand df
agg_stns_inf = [] # aggregated staitons

for i in range(0,tot_idx_10_min_inflow-4,6): # max = 13038
    tot = sum(ten_min_stn_inf.loc[i:i+5].sum())
    agg_stns_inf.append(tot)

min_len = np.minimum(len(agg_stns_inf),len(chosen_clstr_inf))
print(min_len)

agg_stats = np.array(agg_stns_inf[:min_len])
chosen_clstr_inf = np.array(chosen_clstr_inf[:min_len])

# plt.plot(agg_stats-chosen_clstr_inf)
# plt.show()
print(np.where(agg_stats-chosen_clstr_inf<0))
print(np.where(agg_stats-chosen_clstr_inf>0))



"""""""""
Feature Generation
"""""""""

xgboost_features_df = pd.DataFrame(columns=["dt_ts","stn_id","rem_blk_outf","net_inflow_stn","en_route_inf","net_inflow_clstr","DeepAR_agg_outflow",
                                                                   "p_1wk_o","p_2wk_o","p_3wk_o", "block_id","ts_of_day", "hr_of_day", "day_of_wk",
                                                                     "day_of_mn", "wk_of_mon"])


"""

Feature description


# dt_ts : date time and time slot. date is y:m:d time is hour. ts is time slot 0-5 for 6 10-min time slots.
# stn_id : is station number 
# rem_blk_outf : remaining allowed demand. DeepAR_agg_outflow minus the sum of the blocks outflow to one block before present block in the current hour.
# net_inflow_stn : sum of last one days (144 time slots) to current time slot inflow/supply minus the sum of 
#                  last one day to current time slot outflow/demand in current station 
# DeepAR_agg_outflow : Hourly deepar prediction.
# p_1wk_o : previous 1_wk to time slot. seasonal demand 1 week prior to current time slot
# p_2wk_o : previous 2_wk to time slot.
# p_3wk_o : previous 3_wk to time slot.
# block_id : block number, 0 - (num_stations*6 -1). each hour has same block indices/id. combines station and time together as a crossed feature
# en_route_inf : en_route inflow is current supply value at that block
# net_inflow_clstr : how much inflow at cluster level, sum of last one day (24 hours) to current hour inflow/supply minus the sum of 
# last two days to current hour outflow/demand in current cluster, also called 'net_inflow_clstr_10_min' 
# ts_of_day : time slot of day (0-5) 6*10 mins = 1 hour
# hr_of_day : hour of day
# day_of_wk : day of week
# day_of_mn : day of month
# wk_of_mon : week of month
# p_1ts_o : outflow from previous 1 time slot's 
# p_2ts_o : outflow from previous 2 time slot's
# p_3ts_o : outflow from previous 3 time slot's

'dt_ts','stn_id','rem_blk_outf','net_inflow_stn','en_route_inf','net_inflow_clstr_10_min','DeepAR_agg_outflow','p_1wk_o','p_2wk_o','p_3wk_o','block_id','ts_of_day','hr_of_day','day_of_wk','day_of_mn','wk_of_mon','p_1ts_o','p_2ts_o','p_3ts_o'
"""


"""
TARGET Description

# target: 10 min station-level outflow

"""

# create the dataframe of features for each station first and then concatenate them below each other.
# Finally use pivot table to create 3d table.




"""
Identify the stations,stn and generate dt_ts
"""

############ Identify stations ############
all_stations = raw_ds.columns
all_stations

ten_min_stns_outf_list = [s for s in all_stations if s[:11] == "10_min_outf"]
ten_min_stns_outf_list

ten_min_stns_inf_list = [s for s in all_stations if s[:10] == "10_min_inf"]
ten_min_stns_inf_list

stn_list = [ s[12:] for s in ten_min_stns_outf_list]
stn_list
############ Identify stations ############



############ generate dt_ts ############
# ensure that both inflow and outflow start and end at the same time
if (end_10_inf == end_10_outf):
    print("10 min inflow and 10 min outflow of same length")
    end_len_10 = end_10_inf # end length
    end_len_60 = int(end_len_10/6)
else:
    print("10 min inflow and 10 min outflow of DIFFERENT length")
    sys.exit()

# specify start date and time:
date = "2021-09-24"
hr = "00" 
min = "00" # 00, 10, 20, 30, 40, 50 represents the timeslots 0,1,2,3,4,5
ts = "0"
string_dt_hr = date + "-" + hr + "-" + min

dt_hr = dt.datetime.strptime(string_dt_hr,"%Y-%m-%d-%H-%M")

dt_dt_ts = []
dt_dt_ts.append(dt_hr)

# create list of datetimes
for _ in range(end_len_10-1):
    dt_hr = dt_hr + dt.timedelta(minutes=10 )
    dt_dt_ts.append(dt_hr)
dt_dt_ts

# convert to string
dt_ts_list = [] 
for i in range(len(dt_dt_ts)):
    dt_ts_list.append( dt.datetime.strftime(dt_dt_ts[i],"%Y/%m/%d %H:%M") )
#dt_ts_list

############ generate dt_ts ############




"""
######################## Custom Feature generation functions ########################

"""

def create_DeepAR_agg_outflow(dt_ts_list,stn_list):

    col_list = [ "DeepAR_agg_outflow_"+s for s in stn_list] 
    DeepAR_agg_outflow_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    print(len(raw_ds.loc[:end_len_60-1]["60_min_clstr_outflow"].to_numpy()))

    for s in stn_list:
        DeepAR_agg_outflow_df["DeepAR_agg_outflow_"+s] = np.repeat(  raw_ds.loc[:end_len_60-1]["60_min_clstr_outflow"].to_numpy() ,repeats=6)

    print(len(dt_ts_list))

    return DeepAR_agg_outflow_df

df5 = create_DeepAR_agg_outflow(dt_ts_list,stn_list)
df5


def create_rem_blk_outf(dt_ts_list,stn_list):
    
    col_list = [ "rem_blk_outf_"+s for s in stn_list] 
    rem_blk_outf_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    i=0
    for dt in dt_ts_list:
        if (dt[-2:] == "00"):
            sum = 0
        for s in stn_list:
            rem_blk_outf_df.loc[i]["rem_blk_outf_"+s] =  df5.loc[i]["DeepAR_agg_outflow_"+s] - sum 
            sum = sum + raw_ds.loc[i]["10_min_outf_"+s]

        i+=1

    return rem_blk_outf_df

df1 = create_rem_blk_outf(dt_ts_list,stn_list)
df1


def create_net_inflow_stn(dt_ts_list,stn_list):

    col_list = [ "net_inflow_stn_"+s for s in stn_list] 
    net_inflow_stn_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    past_days = 0.2 # decimal
    ten_min_slots = int(past_days*24*6)
    for s in stn_list:
        i=ten_min_slots
        for dt in dt_ts_list[ten_min_slots:]:
            net_inflow_stn_df.loc[i]["net_inflow_stn_"+s] = sum(raw_ds.loc[i-ten_min_slots:i-1]["10_min_inf_"+s]) - sum(raw_ds.loc[i-ten_min_slots:i-1]["10_min_outf_"+s])
            i+=1

    return net_inflow_stn_df

df2 = create_net_inflow_stn(dt_ts_list,stn_list)
df2


print("minimum net_inflow_stn_xx in df = ",np.min(df2.min().values))
print("maximum net_inflow_stn_xx in df = ",np.max(df2.max().values))


def create_en_route_inf(dt_ts_list,stn_list):

    col_list = [ "en_route_inf_"+s for s in stn_list] 
    en_route_inf_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    i=0
    for dt in dt_ts_list:
        for s in stn_list:
            en_route_inf_df.loc[i]["en_route_inf_"+s] = raw_ds.loc[i]["10_min_inf_"+s]
        
        i+=1

    return en_route_inf_df

df3 = create_en_route_inf(dt_ts_list,stn_list)
df3


def create_net_inflow_clstr(dt_ts_list,stn_list):

    col_list = [ "net_inflow_clstr_"+s for s in stn_list] 
    net_inflow_clstr_60 = pd.DataFrame(index= range(end_len_60), columns=col_list)
    col_list = [ "net_inflow_clstr_10_min_"+s for s in stn_list] 
    net_inflow_clstr_10 = pd.DataFrame(index= range(end_len_10), columns=col_list)

    past_hours = 24
    i=past_hours
    for hr in range(past_hours,end_len_60,1):
        tot_sum = sum(raw_ds.loc[i-past_hours:i-1]["60_min_clstr_inflow"]) - sum(raw_ds.loc[i-past_hours:i-1]["60_min_clstr_outflow"]) 
        for s in stn_list:
            net_inflow_clstr_60.loc[i]["net_inflow_clstr_"+s] = tot_sum
        i+=1

    print(len(net_inflow_clstr_60["net_inflow_clstr_130"].to_numpy()))

    for s in stn_list:
        net_inflow_clstr_10["net_inflow_clstr_10_min_"+s] = np.repeat(  net_inflow_clstr_60["net_inflow_clstr_"+s].to_numpy() ,repeats=6)

    print(len(dt_ts_list))

    return net_inflow_clstr_10

df4 = create_net_inflow_clstr(dt_ts_list,stn_list)
df4


print("minimum net_inflow_clstr in df = ",np.min(df4.min().values[1:]))
print("maximum net_inflow_clstr in df = ",np.max(df4.max().values[1:]))


def create_prev_weeks_outflow(dt_ts_list,stn_list):
    
    col_list = [ "p_1wk_o_"+s for s in stn_list] + [ "p_2wk_o_"+s for s in stn_list] + [ "p_3wk_o_"+s for s in stn_list]
    prev_weeks_outflow_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    weekly_lags = [1,2,3] # lags in weeks

    for wk in weekly_lags:
        for s in stn_list:
            prev_weeks_outflow_df["p_"+str(wk)+"wk_o_"+s] = raw_ds["10_min_outf_"+s].shift(wk*7*24*6)

    return prev_weeks_outflow_df

df6 = create_prev_weeks_outflow(dt_ts_list,stn_list)
df6


def create_prev_ts_outflow(dt_ts_list,stn_list):

    col_list = [ "p_1ts_o_"+s for s in stn_list] + [ "p_2ts_o_"+s for s in stn_list] + [ "p_3ts_o_"+s for s in stn_list]
    prev_ts_outflow_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    ts_lags = [1,2,3] # lags in 10-min time slots (ts)

    for ts in ts_lags:
        for s in stn_list:
            prev_ts_outflow_df["p_"+str(ts)+"ts_o_"+s] = raw_ds["10_min_outf_"+s].shift(ts)

    return prev_ts_outflow_df

df13 = create_prev_ts_outflow(dt_ts_list,stn_list)
df13


def create_block_id(dt_ts_list,stn_list):
    
    col_list = [ "block_id_"+s for s in stn_list] 
    block_id_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    i=0
    for dt in dt_ts_list:
        if (dt[-2:] == "00"):
            id = 0
        for s in stn_list:
            block_id_df.loc[i]["block_id_"+s] = id
            id = id + 1
        
        i+=1

    return block_id_df

df7 = create_block_id(dt_ts_list,stn_list)
df7


def create_ts_of_day(dt_ts_list,stn_list):

    col_list = [ "ts_of_day_"+s for s in stn_list] 
    ts_of_day_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    i=0
    for dt in dt_ts_list:
        for s in stn_list:
            ts_of_day_df.loc[i]["ts_of_day_"+s] = int(int(dt[-2:])/10)

        i+=1

    return ts_of_day_df

df8 = create_ts_of_day(dt_ts_list,stn_list)
df8


def create_hr_of_day(dt_ts_list,stn_list):

    col_list = [ "hr_of_day_"+s for s in stn_list] 
    hr_of_day_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    for s in stn_list:
        hr_of_day_df["hr_of_day_"+s] = pd.to_datetime(dt_ts_list).hour

    return hr_of_day_df

df9 = create_hr_of_day(dt_ts_list,stn_list)
df9


def create_day_of_wk(dt_ts_list,stn_list):

    col_list = [ "day_of_wk_"+s for s in stn_list] 
    day_of_wk_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    DayOfWeek = pd.to_datetime(dt_ts_list).day_of_week

    for s in stn_list:
        day_of_wk_df["day_of_wk_"+s] = DayOfWeek
        day_of_wk_df["is_weekend_"+s] = day_of_wk_df["day_of_wk_"+s].apply(lambda x: 1 if x > 4 else 0)

    return day_of_wk_df

df10 = create_day_of_wk(dt_ts_list,stn_list)
df10


def create_day_of_mn(dt_ts_list,stn_list):

    col_list = [ "day_of_mn_"+s for s in stn_list] 
    day_of_mn_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    for s in stn_list:
        day_of_mn_df["day_of_mn_"+s] = pd.to_datetime(dt_ts_list).day

    return day_of_mn_df

df11 = create_day_of_mn(dt_ts_list,stn_list)
df11



def create_wk_of_mon(dt_ts_list,stn_list):

    col_list = [ "wk_of_mon_"+s for s in stn_list] 
    wk_of_mon_df = pd.DataFrame(index= range(end_len_10), columns=col_list)
    wk_of_mon_df["dt_ts"] = dt_ts_list
    
    for s in stn_list:
        wk_of_mon_df["wk_of_mon_"+s] = wk_of_mon_df["dt_ts"].apply(lambda d: (int(d[8:10])-1) // 7 + 1) 

    wk_of_mon_df.drop(columns=["dt_ts"],inplace=True)
    return wk_of_mon_df

df12 = create_wk_of_mon(dt_ts_list,stn_list)
df12


def create_future_inflow_ts(stn_list):

    col_list = [ "next_1ts_inf_"+s for s in stn_list]# + [ "next_2ts_inf_"+s for s in stn_list] + [ "next_3ts_inf_"+s for s in stn_list]
    next_ts_inflow_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    ts_leads = [1,2,3]#[1,2,3] # leads in 10-min time slots (ts)

    for ts in ts_leads:
        for s in stn_list:
            next_ts_inflow_df["next_"+str(ts)+"ts_inf_"+s] = raw_ds["10_min_inf_"+s].shift(-1*ts)

    shave_last_nrows = ts_leads[-1] # remove nan values at the end of df

    return next_ts_inflow_df, shave_last_nrows



df14,shave_last_nrows = create_future_inflow_ts(stn_list)
df14



def create_difference_ts(stn_list):

    col_list = [ "outf_10m_dif1_per1_"+s for s in stn_list] + [ "outf_10m_dif1_per2_"+s for s in stn_list] + [ "outf_10m_dif2_per1_"+s for s in stn_list] \
                + [ "inf_10m_dif1_per1_"+s for s in stn_list] + [ "inf_10m_dif1_per2_"+s for s in stn_list] + [ "inf_10m_dif2_per1_"+s for s in stn_list]
    diff_ts_df = pd.DataFrame(index= range(end_len_10), columns=col_list)

    diff1 = [1,2] # periods of 1st differencing

    for p in diff1:
        for s in stn_list:
            diff_ts_df["outf_10m_dif1_per"+str(p)+"_"+s] = raw_ds["10_min_outf_"+s].diff(periods=p)
            diff_ts_df["inf_10m_dif1_per"+str(p)+"_"+s] = raw_ds["10_min_inf_"+s].diff(periods=p)

    diff2 = [1] # periods of 2nd differencing

    for p in diff2:
        for s in stn_list:
            diff_ts_df["outf_10m_dif2_per"+str(p)+"_"+s] = raw_ds["10_min_outf_"+s].diff(periods=p).diff()
            diff_ts_df["inf_10m_dif2_per"+str(p)+"_"+s] = raw_ds["10_min_inf_"+s].diff(periods=p).diff()

    return diff_ts_df


df15 = create_difference_ts(stn_list)
df15

"""
######################## ######################## ########################

"""


"""
a##############################  add Target ############################
"""

df_list = [df1, df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15]
# df_list1 = []
# df_list1.append(df1)

# for d in df_list[1:]:
#     df_list1.append(d)

target_df = pd.DataFrame()
col_list = ten_min_stn_outf.columns
for s,c in zip(stn_list,col_list):
    target_df["target_"+s] = ten_min_stn_outf[c]

df_list.append(target_df)

full_ds = pd.concat(df_list,axis=1)
full_ds.insert(0,'dt_ts',pd.to_datetime(dt_ts_list))

full_ds # full dataset


"""
Remove NaN elements in xgboost_feat_train_ds
"""

full_ds = full_ds.dropna().reset_index(drop=True)
full_ds


"""
seperate the features dataset into list of stations.
"""

col_list = full_ds.columns

df_list = [0]*len(stn_list)

i=0
for s in stn_list:
    df_list[i] = pd.DataFrame(index=range(len(full_ds)))
    df_list[i] = pd.concat([df_list[i], full_ds[col_list[0]]],axis=1) # add dt_ts
    #print(df_list[i])
    for c in col_list:
        c_id = c.split("_")
        if c_id[-1] == s:
            df_list[i] = pd.concat([df_list[i], full_ds[c]],axis=1)
    i+=1



"""
put each station's time series below each other (stack) by adding station id 
"""

col_list = ['dt_ts','stn_id','rem_blk_outf','net_inflow_stn','en_route_inf','net_inflow_clstr_10_min','DeepAR_agg_outflow','p_1wk_o','p_2wk_o','p_3wk_o','block_id','ts_of_day','hr_of_day','day_of_wk','is_weekend','day_of_mn','wk_of_mon','p_1ts_o','p_2ts_o','p_3ts_o','next_1ts_inf','next_2ts_inf','next_3ts_inf','inf_10m_dif1_per1','inf_10m_dif1_per2','inf_10m_dif2_per1','outf_10m_dif1_per1','outf_10m_dif1_per2','outf_10m_dif2_per1','target']

all_stns_df = pd.DataFrame(columns=col_list)

for i,s in enumerate(stn_list):
    df_list[i].insert(1,'stn_id', np.tile( s, df_list[i].shape[0]) )
    df_list[i].columns = all_stns_df.columns
    all_stns_df = all_stns_df.append(df_list[i],ignore_index=True)

all_stns_df


"""
convert from object type to float64

"""

all_stns_df.stn_id = all_stns_df.stn_id.astype('float64')
all_stns_df.rem_blk_outf = all_stns_df.rem_blk_outf.astype('float64')
all_stns_df.net_inflow_stn = all_stns_df.net_inflow_stn.astype('float64')
all_stns_df.en_route_inf = all_stns_df.en_route_inf.astype('float64')
all_stns_df.net_inflow_clstr_10_min = all_stns_df.net_inflow_clstr_10_min.astype('float64')
all_stns_df.block_id = all_stns_df.block_id.astype('float64')
all_stns_df.ts_of_day = all_stns_df.ts_of_day.astype('float64')
all_stns_df.hr_of_day = all_stns_df.hr_of_day.astype('float64')
all_stns_df.day_of_wk = all_stns_df.day_of_wk.astype('float64')
all_stns_df.is_weekend = all_stns_df.is_weekend.astype('float64')

all_stns_df.day_of_mn = all_stns_df.day_of_mn.astype('float64')
all_stns_df.wk_of_mon = all_stns_df.wk_of_mon.astype('float64')
all_stns_df.target = all_stns_df.target.astype('float64')

all_stns_df.dtypes

print('all_stns_df.shape',all_stns_df.shape)




print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
"""
Add tsfresh features to original features and save as csv
"""

#numeric_cols =  all_stns_df.drop(columns=["block_id","ts_of_day","hr_of_day","day_of_wk","wk_of_mon","is_weekend","day_of_mn"])

win_lengths = [6*24*7]

for w in win_lengths:
    df_rolled = tsf.utilities.dataframe_functions.roll_time_series(all_stns_df,column_id='stn_id',column_sort='dt_ts',min_timeshift=w,max_timeshift=w)
    #df_rolled.drop(columns=['year', 'month', 'day', 'hour', 'wd', 'station'], inplace=True)
    # drop all categorical columns
    df_rolled.drop(columns=['stn_id', 'block_id', 'ts_of_day', 'hr_of_day', 'day_of_wk', 'is_weekend','day_of_mn','wk_of_mon'], inplace=True)
    df_features = tsf.extract_features(df_rolled, column_id='id', column_sort='dt_ts', 
                                        default_fc_parameters=tsf.feature_extraction.MinimalFCParameters())
    df_features.columns

all_stns_df = pd.concat(all_stns_df,df_features,axis=1)

os.chdir('/home/optimusprime/Desktop/peeterson/github/DeepAR_demand_prediction/station_level_prediction/xgboost_forecast/data/demand_data')
all_stns_df.to_csv("xgboost_extra_feat_train_ds_all_stns.csv")



print(all_stns_df.dtypes)









