### libs

import os
import numpy as np
import pandas as pd
import datetime
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns

from datagen_functions import *

sns.set_style('darkgrid', {'axes.facecolor': '0.9'})


### Generate the transactions

(customer_profiles_table, terminal_profiles_table, transactions_df) =\
    generate_dataset(n_customers = 5000, 
                     n_terminals = 10000, 
                     nb_days = 365, 
                     start_date = "2021-01-01", 
                     r = 25)

# print(transactions_df.shape)
# print(transactions_df.head())

## Check average reachable terminals for the customer

# x_y_terminals = terminal_profiles_table[['x_terminal_id','y_terminal_id']].values.astype(float)
# customer_profiles_table['available_terminals'] = customer_profiles_table.apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=25), axis=1)

# print(customer_profiles_table['available_terminals'].apply(len).mean())

## plotting

distribution_amount_times_fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = transactions_df[transactions_df.TX_TIME_DAYS<10]['TX_AMOUNT'].sample(n=10000).values
time_val = transactions_df[transactions_df.TX_TIME_DAYS<10]['TX_TIME_SECONDS'].sample(n=10000).values

sns.distplot(amount_val, ax=ax[0], color='r', hist = True, kde = False)
ax[0].set_title('Distribution of transaction amounts', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])
ax[0].set(xlabel = "Amount",
          ylabel="Number of transactions")

# Divide the time variables by 86400 to transform seconds to days in the plot
sns.distplot(time_val/86400, ax=ax[1], color='b', bins = 100, hist = True, kde = False)
ax[1].set_title('Distribution of transaction times', fontsize=14)
ax[1].set_xlim([min(time_val/86400), max(time_val/86400)])
ax[1].set_xticks(range(10))
ax[1].set(xlabel = "Time (days)",
          ylabel="Number of transactions")



### Adding fraud

transactions_df = add_frauds(customer_profiles_table,
                             terminal_profiles_table,
                             transactions_df)

# print('Percentage: ', round(transactions_df.TX_FRAUD.mean(),3))
# print('Amount    : ', transactions_df.TX_FRAUD.sum())
# print(transactions_df.head())


### Get data state

(nb_tx_per_day,nb_fraud_per_day,nb_fraudcard_per_day) = get_stats(transactions_df)

n_days = len(nb_tx_per_day)
tx_stats = pd.DataFrame({"value":pd.concat([nb_tx_per_day/50,nb_fraud_per_day,nb_fraudcard_per_day])})
tx_stats['stat_type'] = ["nb_tx_per_day"]*n_days + ["nb_fraud_per_day"]*n_days + ["nb_fraudcard_per_day"] * n_days
tx_stats = tx_stats.reset_index()

## plotting

sns.set(style='darkgrid')
sns.set(font_scale=1.4)

fraud_and_transactions_stats_fig = plt.gcf()
fraud_and_transactions_stats_fig.set_size_inches(15, 8)

sns_plot = sns.lineplot(x="TX_TIME_DAYS",
                        y="value", data=tx_stats,
                        hue="stat_type",
                        hue_order=["nb_tx_per_day","nb_fraud_per_day","nb_fraudcard_per_day"], legend=False)
sns_plot.set_title('Total transactions, and number of fraudulent transactions \n and number of compromised cards per day', fontsize=20)
sns_plot.set(xlabel = "Number of days since beginning of data generation",
             ylabel="Number")
sns_plot.set_ylim([0,300])

labels_legend = ["# transactions per day (/50)", "# fraudulent txs per day", "# fraudulent cards per day"]
sns_plot.legend(loc='upper left', labels=labels_legend,bbox_to_anchor=(1.05, 1), fontsize=15)



### Adding datetime columns

transactions_df['TX_DURING_WEEKEND'] = transactions_df.TX_DATETIME.apply(is_weekend)

transactions_df['TX_DURING_NIGHT'] = transactions_df.TX_DATETIME.apply(is_night)
transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)

transactions_df = transactions_df.groupby('TERMINAL_ID').apply(lambda x: get_count_risk_rolling_window(x, delay_period=7, windows_size_in_days=[1,7,30], feature="TERMINAL_ID"))
transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)

# print(transactions_df.head())



### Output dataset

DIR_OUTPUT = '/dataset/'

if not os.path.exists(DIR_OUTPUT):
    os.makedirs(DIR_OUTPUT)

start_date = datetime.datetime.strptime("2021-01-01", "%Y-%m-%d")

for day in range(transactions_df.TX_TIME_DAYS.max()+1):
    
    transactions_day = transactions_df[transactions_df.TX_TIME_DAYS==day].sort_values('TX_TIME_SECONDS')
    
    date = start_date + datetime.timedelta(days=day)
    filename_output = date.strftime("%Y-%m-%d")+'.pkl'
    
    transactions_day.to_pickle(DIR_OUTPUT+filename_output)