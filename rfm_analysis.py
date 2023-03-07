
###############################################################
# Customer Segmentation with RFM
###############################################################

###############################################################
# Business Problem
###############################################################
# FLO wants to segment its customers and determine marketing strategies according to these segments.
# For this, the behavior of the customers will be defined and groups will be formed according to these behavior clusters.

###############################################################
# Dataset Story
###############################################################

# The dataset consists of information obtained from the past shopping behaviors of customers who made their last
# purchases as OmniChannel (both online and offline shopper) in 2020 - 2021.

# master_id: Unique client number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the last purchase was made
# first_order_date : The date of the customer's first purchase
# last_order_date : The date of the last purchase made by the customer
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : The date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : The total price paid by the customer for offline purchases
# customer_value_total_ever_online : The total price paid by the customer for their online shopping
# interested_in_categories_12 : List of categories the customer has purchased from in the last 12 months

###############################################################
# TASKS
###############################################################

import pandas as pd
import numpy as np
import datetime as dt

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500)


###############################################################
# TASK 1: Prepare and Understand Data (Data Understanding)
###############################################################

file_path = 'RFM/flo_data_20k.csv'
main_df = pd.read_csv(file_path)

df = main_df.copy()


# TASK 2: Examine the first 10 rows of the data set, variable names, size, descriptive statistics,
# null values, and variable types.


def check_dataframe(df, row_num=10):
    print("********** Dataset Shape **********")
    print("No. of Rows:", df.shape[0], "\nNo. of Columns:", df.shape[1])
    print("********** Dataset Information **********")
    print(df.info())
    print("********** Types of Columns **********")
    print(df.dtypes)
    print(f"********** First {row_num} Rows **********")
    print(df.head(row_num))
    print(f"********** Last {row_num} Rows **********")
    print(df.tail(row_num))
    print("********** Summary Statistics of The Dataset **********")
    print(df.describe())
    print("********** No. of Null Values In The Dataset **********")
    print(df.isnull().sum())


check_dataframe(df)


# TASK 3: Omnichannel means that customers shop from both online and offline platforms.
# Create new variables for each customer's total number of purchases and spending.

# total purchase of omnichannel (offline + online), total number of purchases
df["total_purchase"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
# total spend of omnichannel (offline + online), total expenditure
df["total_spend"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.head()

# TASK 4: Examine the types of variables. Convert the object variables containing date in the data set to date format.
df.info()

date_columns = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
df[date_columns] = df[date_columns].apply(lambda x: [pd.to_datetime(date) for date in x])
df.dtypes

# TASK 5: Look at the distribution of the number of customers in the shopping channels, the total number of products
# purchased and total expenditures.
# master_id count shows us how many purchases there are.

df.groupby('order_channel').agg({'total_purchase': 'sum',
                                 'total_spend': 'sum',
                                 'master_id': 'count'}).sort_values(by='master_id', ascending=False)

# TASK 6: Rank the top 10 customers who spend the most.

df[["master_id", "total_spend"]].sort_values(by="total_spend", ascending=False).head(10)

# TASK 7: Rank the top 10 customers with the most purchases.

df[["master_id", "total_purchase"]].sort_values(by="total_purchase", ascending=False).head(10)

# TASK 8: Functionalize the data provisioning process.


def data_processing(df):
    df["total_purchase"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_spend"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    date_columns = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    df[date_columns] = df[date_columns].apply(lambda x: [pd.to_datetime(date) for date in x])

    return df


data_processing(df)

###############################################################
# TASK 2: Calculating RFM Metrics
###############################################################

# Step 1: Make the definitions of Recency, Frequency and Monetary.

"""
Recency (Taze Görünüm): Müşterinin son satın alma tarihinden bugüne kadar geçen süreyi ifade eder. Bu süre ne kadar 
kısa ise, müşterinin daha "taze" olduğu ve daha aktif olduğu anlamına gelir.

Frequency (Sıklık): Müşterinin belirli bir zaman aralığında yaptığı satın alma sayısını ifade eder. Bu, müşterinin ne 
sıklıkla satın aldığına ve dolayısıyla markanın ne kadar sıklıkla etkileşimde olduğuna işaret eder.

Monetary (Parasal): Müşterinin belirli bir zaman aralığında yaptığı toplam harcama tutarını ifade eder. Bu, müşterinin 
ne kadar değerli olduğunu gösterir ve markanın ne kadar gelir elde ettiğini yansıtır.
"""

# Step 2: Calculate the Recency, Frequency and Monetary metrics for the customer.
# Step 3: Assign your calculated metrics to a variable named rfm.
# Step 4: Change the names of the metrics you created to recency, frequency and monetary.

# Find the last order date
last_order = df["last_order_date"].max()
# Setting the recency date for 2 days after the last order date
recency_date = dt.datetime(2021, 6, 2)


rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (recency_date - last_order_date.max()).days,
                                   'total_purchase': lambda total_purchase: total_purchase.sum(),
                                   'total_spend': lambda total_spend: total_spend.sum()})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm.describe().T

###############################################################
# TASK 3: Calculating RF and RFM Scores
###############################################################

# Converting Recency, Frequency and Monetary metrics to scores between 1-5 with the help of qcut and recording
# these scores as recency_score, frequency_score and monetary_score

rfm["recency_score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

# Express recency_score and frequency_score as a single variable and save it as RF_SCORE

rfm["rfm_score"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.head()

###############################################################
# TASK 4: Defining RF Scores as Segments
###############################################################

# Segment definition and converting RF_SCORE to segments with the help of defined seg_map so that the
# generated RFM scores can be explained more clearly.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

seg_map

rfm['segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

rfm['segment'] = rfm['segment'].replace(seg_map, regex=True)

rfm.head()

###############################################################
# TASK 5: Time for action!
###############################################################

# Step 1: Examine the recency, frequency and monetary averages of the segments.


rfm[["segment", "Recency", "Frequency", "Monetary"]].groupby("segment").agg(["mean", "count"])
segments = rfm['segment'].value_counts().sort_values(ascending=False)
segments

# library
import matplotlib.pyplot as plt
import seaborn as sns

# declaring data
data = segments.values
keys = segments.keys().values

# define Seaborn color palette to use
palette_color = sns.color_palette('bright')

# plotting data on chart
plt.pie(data, labels=keys, colors=palette_color, autopct='%.0f%%')

# displaying chart
plt.show()


# Step 2: With the help of RFM analysis, find the customers in the relevant profile for 2 cases and save
# the customer IDs to the csv.

# a. FLO includes a new women's shoe brand. The product prices of the brand it includes are above the general
# customer preferences. For this reason, customers in the profile who will be interested in the promotion of the
# brand and product sales are requested to be contacted privately. These customers were planned to be loyal and female
# shoppers. Save the id numbers of the customers to the csv file as new_brand_target_customer_id.cvs.

new_df = df.merge(rfm, on="master_id")

new_df = new_df[new_df["segment"].isin(["loyal_customers", "champions"])]

new_df.head()

new_df = new_df[new_df["interested_in_categories_12"].str.contains("KADIN")]

new_df.reset_index(drop=True, inplace=True)

new_df['segment'].unique()
new_df['interested_in_categories_12'].unique()

new_df["master_id"].to_csv("yeni_marka_hedef_musteri_id.csv")


# b. Up to 40% discount is planned for Men's and Children's products. We want to specifically target customers who
# are good customers in the past who are interested in categories related to this discount, but have not shopped for
# a long time and new customers. Save the ids of the customers in the appropriate profile to the csv file as
# discount_target_customer_ids.csv.

new_df2 = df.merge(rfm, on="master_id")

new_df2 = new_df2[new_df2["segment"].isin(["about_to_sleep", "new_customers"])]

new_df2.reset_index(drop=True, inplace=True)

new_df2["master_id"].to_csv("indirim_hedef_musteri_ids.csv")