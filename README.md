# Customer Segmentation by RFM Analysis - FLO RFM Analysis

![rfm](https://github.com/mtahiraslan/flo_rfm_analysis/blob/main/rfm.PNG)

## What is RFM?

RFM analysis is a marketing technique used to quantitatively rank and group customers based on the recency, frequency and monetary total of their recent transactions to identify the best customers and perform targeted marketing campaigns. The system assigns each customer numerical scores based on these factors to provide an objective analysis. RFM analysis is based on the marketing adage that "80% of your business comes from 20% of your customers."

RFM analysis ranks each customer on the following factors:

**Recency:** How recent was the customer's last purchase? Customers who recently made a purchase will still have the product on their mind and are more likely to purchase or use the product again. Businesses often measure recency in days. But, depending on the product, they may measure it in years, weeks or even hours.

**Frequency:** How often did this customer make a purchase in a given period? Customers who purchased once are often are more likely to purchase again. Additionally, first time customers may be good targets for follow-up advertising to convert them into more frequent customers.

**Monetary:** How much money did the customer spend in a given period? Customers who spend a lot of money are more likely to spend money in the future and have a high value to a business.

## Business Problem

FLO, which is an online shoe store, wants to divide its customers into segments and determine marketing strategies according to these segments. For this, the behavior of customers will be defined and groups will be formed according to the clutches in these behaviors.

## Features of Dataset

- Total Features : 12
- Total Row : 19.945
- CSV File Size : 2.7 MB

## The story of the dataset

Dataset consists of information obtained from the past shopping behaviors of customers who make their latest shopping from FLO as Omnichannel (both online and offline shopping) in 2020 - 2021.

- **master_id:** Unique customer number
- **order_channel:** Which channel of the shopping platform used (Android, iOS, desktop, mobile)
- **last_order_channel:** Channel where the last shopping was made
- **first_order_date:** Customer's first shopping date
- **last_order_date:** Customer's latest shopping date
- **last_order_date_online:** Customer's latest shopping date on the online platform
- **last_order_date_offline:** Customer's latest shopping date on offline platform
- **order_num_total_ever_online:** Customer's total number of shopping on the online platform
- **order_num_total_ever_offline:** Customer's total number of shopping on the offline platform
- **customer_value_total_ever_offline:** Total fee paid by the customer in offline shopping
- **customer_value_total_ever_online:** Total fee paid by the customer in online shopping
- **interested_in_categories_12:** List of categories where the customer shopping in the last 12 months

![segments](https://github.com/mtahiraslan/flo_rfm_analysis/blob/main/segments.PNG)

## Methods and libraries used in the project

- pandas, numpy, datetime
- Segmentation

## Requirements.txt

- Please review the 'requirements.txt' file for required libraries.
