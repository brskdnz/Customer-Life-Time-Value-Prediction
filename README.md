# Customer-Life-Time-Value-Prediction
# Data Manipulation
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%3f' % x)

df_ = pd.read_csv("../input/online-retail-listing/online_retail_listing.csv",delimiter=';',encoding="latin-1")
df = df_.copy()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])    
df["Price"] = df["Price"].apply(lambda x: float(str(x.replace(',','.'))))
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df['Quantity'] > 0)]
df = df[(df['Price'] > 0)]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# Data Preparation
cltv_df = df.groupby("Customer ID").agg({'InvoiceDate': [lambda Invoicedate: (Invoicedate.max() - Invoicedate.min()).days,
                                                         lambda Invoicedate: (today_date - Invoicedate.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                        'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['Recency', 'T', 'Frequency', 'Monetary']

cltv_df["Monetary"] = cltv_df["Monetary"] / cltv_df["Frequency"]

cltv_df = cltv_df[(cltv_df['Frequency'] > 1)]
cltv_df["Recency"] = cltv_df["Recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

cltv_df.describe().T
!pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_period_transactions
from scipy.special import logsumexp

bgf = BetaGeoFitter(penalizer_coef = 0.001)

bgf.fit(cltv_df['Frequency'],
       cltv_df['Recency'],
       cltv_df['T'])

# what is the expected number of transaction in first month ?
cltv_df["exp_1_month"] = bgf.predict(4,
                                  cltv_df['Frequency'],
                                  cltv_df['Recency'],
                                  cltv_df['T'])

cltv_df["exp_1_month"].sort_values(ascending=False).head(10)

# what is the expected number of transaction in sixth month ?
cltv_df["exp_6_month"] = bgf.predict(26,
                                  cltv_df['Frequency'],
                                  cltv_df['Recency'],
                                  cltv_df['T'])

cltv_df["exp_6_month"].sort_values(ascending=False).head(10)

cltv_df["exp_1_year"] = bgf.predict(52,
                                  cltv_df['Frequency'],
                                  cltv_df['Recency'],
                                  cltv_df['T'])

cltv_df["exp_1_year"].sort_values(ascending=False).head(10)

# Expected Average Profit
from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['Frequency'],cltv_df['Monetary'])

cltv_df["Expected_Avarege_Profit"] = ggf.conditional_expected_average_profit(cltv_df['Frequency'], cltv_df['Monetary'])

ggf.conditional_expected_average_profit(cltv_df['Frequency'],
                                        cltv_df['Monetary']).sort_values(ascending=False).head(10)
                                       
# BG/NBD and Gamma-Gamma Model

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['Frequency'],
                                   cltv_df['Recency'],
                                   cltv_df['T'],
                                   cltv_df['Monetary'],
                                   time=12, # 1 year
                                   freq="W", #frequency of tenure
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)

# Customer Segmentation

cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.head(10)

# 3D Graph Model

import plotly.express as px
fig = px.scatter_3d(cltv_final, x=cltv_final["Recency"], y=cltv_final["Frequency"], z=cltv_final["Monetary"], color=cltv_final["clv"])
fig.show()

import seaborn as sns 
sns.pairplot(cltv_final,hue="clv")

# Also you can find this work in kaggle
# https://www.kaggle.com/code/brskdnz/customer-segmentation-with-rfm-and-cltv-prediction
