import seaborn as sns
import streamlit as st
from vnstock import *
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier
from pypfopt import plotting
from pypfopt import risk_models
from pypfopt import expected_returns


st.set_page_config(page_title="Vietnam Market Stock")

list_name = []
list_df = []
start = "2000-1-1"
end = dt.date.today().strftime("%Y-%m-%d")
st.title("Stock's Portfolio Optimization in Viet Nam Market")
tab1, tab2 , tab3, tab4 = st.tabs(["Data", "Chart Line","Daily Return","Portfolio Optimization"])

#Get data
tab1.subheader("Amount of Stock")
number_of_stock = tab1.number_input("_Please input amount of stocks you want to invest:_",step =1,min_value=0,max_value=10)
count_stock = 0
if number_of_stock == 0:
    tab1.write(":red[Please input amount !]")
elif 0< number_of_stock <=2:
    tab1.write(":red[Please input amount of stock  > 2 !]")
else:
    tab1.subheader("Stock's Code")
    for i in range(0,number_of_stock):
        selected_stock = tab1.text_input("_Please input stock code:_", key = count_stock)
        count_stock = 1 + count_stock
        list_name.append(selected_stock)
        if selected_stock == "":
            tab1.write(":red[Please input the stock code !]")
        else:
            fd = int(time.mktime(time.strptime(start, "%Y-%m-%d")))
            td = int(time.mktime(time.strptime(end, "%Y-%m-%d")))
            symbol = selected_stock
            data = requests.get(
                'https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={}&type=stock&resolution=D&from={}&to={}'.format(
                    symbol, fd, td)).json()
            df = json_normalize(data['data'])
            if df.empty:
                tab1.write(":red[You input a wrong format or a wrong code, please try again !]")
            else:
                df['tradingDate'] = pd.to_datetime(df.tradingDate.str.split("T", expand=True)[0])
                df.columns = df.columns.str.title()
                df.rename(columns={'Tradingdate': 'TradingDate'}, inplace=True)
                df["TradingDate"]= df["TradingDate"].dt.strftime("%Y-%m-%d")
                df = pd.DataFrame(df)
                option = tab1.checkbox('Show the history of stock', key=count_stock + 100)
                list_df.append(df)
                if option == True:
                    tab1.subheader(selected_stock + "'s stock history data")
                    tab1.dataframe(df)

tab1.write("_if you're done , you can click on the next tab for next step_")

#Tab2
tab2.subheader("Chart line of stocks")
option_2 = tab2.checkbox('Show close price of stocks data')
if option_2 == True:
    if list_df == []:
        tab2.write(":red[Input the stock code at tab Data first !]")
    else:
        n = 0
        for i in range(0, len(list_name)):
            if n == 0:
                n = n + 1
                data_1 = list_df[i].drop(columns=['Open', "High", "Low", "Volume"])
                data_1.columns = [list_name[i], "TradingDate"]
                data_1.set_index("TradingDate", inplace=True)
            if n > 0:
                data_2 = list_df[i].drop(columns=['Open', "High", "Low", "Volume"])
                data_2.columns = [list_name[i], "TradingDate"]
                data_2.set_index("TradingDate", inplace=True)
                data_1 = pd.merge(data_1, data_2, left_index=True, right_index=True)
        data_1 = data_1.drop(columns=[list_name[0] + "_x"])
        data_1 = data_1.rename(columns={list_name[0] + "_y": list_name[0]})
        tab2.dataframe(data_1)

# visualization
option_3 = tab2.checkbox('Show chart line of stocks data')
if option_3 == True:
    if list_df == []:
        tab2.write(":red[Input the stock code at tab Data first !]")
    else:
        n = 0
        for i in range(0, len(list_name)):
            if n == 0:
                n = n + 1
                data_1 = list_df[i].drop(columns=['Open', "High", "Low", "Volume"])
                data_1.columns = [list_name[i], "TradingDate"]
                data_1.set_index("TradingDate", inplace=True)
            if n > 0:
                data_2 = list_df[i].drop(columns=['Open', "High", "Low", "Volume"])
                data_2.columns = [list_name[i], "TradingDate"]
                data_2.set_index("TradingDate", inplace=True)
                data_1 = pd.merge(data_1, data_2, left_index=True, right_index=True)
        data_1 = data_1.drop(columns=[list_name[0] + "_x"])
        data_1 = data_1.rename(columns={list_name[0] + "_y": list_name[0]})
        tab2.subheader("The chart line of stocks")
        tab2.line_chart(data_1)

tab2.write("_if you're done , you can click on the next tab for next step_")

#Tab3
Portfolio = False
option_4 = tab3.checkbox('Show the daily-return of stocks data')
if option_4 == True:
    if list_df == []:
        tab3.write(":red[Input the stock code at tab Data first !]")
    else:
        tab3.subheader("The daily-return of stocks data")
        col1, col2 = tab3.columns([3, 1])
        col1.write("The Daily Return Data")
        n = 0
        for i in range(0, len(list_name)):
            if n == 0:
                n = n + 1
                data_3 = list_df[i].drop(columns=['Open', "High", "Low", "Volume"])
                data_3.columns = [list_name[i], "TradingDate"]
                data_3['TradingDate'] = pd.to_datetime(data_3['TradingDate'],format='%Y-%m-%d')
                data_3.set_index("TradingDate", inplace=True)
                data_3[list_name[i]] = data_3[list_name[i]].resample('d').ffill().pct_change()
                data_3 = data_3.dropna()
            if n > 0:
                data_4 = list_df[i].drop(columns=['Open', "High", "Low", "Volume"])
                data_4.columns = [list_name[i], "TradingDate"]
                data_4['TradingDate'] = pd.to_datetime(data_4['TradingDate'],format= '%Y-%m-%d')
                data_4.set_index("TradingDate", inplace=True)
                data_4[list_name[i]] = data_4[list_name[i]].resample('d').ffill().pct_change()
                data_4 = data_4.dropna()
                data_3 = pd.merge(data_3, data_4, left_index=True, right_index=True)
        data_3 = data_3.drop(columns=[list_name[0]+"_x"])
        data_3 = data_3.rename(columns={list_name[0]+"_y": list_name[0]})
        col1.dataframe(data_3)
        col2.write("Standard Deviation of Daily Return")
        col2.write(data_3.std())
        Portfolio = True


if Portfolio == True:
    tab3.subheader("Chart of Volatility's Daily Return")
    fig = sns.displot(data=data_3, kind='kde', aspect=2.5)
    tab3.pyplot(fig)

    tab3.subheader("Chart of Cumulative Returns of Individual Stocks")
    daily_cum_returns = (1 + data_3).cumprod() * 2000000
    fig = plt.figure(figsize=(20, 10))
    plt.plot(daily_cum_returns)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns of Individual Stocks")
    plt.title("Cumulative Returns of Individual Stocks Starting with 2.000.000 VND ")
    plt.legend(daily_cum_returns)
    tab3.pyplot(fig)

tab3.write("_if you're done , you can click on the next tab for next step_")

#Tab4
if Portfolio == True:
    # Calculate expected returns and sample covariance matrix
    tab4.subheader("Expected returns of each stock")
    mu = expected_returns.mean_historical_return(data_1)
    S = risk_models.sample_cov(data_1)
    tab4.dataframe(mu)
    tab4.subheader("Optimize portfolio")
    # Optimize portfolio for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    fig,ax = plt.subplots(figsize=(8, 6))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Find the max sharpe portfolio
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=0.015)
    ret_tangent, std_tangent, _ = ef.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_title("Efficient Frontier with Random Portfolios")
    ax.legend()
    plt.tight_layout()
    tab4.pyplot(fig)

    #Weight
    tab4.subheader("Weight and some expectations:")
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=0.015)
    weights = ef.clean_weights()
    weights_df = pd.DataFrame.from_dict(weights, orient='index')
    weights_df.columns = ['weights']
    tab4.dataframe(weights_df)

    #Expected
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
    tab4.write('Expected annual return: {}%'.format((expected_annual_return * 100).round(2)))
    tab4.write('Annual volatility: {}%'.format((annual_volatility * 100).round(2)))
    tab4.write('Sharpe ratio: {}'.format(sharpe_ratio.round(2)))
else:
    tab4.write(":red[Input the stock code at tab Data first !]")

