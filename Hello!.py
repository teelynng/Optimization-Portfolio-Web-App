import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.write("# Welcome to Efficient Frontier Optimization Web App! 👋")

st.markdown(
    """
    Efficient Frontier Optimization Web App is an open-source Web App built specifically for
    being a portfolio building tool for beginners to invest in stocks.Website uses the efficient frontier method to build 
    a portfolio with the input of stocks that investors want to invest in. Website is built on Streamlit
    ### What is  Efficient Frontier?
    The efficient frontier is the set of optimal portfolios that offer the highest expected return for a defined level of risk or the lowest risk for a given level of expected return. 
    Portfolios that lie below the efficient frontier are sub-optimal because they do not provide enough return for the level of risk. 
    Portfolios that cluster to the right of the efficient frontier are sub-optimal because they have a higher level of risk for the defined rate of return.
    - The efficient frontier comprises investment portfolios that offer the highest expected return for a specific level of risk.
    - The standard deviation of returns in a portfolio measures investment risk and consistency in investment earnings.
    - Lower covariance between portfolio securities results in lower portfolio standard deviation.
    - Successful optimization of the return versus risk paradigm should place a portfolio along the efficient frontier line.
    - Optimal portfolios that comprise the efficient frontier usually exhibit a higher degree of diversification
    
    See more:
    Check out [efficientfrontier](http://www.efficientfrontier.com/)
"""
)
