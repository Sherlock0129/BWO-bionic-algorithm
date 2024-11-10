import streamlit as st
from OptimizationPage.Page import optimization_page
from HomePage.homePage import home_page
from PythonPage.FundamentalPage import fundamental_page

# Sidebar navigation
title = "Navigation"
st.sidebar.title(title)
page = st.sidebar.selectbox("Go to", ["Home", "Python Basics", "Optimization"])

# Page routing
if page == "Home":
    home_page()
elif page == "Optimization":
    optimization_page()
elif page == "Python Basics":
    fundamental_page()
else:
    st.error("Invalid page selection. Please select a valid page.")

