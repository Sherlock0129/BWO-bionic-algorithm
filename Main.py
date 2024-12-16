import streamlit as st
from OptimizationPage.Page import optimization_page
from HomePage.homePage import home_page
from PythonPage.FundamentalPage import fundamental_page
from OptimizationAlgorithmImprovementExperimentPage.Page import experiment_page
import cupy as cp

print(cp.cuda.runtime.getDeviceCount())  # 输出可用 GPU 数量
print(cp.cuda.Device().name)
# Sidebar navigation
title = "Navigation"
st.sidebar.title(title)
page = st.sidebar.selectbox("Go to", ["Home", "Python Basics", "Optimization", "Optimization Algorithm Improvement Experiment"])

# Page routing
if page == "Home":
    home_page()
elif page == "Optimization":
    optimization_page()
elif page == "Python Basics":
    fundamental_page()
elif page == "Optimization Algorithm Improvement Experiment":
    experiment_page()
else:
    st.error("Invalid page selection. Please select a valid page.")

