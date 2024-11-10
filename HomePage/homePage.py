import streamlit as st

import streamlit as st


def home_page():
    """Display the Home page."""
    st.title("🌟 Welcome to the Optimization App 🌟")
    st.write("Select a page from the sidebar to start exploring different optimization algorithms.")

    # Brief introduction with colorful text
    st.markdown("<h2 style='color: #4CAF50;'>About This App</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 18px;'>After reading the paper, we implemented the algorithm in Python based on the "
        "MATLAB code provided in the study. We also implemented <span style='color: #FF5722;'>30 test functions</span> "
        "(including <span style='color: #FF5722;'>9 essential ones</span> required for experiments) to thoroughly test "
        "the algorithm.</p>",
        unsafe_allow_html=True,
    )

    # Further details with highlighted parts
    st.markdown(
        "<p style='font-size: 18px;'>In addition to these test functions, users can input their own test functions "
        "for <span style='color: #03A9F4;'>algorithm optimization testing</span>, greatly enhancing flexibility. We "
        "also implemented <span style='color: #FFC107;'>5 comparison algorithms</span> to showcase the advantages of "
        "the <span style='color: #673AB7;'>BWO algorithm</span>.</p>",
        unsafe_allow_html=True,
    )

    # Experiment settings with colorful highlights
    st.markdown(
        "<h3 style='color: #009688;'>Experiment Settings</h3>", unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size: 18px;'>On this webpage, you can:</p>"
        "<ul>"
        "<li>Modify <span style='color: #FF9800;'>dimensions</span> and <span style='color: #FF9800;'>iteration counts</span></li>"
        "<li>Choose from the implemented <span style='color: #FF9800;'>optimization algorithms</span> and <span style='color: #FF9800;'>test functions</span></li>"
        "</ul>"
        "<p style='font-size: 18px;'>After executing the optimization, you’ll see a <span style='color: #E91E63;'>table of results</span> "
        "and a <span style='color: #E91E63;'>curve</span> showing iteration progress.</p>",
        unsafe_allow_html=True,
    )


