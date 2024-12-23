import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib.ticker import ScalarFormatter

from OptimizationPage import Function
from ResultPage import ENBWO
from OptimizationPage.Algorithm import BWO


def result_page():
    """Display the Enhanced Beluga Whale Optimization page."""
    function_expressions = {
        'Sphere': r'f(x) = x_1^2 + x_2^2 + \ldots + x_n^2',
        "Schwefel's 2.22": r'f(x) = \sum |x_i| + \prod |x_i|',
        'Powell Sum': r'f(x) = \sum |x_i|^i',
        "Schwefel's 1.2": r'f(x) = \sum \left(\sum x_i\right)^2',
        "Schwefel's 2.21": r'f(x) = \max |x_i|',
        'Rosenbrock': r'f(x) = \sum \left[100 \cdot (x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right]',
        'Step': r'f(x) = \sum |x_i + 0.5|^2',
        'Quartic': r'f(x) = \sum (i+1) \cdot x_i^4 + \text{random}',
        'Zakharov': r'f(x) = \sum x_i^2 + \sum \frac{1}{2}i x_i',
        'Schwefel': r'f(x) = -x_i \sin(\sqrt{|x_i|})',
        'Periodic': r'f(x) = 1 + \sum \sin(x_i^2) - e^{-\sum x_i^2}',
        'Styblinski-Tang': r'f(x) = 0.5 \cdot \sum (x_i^4 - 16x_i^2 + 5x_i)',
        'Rastrigin': r'f(x) = 10n + \sum \left[x_i^2 - 10\cos(2\pi x_i)\right]',
        'Ackley 1': r'f(x) = -20 e^{-0.2 \sqrt{\sum x_i^2}} - e^{\sum \cos(2\pi x_i)} + 20 + e',
        'Griewank': r'f(x) = 1 + \sum \frac{x_i^2}{4000} - \prod \cos\left(\frac{x_i}{\sqrt{i}}\right)',
        'Xin-She Yang N.4': r'f(x) = \left(\sum \sin(x_i^2)\right)^2 - e^{-\sum \sin(\sqrt{|x_i|})^2}',
        'Penalized': r'f(x) = \sum \left[x_i^2 + \sum Ufun(x, 10, 100, 4)\right]',
        'Penalized2': r'f(x) = 0.1 \cdot \left(\sum (x_i - 1)^2 + \sum \left(\sin(3\pi x_i)\right)^2\right)',
        'Foxholes': r'f(x) = \sum \left[\sum (x_i - aSH[i])^2 + cSH[i]\right]^{-1}',
        'Kowalik': r'f(x) = \sum \left(aK - \frac{x[0](bK^2 + x[1]bK)}{bK^2 + x[2]bK + x[3]}\right)^2',
        'Six Hump Camel': r'f(x) = 4x_1^2 - 2.1x_1^4 + \frac{x_1^6}{3} + x_1 x_2 - 4x_2^2 + 4x_2^4',
        'Shekel 5': r'f(x) = -\sum \left[(x - aSH[i])^2 + cSH[i]\right]^{-1}',
        'Shekel 7': r'f(x) = -\sum \left[(x - aSH[i])^2 + cSH[i]\right]^{-1}',
        'Shekel 10': r'f(x) = -\sum \left[(x - aSH[i])^2 + cSH[i]\right]^{-1}'
    }

    st.title("Enhanced Beluga Whale Optimization")
    st.write("This page allows you to optimize a function using the Enhanced Beluga Whale Optimization Algorithm.")

    exist_function = st.toggle("Use existing function", True)
    if exist_function:
        # Function selection
        function_real_name = st.selectbox("**Select Function**", [
            'Sphere',
            'Schwefel\'s 2.22',
            'Powell Sum',
            'Schwefel\'s 1.2',
            'Schwefel\'s 2.21',
            'Rosenbrock', 'Step',
            'Quartic',
            'Zakharov',
            'Schwefel',
            'Periodic',
            'Styblinski-Tang',
            'Rastrigin',
            'Ackley 1',
            'Griewank',
            'Xin-She Yang N.4',
            'Penalized',
            'Penalized2',
            'Foxholes',
            'Kowalik',
            'Six Hump Camel',
            'Shekel 5',
            'Shekel 7',
            'Shekel 10'
        ])
        expression = function_expressions[function_real_name]
        st.latex(expression)

        model_name = "Enhanced Beluga Whale Optimization"  # Only the enhanced BWO in this case

        Npop = st.slider("**Population Size**", 10, 100, 50)
        Max_it = st.slider("**Max Iterations**", 10, 2000, 500)

        if st.button("**Run Optimization**"):
            # List to store best positions and values for each run
            results = []
            curves_enbwo = []  # To store curves for Enhanced BWO
            curves_bwo = []  # To store curves for original BWO

            for i in range(20):
                # Get function details
                function_name = Function.get_function(function_real_name)
                lb, ub, nD, fobj = Function.get_function_details(function_name)

                # Run Enhanced BWO algorithm

                xposbest_enbwo, fvalbest_enbwo, Curve_enbwo = ENBWO.optimize(Npop, nD, lb, ub, Max_it, fobj)

                # Run Original BWO algorithm

                xposbest_bwo, fvalbest_bwo, Curve_bwo = BWO.bwo(Npop, nD, lb, ub, Max_it, fobj)

                # Store results for each run
                results.append({
                    "Run": i + 1,
                    "Best Position (ENBWO)": xposbest_enbwo,
                    "Best Value (ENBWO)": fvalbest_enbwo,
                    "Best Position (BWO)": xposbest_bwo,
                    "Best Value (BWO)": fvalbest_bwo
                })
                curves_enbwo.append(Curve_enbwo)
                curves_bwo.append(Curve_bwo)

            # Convert results to DataFrame and display
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the mean curve for both algorithms
            mean_curve_enbwo = np.mean(curves_enbwo, axis=0)
            mean_curve_bwo = np.mean(curves_bwo, axis=0)

            ax.plot(mean_curve_enbwo, label='Enhanced BWO', color='blue')
            ax.plot(mean_curve_bwo, label='Original BWO', color='red')

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best Value")

            # Format y-axis to scientific notation
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.grid()

            # Display the plot
            st.pyplot(fig)

            # Display average and standard deviation for both algorithms
            mean_best_value_enbwo = np.mean(results_df["Best Value (ENBWO)"])
            std_best_value_enbwo = np.std(results_df["Best Value (ENBWO)"])
            mean_best_value_bwo = np.mean(results_df["Best Value (BWO)"])
            std_best_value_bwo = np.std(results_df["Best Value (BWO)"])

            st.write("**Optimization Summary:**")
            st.write(f"Enhanced BWO:")
            st.write(f"Average Best Value: {mean_best_value_enbwo}")
            st.write(f"Standard Deviation of Best Value: {std_best_value_enbwo}")

            st.write(f"\nOriginal BWO:")
            st.write(f"Average Best Value: {mean_best_value_bwo}")
            st.write(f"Standard Deviation of Best Value: {std_best_value_bwo}")

        else:
            st.error("Currently only existing functions are supported for Enhanced BWO.")