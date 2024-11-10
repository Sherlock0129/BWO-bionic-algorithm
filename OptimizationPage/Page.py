import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib.ticker import ScalarFormatter

from OptimizationPage import Function
from OptimizationPage.Algorithm import BWO, PSO, DE, AOA, HHO, CSA


def optimization_page():
    """Display the Fish Swarm Optimization page."""
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
        'Shekel 10': r'f(x) = -\sum \left[(x - aSH[i])^2 + cSH[i]\right]^{-1}',
        'composite_function_1': r'\text{CF1}(x) = \sum_{i=1}^{10} \lambda_i \cdot f_i(x), \quad \text{where} '
                                r'\quad f_i(x) = \text{Sphere}(x), \; \sigma_i = 1, \; \lambda_i = \frac{5}{100} \; '
                                r'\text{for} \; i = 1, 2, \ldots, 10\\x \in [-5, 5]^{10}',
        'composite_function_2': r'\text{CF2}(x) = \sum_{i=1}^{10} \lambda_i \cdot f_i(x), \quad \text{where} \quad '
                                r'f_i(x) = \text{Griewank}(x), \; \sigma_i = 1, \; \lambda_i = \frac{5}{100} \; '
                                r'\text{for} \; i = 1, 2, \ldots, 10\\x \in [-5, 5]^{10} ',
        'composite_function_3': r'\text{CF3}(x) = \sum_{i=1}^{10} \lambda_i \cdot f_i(x), \quad \text{where} \quad '
                                r'f_i(x) = \text{Griewank}(x), \; \sigma_i = 1, \; \lambda_i = 1 \; \text{for} \; i = '
                                r'1, 2, \ldots, 10\\x \in [-5, 5]^{10}',
        'composite_function_4': r'\text{CF4}(x) = \sum_{i=1}^{10} \lambda_i \cdot f_i(x), \quad \text{'
                                r'where}\\\begin{align*}f_1(x) &= \text{Ackley}(x_1), & f_2(x) &= \text{Ackley}('
                                r'x_2), \\f_3(x) &= \text{Rastrigin}(x_3), & f_4(x) &= \text{Rastrigin}(x_4), '
                                r'\\f_5(x) &= \text{Weierstrass}(x_5), & f_6(x) &= \text{Weierstrass}(x_6), '
                                r'\\f_7(x) &= \text{Griewank}(x_7), & f_8(x) &= \text{Griewank}(x_8), \\f_9(x) &= '
                                r'\text{Sphere}(x_9), & f_{10}(x) &= \text{Sphere}(x_{10}),\end{align*}\\\sigma_i = '
                                r'1, \; \lambda = \left[\frac{5}{32}, \frac{5}{32}, 1, 1, \frac{5}{0.5}, '
                                r'\frac{5}{0.5}, \frac{5}{100}, \frac{5}{100}, \frac{5}{100}, '
                                r'\frac{5}{100}\right]\\x \in [-5, 5]^{10}',
        'composite_function_5': r'\text{CF5}(x) = \sum_{i=1}^{10} \lambda_i \cdot f_i(x), \quad \text{where}\\\begin{'
                                r'align*}f_1(x) &= \text{Rastrigin}(x_1), & f_2(x) &= \text{Rastrigin}(x_2), '
                                r'\\f_3(x) &= \text{Weierstrass}(x_3), & f_4(x) &= \text{Weierstrass}(x_4), '
                                r'\\f_5(x) &= \text{Griewank}(x_5), & f_6(x) &= \text{Griewank}(x_6), \\f_7(x) &= '
                                r'\text{Ackley}(x_7), & f_8(x) &= \text{Ackley}(x_8), \\f_9(x) &= \text{Sphere}(x_9), '
                                r'& f_{10}(x) &= \text{Sphere}(x_{10}),\end{align*}\\\sigma_i = 1, \; \lambda = '
                                r'\left[\frac{1}{5}, \frac{1}{5}, \frac{5}{0.5}, \frac{5}{0.5}, \frac{5}{100}, '
                                r'\frac{5}{100}, \frac{5}{32}, \frac{5}{32}, \frac{5}{100}, \frac{5}{100}\right]\\x '
                                r'\in [-5, 5]^{10}',
        'composite_function_6': r'\text{CF6}(x) = \sum_{i=1}^{10} \lambda_i \cdot f_i(x), \quad \text{where}\\\begin{'
                                r'align*}f_1(x) &= \text{Rastrigin}(x_1), & f_2(x) &= \text{Rastrigin}(x_2), '
                                r'\\f_3(x) &= \text{Weierstrass}(x_3), & f_4(x) &= \text{Weierstrass}(x_4), '
                                r'\\f_5(x) &= \text{Griewank}(x_5), & f_6(x) &= \text{Griewank}(x_6), \\f_7(x) &= '
                                r'\text{Ackley}(x_7), & f_8(x) &= \text{Ackley}(x_8), \\f_9(x) &= \text{Sphere}(x_9), '
                                r'& f_{10}(x) &= \text{Sphere}(x_{10}),\end{align*}\\\sigma = [0.1, 0.2, 0.3, 0.4, '
                                r'0.5, 0.6, 0.7, 0.8, 0.9, 1], \;\\ \lambda = \left[0.1 \cdot \frac{1}{5}, '
                                r'0.2 \cdot \frac{1}{5}, 0.3 \cdot \frac{5}{0.5}, 0.4 \cdot \frac{5}{0.5}, '
                                r'0.5 \cdot \frac{5}{100}, 0.6 \cdot \frac{5}{100}, 0.7 \cdot \frac{5}{32}, '
                                r'0.8 \cdot \frac{5}{32}, 0.9 \cdot \frac{5}{100}, 1 \cdot \frac{5}{100}\right]\\x '
                                r'\in [-5, 5]^{10}'
    }
    st.title("Optimization")
    st.write("This page allows you to optimize a function using Fish Swarm Optimization or Particle Swarm Optimization.")

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
            'Shekel 10',
            'composite_function_1',
            'composite_function_2',
            'composite_function_3',
            'composite_function_4',
            'composite_function_5',
            'composite_function_6',
        ])
        expression = function_expressions[function_real_name]
        st.latex(expression)


        model_name = st.selectbox("**Select Model**",
                                  ["Beluga whale optimization", "Particle Swarm Optimization", "Differential Evolution",
                                   "Arithmetic Optimization Algorithm", "Crow Search Algorithm",
                                   "Harris Hawks Optimization"])

        Npop = st.slider("**Population Size**", 10, 100, 50)
        Max_it = st.slider("**Max Iterations**", 10, 2000, 500)
        # Run Optimization button
        if st.sidebar.button("**Run Optimization**"):
            # 列表用于存储每次运行的最佳位置和最佳值
            results = []
            curves = []  # 如果想平均曲线，也可以存储每次的曲线

            for i in range(20):
                # 获取函数的详细信息
                function_name = Function.get_function(function_real_name)
                lb, ub, nD, fobj = Function.get_function_details(function_name)
                # 运行优化算法
                if model_name == "Beluga whale optimization":
                    xposbest, fvalbest, Curve = BWO.bwo(Npop, Max_it, lb, ub, nD, fobj)
                elif model_name == "Particle Swarm Optimization":
                    xposbest, fvalbest, Curve = PSO.pso(Npop, Max_it, lb, ub, nD, fobj)
                elif model_name == "Differential Evolution":
                    xposbest, fvalbest, Curve = DE.de(Npop, Max_it, lb, ub, nD, fobj)
                elif model_name == "Arithmetic Optimization Algorithm":
                    xposbest, fvalbest, Curve = AOA.aoa(Npop, Max_it, lb, ub, nD, fobj)
                elif model_name == "Crow Search Algorithm":
                    xposbest, fvalbest, Curve = CSA.csa(Npop, Max_it, lb, ub, nD, fobj)
                elif model_name == "Harris Hawks Optimization":
                    xposbest, fvalbest, Curve = HHO.hho(Npop, Max_it, lb, ub, nD, fobj)
                else:
                    st.error("Invalid model name")
                    return

                # 将每次运行的结果存储在列表中
                results.append({"Run": i + 1, "Best Position": xposbest, "Best Value": fvalbest})
                curves.append(Curve)

            # 将结果转换为 DataFrame 并显示为表格
            results_df = pd.DataFrame(results)
            # st.table(results_df)  # 或使用 st.dataframe(results_df) 以获得更丰富的表格交互功能
            st.dataframe(results_df)
            # 绘制图
            fig, ax = plt.subplots()
            # for curve in curves:
            #     ax.plot(curve)
            # 计算平均曲线
            mean_curve = np.mean(curves, axis=0)
            ax.plot(mean_curve)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best Value")
            # Use ScalarFormatter to set the y-axis to scientific notation
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Show in scientific notation

            # Optionally set grid lines for better readability
            ax.grid()
            st.pyplot(fig)

            # 计算并显示平均值和标准差
            mean_best_value = np.mean(results_df["Best Value"])
            std_best_value = np.std(results_df["Best Value"])
            # 将mean_best_value，std_best_value存放在一个set中
            value_set = {mean_best_value, std_best_value}
            # 将set转换为list
            value_list = list(value_set)


            st.write("**Optimization Summary:**")
            st.write(f"Average Best Value: {mean_best_value}")
            st.write(f"Standard Deviation of Best Value: {std_best_value}")


    else:
        model_name = st.selectbox("**Select Model**",
                                  ["Fish Swarm Optimization", "Particle Swarm Optimization", "Differential Evolution",
                                   "Arithmetic Optimization Algorithm", "Crow Search Algorithm",
                                   "Harris Hawks Optimization"])
        objective_function = st.text_input("**Enter Your Function(Objective Function)**", "x**4")
        lb = st.number_input("**Enter Lower Bound**", value=-100)
        ub = st.number_input("**Enter Upper Bound**", value=100)
        nD = st.number_input("**Enter Number of Dimensions**", value=1)
        Npop = st.slider("**Population Size**", 10, 100, 50)
        Max_it = st.slider("**Max Iterations**", 10, 2000, 1000)


        if st.sidebar.button("**Run Optimization**"):
            # 列表用于存储每次运行的最佳位置和最佳值
            results = []
            curves = []  # 如果想平均曲线，也可以存储每次的曲线

            for i in range(20):
                # 获取函数的详细信息
                fobj = sp.lambdify(sp.symbols('x'), sp.sympify(objective_function))
                # 运行优化算法
                if model_name == "Fish Swarm Optimization":
                    xposbest, fvalbest, Curve = BWO.bwo(Npop, Max_it, lb, ub, nD, fobj)
                elif model_name == "Particle Swarm Optimization":
                    xposbest, fvalbest, Curve = PSO.pso(Npop, Max_it, lb, ub, nD, fobj)
                elif model_name == "Differential Evolution":
                    xposbest, fvalbest, Curve = DE.de(Npop, Max_it, lb, ub, nD, fobj)
                elif model_name == "Arithmetic Optimization Algorithm":
                    xposbest, fvalbest, Curve = AOA.aoa(Npop, Max_it, lb, ub, nD, fobj)
                elif model_name == "Crow Search Algorithm":
                    xposbest, fvalbest, Curve = CSA.csa(Npop, Max_it, lb, ub, nD, fobj)
                elif model_name == "Harris Hawks Optimization":
                    xposbest, fvalbest, Curve = HHO.hho(Npop, Max_it, lb, ub, nD, fobj)
                else:
                    st.error("Invalid model name")
                    return

                # 将每次运行的结果存储在列表中
                results.append({"Run": i + 1, "Best Position": xposbest, "Best Value": fvalbest})
                curves.append(Curve)

            # 将结果转换为 DataFrame 并显示为表格
            results_df = pd.DataFrame(results)
            # st.table(results_df)  # 或使用 st.dataframe(results_df) 以获得更丰富的表格交互功能
            st.dataframe(results_df)
            # 绘制图
            fig, ax = plt.subplots()
            mean_curve = np.mean(curves, axis=0)
            ax.plot(mean_curve)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best Value")
            # Use ScalarFormatter to set the y-axis to scientific notation
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Show in scientific notation

            # Optionally set grid lines for better readability
            ax.grid()
            st.pyplot(fig)

            # 计算并显示平均值和标准差
            mean_best_value = np.mean(results_df["Best Value"])
            std_best_value = np.std(results_df["Best Value"])
            # 将mean_best_value，std_best_value存放在一个set中
            value_set = {mean_best_value, std_best_value}
            # 将set转换为list
            value_list = list(value_set)

            st.write("**Optimization Summary:**")
            st.write(f"Average Best Value: {value_list[0]}")
            st.write(f"Standard Deviation of Best Value: {value_list[1]}")




