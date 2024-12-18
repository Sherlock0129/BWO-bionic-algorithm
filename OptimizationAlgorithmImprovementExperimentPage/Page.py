import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib.ticker import ScalarFormatter

from OptimizationPage import Function
from OptimizationPage.Algorithm import BWO, PSO, DE, AOA, HHO, CSA
from OptimizationAlgorithmImprovementExperimentPage.DecompositionAlgorithm import BWO_1, BWO_2, BWO_3
from OptimizationAlgorithmImprovementExperimentPage.DecompositionAlgorithm import CPO_2, CPO_3, CPO_1


def experiment_page():
    def run_experiment():
        function_name = Function.get_function(function_real_name)
        lb, ub, nD, fobj = Function.get_function_details(function_name)

        algorithms = [
            ('B', BWO_1.exploration_phase, BWO_2.exploitation_phase, BWO_3.whale_fall_phase),
            ('C', CPO_1.CPO_exploration_phase, CPO_2.CPO_exploitation_phase, CPO_3.CPO_convergence_phase),
        ]

        results = []
        curves = []
        labels = []

        for alg1 in algorithms:
            for alg2 in algorithms:
                for alg3 in algorithms:
                    all_curves = []
                    all_best_values = []
                    for i in range(20):
                        # Phase 1
                        xposbest, fvalbest, Curve = alg1[1](Npop, Max_it, lb, ub, nD, fobj)
                        # Phase 2
                        xposbest, fvalbest, Curve = alg2[2](Npop, Max_it, lb, ub, nD, fobj, xposbest)
                        # Phase 3
                        xposbest, fvalbest, Curve = alg3[3](Npop, Max_it, lb, ub, nD, fobj, xposbest)

                        all_curves.append(Curve)
                        all_best_values.append(fvalbest)

                    mean_curve = np.mean(all_curves, axis=0)
                    curves.append(mean_curve)
                    labels.append(f"{alg1[0]}{alg2[0]}{alg3[0]}")

                    mean_best_value = np.mean(all_best_values)
                    std_best_value = np.std(all_best_values)
                    results.append({
                        "Combination": f"{alg1[0]}{alg2[0]}{alg3[0]}",
                        "Average Best Value": mean_best_value,
                        "Standard Deviation": std_best_value
                    })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Plotting
        num_plots = len(curves) // 4
        for plot_idx in range(num_plots):
            fig, ax = plt.subplots()
            for curve_idx in range(4):
                idx = plot_idx * 4 + curve_idx
                ax.plot(curves[idx], label=labels[idx])
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best Value")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.legend()
            ax.grid()
            st.pyplot(fig)

        st.write("**Optimization Summary:**")
        for _, row in results_df.iterrows():
            st.write(
                f"Combination: {row['Combination']}, Average Best Value: {row['Average Best Value']}, Standard Deviation: {row['Standard Deviation']}")

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
        'composite_function_4': r"""
                                \text{CF5}(x) = \sum_{i=1}^{5} \lambda_i \cdot f_i(x), \quad \text{where}\\
                                \begin{align*}
                                f_1(x) &= \text{Rotated Rosenbrock's Function}(x_1), & f_2(x) &= \text{High Conditioned Elliptic Function}(x_2), \\
                                f_3(x) &= \text{Rotated Bent Cigar Function}(x_3), & f_4(x) &= \text{Rotated Discus Function}(x_4), \\
                                f_5(x) &= \text{High Conditioned Elliptic Function}(x_5),
                                \end{align*}\\
                                \sigma_i = [10, 20, 30, 40, 50], \quad \lambda = [1, 1e-6, 1e-26, 1e-6, 1e-6], \quad \text{bias} = [0, 100, 200, 300, 400],\\
                                x \in [-5, 5]^5
                                """,
        'composite_function_5':r"""
                                \text{CF3}(x) = \sum_{i=1}^{3} \lambda_i \cdot f_i(x), \quad \text{where}\\
                                \begin{align*}
                                f_1(x) &= \text{Schwefel's Function}(x_1), & f_2(x) &= \text{Rotated Rastrigin's Function}(x_2), \\
                                f_3(x) &= \text{Rotated HGBat Function}(x_3),
                                \end{align*}\\
                                \sigma_i = [20, 20, 20], \quad \lambda = [1, 1, 1], \quad \text{bias} = [0, 100, 200],\\
                                x \in [-5, 5]^3
                                """,
        'composite_function_6': r"""
                                \text{CF3}(x) = \sum_{i=1}^{3} \lambda_i \cdot f_i(x), \quad \text{where}\\
                                \begin{align*}
                                f_1(x) &= \text{Rotated Schwefel's Function}(x_1), & f_2(x) &= \text{Rotated Rastrigin's Function}(x_2), \\
                                f_3(x) &= \text{Rotated High Conditioned Elliptic Function}(x_3),
                                \end{align*}\\
                                \sigma_i = [10, 30, 50], \quad \lambda = [0.25, 1, 1e-7], \quad \text{bias} = [0, 100, 200],\\
                                x \in [-5, 5]^3
                                """
    }
    st.title("Algorithm Improvement Experiment")
    st.write("This page aims at creating a better method to improve BWO algorithm by decomposing algorithms into "
             "different stages along with combing each stages from different algorithms together.")

    Hybrid_algorithm = st.toggle("Select Hybrid Algorithm",False)
    if not Hybrid_algorithm:

        function_type = st.selectbox("**Select Function Type**",
                                     ["Unimodal Functions",
                                      "Multimodal Functions",
                                      "Composite Functions",
                                      "Convex Functions",
                                      "Non-Convex Functions",
                                      "Stochastic Functions",
                                      "Nonlinear Constrained Functions"])
        if function_type == "Unimodal Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  'Sphere',
                                                  'Rosenbrock',
                                                  'Griewank',
                                              ])
        elif function_type == "Multimodal Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  'Schwefel',
                                                  'Schwefel\'s 2.22',
                                                  'Ackley 1',
                                                  'Rastrigin',
                                                  'Xin-She Yang N.4',
                                                  'Styblinski-Tang',
                                              ])
        elif function_type == "Composite Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  'composite_function_1',
                                                  'composite_function_2',
                                                  'composite_function_3',
                                                  'composite_function_4',
                                                  'composite_function_5',
                                                  'composite_function_6',
                                              ])
        elif function_type == "Convex Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  'Zakharov',
                                                  'Step',
                                              ])
        elif function_type == "Non-Convex Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  "Six Hump Camel",
                                                  "Foxholes",
                                                  "Shekel 5",
                                                  "Shekel 7",
                                                  "Shekel 10",
                                              ])
        elif function_type == "Stochastic Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  "Quartic",
                                              ])
        elif function_type == "Nonlinear Constrained Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  "Penalized",
                                                  "Penalized2",
                                              ])
        else:
            st.error("Invalid function type")
            return

        expression = function_expressions[function_real_name]
        st.latex(expression)

        model_name = st.selectbox("**Select Model**",
                                  ["Beluga whale optimization", "Particle Swarm Optimization",
                                   "Differential Evolution",
                                   "Arithmetic Optimization Algorithm", "Crow Search Algorithm",
                                   "Harris Hawks Optimization"])

        Npop = st.slider("**Population Size**", 10, 100, 50)
        Max_it = st.slider("**Max Iterations**", 10, 2000, 500)

        model_stage = st.selectbox("**Select Stage**", ["Stage1", "Stage2", "Stage3"])
        # Run Optimization button
        if st.button("**Run Optimization**"):
            # 列表用于存储每次运行的最佳位置和最佳值
            results = []
            curves = []  # 如果想平均曲线，也可以存储每次的曲线

            for i in range(20):
                # 获取函数的详细信息
                function_name = Function.get_function(function_real_name)
                lb, ub, nD, fobj = Function.get_function_details(function_name)
                # 运行优化算法
                if model_name == "Beluga whale optimization":
                    if model_stage == "Stage1":
                        xposbest, fvalbest, Curve = BWO_1.exploration_phase(Npop, Max_it, lb, ub, nD,fobj)
                    elif model_stage == "Stage2":
                        xposbest, fvalbest, Curve = BWO_2.exploitation_phase(Npop, Max_it, lb, ub, nD,fobj)
                    elif model_stage == "Stage3":
                        xposbest, fvalbest, Curve = BWO_3.whale_fall_phase(Npop, Max_it, lb, ub, nD, fobj)
                    else:
                        st.error("Invalid model stage")
                        return

                elif model_name == "Particle Swarm Optimization":
                    if model_stage == "Stage1":
                        xposbest, fvalbest, Curve = CPO_1.initialization_phase(Npop, Max_it, lb, ub, nD, fobj)
                    elif model_stage == "Stage2":
                        xposbest, fvalbest, Curve = CPO_2.exploration_phase(Npop, Max_it, lb, ub, nD, fobj)
                    elif model_stage == "Stage3":
                        xposbest, fvalbest, Curve = CPO_3.exploitation_phase(Npop, Max_it, lb, ub, nD, fobj)
                    else:
                        st.error("Invalid model stage")
                        return

                # if model_name == "Differential Evolution":
                #         xposbest, fvalbest, Curve = DE.de(Npop, Max_it, lb, ub, nD, fobj)
                # if model_name == "Arithmetic Optimization Algorithm":
                #         xposbest, fvalbest, Curve = AOA.aoa(Npop, Max_it, lb, ub, nD, fobj)
                # if model_name == "Crow Search Algorithm":
                #         xposbest, fvalbest, Curve = CSA.csa(Npop, Max_it, lb, ub, nD, fobj)
                # if model_name == "Harris Hawks Optimization":
                #         xposbest, fvalbest, Curve = HHO.hho(Npop, Max_it, lb, ub, nD, fobj)
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

        function_type = st.selectbox("**Select Function Type**",
                                     ["Unimodal Functions",
                                      "Multimodal Functions",
                                      "Composite Functions",
                                      "Convex Functions",
                                      "Non-Convex Functions",

                                      "Stochastic Functions",
                                     "Nonlinear Constrained Functions"])
        if function_type == "Unimodal Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  'Sphere',
                                                  'Rosenbrock',
                                                  'Griewank',
                                              ])
        elif function_type == "Multimodal Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  'Schwefel',
                                                  'Schwefel\'s 2.22',
                                                  'Ackley 1',
                                                  'Rastrigin',
                                                  'Xin-She Yang N.4',
                                                  'Styblinski-Tang',
                                              ])
        elif function_type == "Composite Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  'composite_function_1',
                                                  'composite_function_2',
                                                  'composite_function_3',
                                                  'composite_function_4',
                                                  'composite_function_5',
                                                  'composite_function_6',
                                              ])
        elif function_type == "Convex Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  'Zakharov',
                                                  'Step',
                                              ])
        elif function_type == "Non-Convex Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  "Six Hump Camel",
                                                  "Foxholes",
                                                  "Shekel 5",
                                                  "Shekel 7",
                                                  "Shekel 10",
                                              ])
        elif function_type == "Stochastic Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  "Quartic",
                                              ])
        elif function_type == "Nonlinear Constrained Functions":
            function_real_name = st.selectbox("**Select Function**",
                                              [
                                                  "Penalized",
                                                  "Penalized2",
                                              ])
        else:
            st.error("Invalid function type")
            return

        expression = function_expressions[function_real_name]
        st.latex(expression)

        Npop = st.slider("**Population Size**", 10, 100, 50)
        Max_it = st.slider("**Max Iterations**", 10, 2000, 500)

        if st.button("**Run Optimization**"):

            run_experiment()

            # # 列表用于存储每次运行的最佳位置和最佳值
            # B1results = []
            # B2results = []
            # B3results = []
            # C1results = []
            # C2results = []
            # C3results = []
            # B1curves = []
            # B2curves = []
            # B3curves = []
            # C1curves = []
            # C2curves = []
            # C3curves = []
            #
            # for i in range(20):
            #     # 获取函数的详细信息
            #     function_name = Function.get_function(function_real_name)
            #     lb, ub, nD, fobj = Function.get_function_details(function_name)
            #     B1xposbest, B1fvalbest, B1Curve = BWO_1.exploration_phase(Npop, Max_it, lb, ub, nD, fobj)
            #     B2xposbest, B2fvalbest, B2Curve = BWO_2.exploitation_phase(Npop, Max_it, lb, ub, nD, fobj)
            #     B3xposbest, B3fvalbest, B3Curve = BWO_3.whale_fall_phase(Npop, Max_it, lb, ub, nD, fobj)
            #     C1xposbest, C1fvalbest, C1Curve = CPO_1.CPO_exploration_phase(Npop, Max_it, lb, ub, nD, fobj)
            #     # C2xposbest, C2fvalbest, C2Curve = CPO_2.CPO_exploitation_phase(Npop, Max_it, lb, ub, nD, fobj)
            #     # C3xposbest, C3fvalbest, C3Curve = CPO_3.CPO_convergence_phase(Npop, Max_it, lb, ub, nD, fobj)
            #     # 将每次运行的结果存储在列表中
            #     B1results.append({"Run": i + 1, "Best Position": B1xposbest, "Best Value": B1fvalbest})
            #     B2results.append({"Run": i + 1, "Best Position": B2xposbest, "Best Value": B2fvalbest})
            #     B3results.append({"Run": i + 1, "Best Position": B3xposbest, "Best Value": B3fvalbest})
            #     C1results.append({"Run": i + 1, "Best Position": C1xposbest, "Best Value": C1fvalbest})
            #     # C2results.append({"Run": i + 1, "Best Position": C2xposbest, "Best Value": C2fvalbest})
            #     # C3results.append({"Run": i + 1, "Best Position": C3xposbest, "Best Value": C3fvalbest})
            #     B1curves.append(B1Curve)
            #     B2curves.append(B2Curve)
            #     B3curves.append(B3Curve)
            #     C1curves.append(C1Curve)
            #     # C2curves.append(C2Curve)
            #     # C3curves.append(C3Curve)
            # # 将结果转换为 DataFrame 并显示为表格
            # B1results_df = pd.DataFrame(B1results)
            # B2results_df = pd.DataFrame(B2results)
            # B3results_df = pd.DataFrame(B3results)
            # C1results_df = pd.DataFrame(C1results)
            # # C2results_df = pd.DataFrame(C2results)
            # # C3results_df = pd.DataFrame(C3results)
            # # st.table(results_df)
            # st.dataframe(B1results_df)
            # st.dataframe(B2results_df)
            # st.dataframe(B3results_df)
            # st.dataframe(C1results_df)
            # # st.dataframe(C2results_df)
            # # st.dataframe(C3results_df)
            # # 绘制图
            # fig, ax = plt.subplots()
            # # 计算平均曲线
            # B1mean_curve = np.mean(B1curves, axis=0)
            # B2mean_curve = np.mean(B2curves, axis=0)
            # B3mean_curve = np.mean(B3curves, axis=0)
            # C1mean_curve = np.mean(C1curves, axis=0)
            # # C2mean_curve = np.mean(C2curves, axis=0)
            # # C3mean_curve = np.mean(C3curves, axis=0)
            # ax.plot(B1mean_curve, label="BWO_1")
            # ax.plot(B2mean_curve, label="BWO_2")
            # ax.plot(B3mean_curve, label="BWO_3")
            # ax.plot(C1mean_curve, label="CPO_1")
            # # ax.plot(C2mean_curve, label="CPO_2")
            # # ax.plot(C3mean_curve, label="CPO_3")
            # ax.set_xlabel("Iteration")
            # ax.set_ylabel("Best Value")
            # # Use ScalarFormatter to set the y-axis to scientific notation
            # ax.yaxis.set_major_formatter(ScalarFormatter())
            # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            # ax.legend()
            # # Optionally set grid lines for better readability
            # ax.grid()
            # st.pyplot(fig)
            # # 计算并显示平均值和标准差
            # B1mean_best_value = np.mean(B1results_df["Best Value"])
            # B1std_best_value = np.std(B1results_df["Best Value"])
            # B2mean_best_value = np.mean(B2results_df["Best Value"])
            # B2std_best_value = np.std(B2results_df["Best Value"])
            # B3mean_best_value = np.mean(B3results_df["Best Value"])
            # B3std_best_value = np.std(B3results_df["Best Value"])
            # C1mean_best_value = np.mean(C1results_df["Best Value"])
            # C1std_best_value = np.std(C1results_df["Best Value"])
            # # C2mean_best_value = np.mean(C2results_df["Best Value"])
            # # C2std_best_value = np.std(C2results_df["Best Value"])
            # # C3mean_best_value = np.mean(C3results_df["Best Value"])
            # # C3std_best_value = np.std(C3results_df["Best Value"])
            # # 将mean_best_value，std_best_value存放在一个set中
            # B1value_set = {B1mean_best_value, B1std_best_value}
            # B2value_set = {B2mean_best_value, B2std_best_value}
            # B3value_set = {B3mean_best_value, B3std_best_value}
            # C1value_set = {C1mean_best_value, C1std_best_value}
            # # C2value_set = {C2mean_best_value, C2std_best_value}
            # # C3value_set = {C3mean_best_value, C3std_best_value}
            # # 将set转换为list
            # B1value_list = list(B1value_set)
            # B2value_list = list(B2value_set)
            # B3value_list = list(B3value_set)
            # C1value_list = list(C1value_set)
            # # C2value_list = list(C2value_set)
            # # C3value_list = list(C3value_set)
            # st.write("**Optimization Summary:**")
            # st.write(f"BWO_1 Average Best Value: {B1value_list[0]}")
            # st.write(f"BWO_1 Standard Deviation of Best Value: {B1value_list[1]}")
            # st.write(f"BWO_2 Average Best Value: {B2value_list[0]}")
            # st.write(f"BWO_2 Standard Deviation of Best Value: {B2value_list[1]}")
            # st.write(f"BWO_3 Average Best Value: {B3value_list[0]}")
            # st.write(f"BWO_3 Standard Deviation of Best Value: {B3value_list[1]}")
            # st.write(f"CPO_1 Average Best Value: {C1value_list[0]}")
            # st.write(f"CPO_1 Standard Deviation of Best Value: {C1value_list[1]}")
            # # st.write(f"CPO_2 Average Best Value: {C2value_list[0]}")
            # # st.write(f"CPO_2 Standard Deviation of Best Value: {C2value_list[1]}")
            # # st.write(f"CPO_3 Average Best Value: {C3value_list[0]}")
            # # st.write(f"CPO_3 Standard Deviation of Best Value: {C3value_list[1]}")
            # #
            #









