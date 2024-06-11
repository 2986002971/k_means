import streamlit as st
from matplotlib import pyplot as plt
import matplotlib
import time

from k_means import generate_data1, generate_data2, k_means_euclid_c, k_means_manhattan_c, k_means_euclid_py, \
    k_means_manhattan_py

# 在终端输入 streamlit run ./main.py 启动
st.title("K聚类算法可视化")

# 初始化或获取session_state中的状态
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'dataX' not in st.session_state:
    st.session_state.dataX = None
if 'dataY' not in st.session_state:
    st.session_state.dataY = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'centroids' not in st.session_state:
    st.session_state.centroids = None

with st.sidebar:
    data_mode = st.selectbox("数据模式", ["模式1", "模式2"])
    n_samples = st.number_input("样本量", 10, 100000, 1000, 10)
    distance_metric = st.selectbox("距离度量", ["欧几里得", "曼哈顿"])
    K = st.number_input("聚类数量", 1, 100, 9, 1)
    max_iters = st.number_input("最大迭代次数", 1, 100000, 10000, 1000)
    implementation = st.selectbox("实现方式", ["DLL", "Python"])

    if st.button("生成数据"):
        if data_mode == "模式1":
            dataX, dataY = generate_data1(n_samples)
        else:
            dataX, dataY = generate_data2(n_samples)
        st.session_state.dataX = dataX
        st.session_state.dataY = dataY
        st.session_state.data_generated = True

if st.session_state.data_generated:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(st.session_state.dataX[:, 0], st.session_state.dataX[:, 1], c='b', marker='o', label='data points')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('DataSet')
    ax.legend()
    st.pyplot(fig)

    if st.button("开始聚类"):
        start_time = time.time()

        if implementation == "DLL":
            if distance_metric == "欧几里得":
                labels, centroids = k_means_euclid_c(st.session_state.dataX, K, max_iters)
            else:
                labels, centroids = k_means_manhattan_c(st.session_state.dataX, K, max_iters)
        else:
            if distance_metric == "欧几里得":
                labels, centroids = k_means_euclid_py(st.session_state.dataX, K, max_iters)
            else:
                labels, centroids = k_means_manhattan_py(st.session_state.dataX, K, max_iters)

        end_time = time.time()
        elapsed_time = end_time - start_time

        st.session_state.labels = labels
        st.session_state.centroids = centroids

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = matplotlib.colormaps['tab10']
        for k in range(K):
            cluster_points = st.session_state.dataX[st.session_state.labels == k]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(k), label=f'Cluster {k + 1}')
        ax.scatter(st.session_state.centroids[:, 0], st.session_state.centroids[:, 1], c='black', marker='x', s=100,
                   label='Centroids')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Clustered DataSet')
        ax.legend()
        st.pyplot(fig)

        st.write(f"算法运行时间: {elapsed_time:.4f} 秒")

st.write("感谢使用Streamlit进行可视化！")
