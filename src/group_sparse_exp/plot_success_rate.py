import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

plot_color = [
    "b",  # 蓝色
    "r",  # 红色
    "g",  # 绿色
    "c",  # 青色
    "m",  # 品红
    "y",  # 黄色
    "k",  # 黑色
    "#FFA500",  # 橙色
    "#800080",  # 紫色
    "#FFD700",  # 金色
    "#00FF7F",  # 春绿色
    "#FF4500",  # 橙红色
    "#1E90FF",  # 道奇蓝
    "#8A2BE2",  # 蓝紫色
    "#FF1493"  # 深粉色
]
marker_list=['x','x','x','x','o','o','o','o','o','o','o']

def load_df(path_: str,sheet_name):
    df = pd.read_excel(path_, sheet_name=sheet_name)  # l2_1+l1	l2_1over2_l1over2	l2_2over3+l2over3	l1 mcp
    df.columns = [
        'Sparse-Level','Sparsity',
        r'$L_{2,|\cdot|_0}+\ell_0$',
        r'$L_{2,|\cdot|}+\ell_1$',
        r'$L_{2,|\cdot|^{1/2}}+\ell_{1/2}$',
        r'$L_{2,|\cdot|^{2/3}}+\ell_{2/3}$',
        r'$L_{1,|\cdot|^{1/2}}$',
        r'$L_{1,|\cdot|^{2/3}}$',
        r'$L_{1,{\rm MCP}}$',
        r'$L_{1,{\rm SCAD}}$',
        r'$L_{1,{\rm TL1}}$'
    ]
    return df
def plot_fig(path_1: str,path_2):
    df=load_df(path_1,'glen8')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.subplots_adjust(hspace=0.3)  #
    for index, value in enumerate(df.columns[2:].to_list()):
        linestyle = '-'
        ax1.plot(df['Sparse-Level'] * 100,df[value], color=plot_color[index], label=value ,linestyle=linestyle, marker=marker_list[index])

    df = load_df(path_2,'glen16')
    for index, value in enumerate(df.columns[2:].to_list()):
        linestyle = '-'
        ax2.plot(df['Sparse-Level'] * 100,df[value], color=plot_color[index], label=value ,linestyle=linestyle, marker=marker_list[index])
    ax1.set_xlim(left=0)
    ax2.set_xlim(left=0)
    ax1.set_ylabel('Success Rate')
    ax1.set_xlabel('Sparsity Level')
    ax2.set_ylabel('Success Rate')
    ax2.set_xlabel('Sparsity Level')
    ax1.set_xticks(ticks=np.linspace(0, 30, 7), labels=['0%', '5%', '10%', '15%', '20%', '25%','30%'])
    ax2.set_xticks(ticks=np.linspace(0, 30, 7), labels=['0%', '5%', '10%', '15%', '20%', '25%','30%'])

    lines, labels = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=11)
    # plt.legend()
   # plt.xticks(ticks=np.linspace(0, 30, 7), labels=['0%', '5%', '10%', '15%', '20%', '25%','30%'])
    #
    plt.savefig(dpi=400, fname='success_rate_20250923.png')
    plt.show()
    return df


path_1 = r'C:\Users\1\Desktop\1psi20250909.xlsx'
path_2 = r'C:\Users\1\Desktop\1psi20250909.xlsx'
plot_fig(path_1,path_2)
