import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#
#
# def plot_fig(path: str):
#     try:
#         df = pd.read_excel(path)  #	l2_1+l1	l2_1over2_l1over2	l2_2over3+l2over3	l1 mcp
#         df.columns = ['Sparsity', 'Sparse-Level', r"$L_{2,1}+L_{1}$",
#                       r"$L_{2,\frac{1}{2}}+L_{\frac{1}{2}}$",
#                       r"$L_{2,\frac{2}{3}}+L_{\frac{2}{3}}$",
#                       r"$L_{1,\frac{1}{2}}$",
#                       r"$L_{1,\frac{2}{3}}$",
#                       r"$L_{1,MCP}$",
#                       r"$L_{1,SCAD}$",
#                       r"$L_{1,TL1}$"
#                       ]
#     except:
#         raise pd.errors.DataError("Failed to read file")
#     x = list(range(len(df)))
#     # for i in df.columns[2:].to_list():
#     plt.plot(df['Sparse-Level'] * 100, df[ r"$L_{2,1}+L_{1}$"], label=df[ r"$L_{2,1}+L_{1}$"].name, color='r', marker='x',
#              linestyle='--')
#     plt.plot(df['Sparse-Level'] * 100, df[r"$L_{2,\frac{1}{2}}+L_{\frac{1}{2}}$"], label=df[r"$L_{2,\frac{1}{2}}+L_{\frac{1}{2}}$"].name, color='b', marker='x',
#              linestyle='--')
#     plt.plot(df['Sparse-Level'] * 100, df[r"$L_{2,\frac{2}{3}}+L_{\frac{2}{3}}$"], label=df[r"$L_{2,\frac{2}{3}}+L_{\frac{2}{3}}$"].name, color='g', marker='x',
#              linestyle='--')
#     plt.plot(df['Sparse-Level'] * 100, df[r"$L_{1,\frac{1}{2}}$"], label=df[r"$L_{1,\frac{1}{2}}$"].name, color='b', marker='o',
#              linestyle='-')
#     plt.plot(df['Sparse-Level'] * 100, df[ r"$L_{1,\frac{2}{3}}$"], label=df[ r"$L_{1,\frac{2}{3}}$"].name, color='g', marker='o',#o * x
#              linestyle='-')
#     plt.plot(df['Sparse-Level'] * 100, df[ r"$L_{1,MCP}$"], label=df[ r"$L_{1,MCP}$"].name, color='orange', marker='o',
#              linestyle='-')
#     plt.plot(df['Sparse-Level'] * 100, df[ r"$L_{1,SCAD}$"], label= df[ r"$L_{1,SCAD}$"].name, color='black', marker='o',
#              linestyle='-')
#     plt.plot(df['Sparse-Level'] * 100, df[r"$L_{1,TL1}$"], label=df[r"$L_{1,TL1}$"].name, color='purple',
#              marker='o',
#              linestyle='-')
#     plt.legend()
#     plt.xticks(ticks=np.linspace(0, 25, 6), labels=['0%', '5%', '10%', '15%', '20%', '25%'])
#
#     plt.savefig(dpi=300, fname='success_rate.png')
#     plt.show()
#     return df
#
#
# path = r'.\joint_sp_success_rate.xlsx'
# plot_fig(path)





def plot_fig(path: str):
    try:
        df = pd.read_excel(path,sheet_name='glen8')  #	l2_1+l1	l2_1over2_l1over2	l2_2over3+l2over3	l1 mcp
        df.columns = [ 'Sparse-Level','Sparsity',
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
    except:
        raise pd.errors.DataError("Failed to read file")
    x = list(range(len(df)))
    # for i in df.columns[2:].to_list():
    plt.plot(df['Sparse-Level']*100, df[r'$L_{2,|\cdot|_0}+\ell_0$'], label=df[r'$L_{2,|\cdot|_0}+\ell_0$'].name, color='b', marker='x',
             linestyle='--')

    plt.plot(df['Sparse-Level'] *100, df[r'$L_{2,|\cdot|}+\ell_1$'], label=df[r'$L_{2,|\cdot|}+\ell_1$'].name, color='r', marker='x',
             linestyle='--')

    plt.plot(df['Sparse-Level'] * 100, df[r'$L_{2,|\cdot|^{1/2}}+\ell_{1/2}$'], label=df[r'$L_{2,|\cdot|^{1/2}}+\ell_{1/2}$'].name,
             color='b', marker='x',
             linestyle='--')
    plt.plot(df['Sparse-Level'] * 100, df['$L_{2,|\cdot|^{2/3}}+\ell_{2/3}$'], label=df['$L_{2,|\cdot|^{2/3}}+\ell_{2/3}$'].name,
             color='b', marker='x',
             linestyle='--')


    plt.plot(df['Sparse-Level']*100 , df[r'$L_{1,|\cdot|^{1/2}}$'], label=df[r'$L_{1,|\cdot|^{1/2}}$'].name, color='m', marker='o',
             linestyle='-')
    plt.plot(df['Sparse-Level']*100 , df[r'$L_{1,|\cdot|^{2/3}}$'], label=df[r'$L_{1,|\cdot|^{2/3}}$'].name, color='y', marker='o',
             linestyle='-')
    plt.plot(df['Sparse-Level']*100, df[r'$L_{1,{\rm MCP}}$'], label=df[r'$L_{1,{\rm MCP}}$'].name, color='k', marker='o',
             linestyle='-')
    plt.plot(df['Sparse-Level']*100 , df[r'$L_{1,{\rm SCAD}}$'], label=df[r'$L_{1,{\rm SCAD}}$'].name, color='#FFA500', marker='o',
             linestyle='-')
    plt.plot(df['Sparse-Level']*100 , df[r'$L_{1,{\rm TL1}}$'], label=df[r'$L_{1,{\rm TL1}}$'].name, color='#800080', marker='o',
             linestyle='-')
    plt.legend()

    plt.xticks(ticks=np.linspace(0, 30, 7), labels=['0%', '5%', '10%', '15%', '20%', '25%', '30%'])

    plt.savefig(dpi=300, fname='success_rate20250923.png')
    plt.show()
    return df


path = r'C:\Users\1\Desktop\1psi20250909.xlsx'#r'.\joint_sp_success_rate.xlsx'
plot_fig(path)
