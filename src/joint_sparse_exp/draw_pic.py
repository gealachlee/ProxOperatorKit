import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

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


import pandas as pd
import munch

total_results1 =pd.read_excel('./exp/tmpres_result1.xlsx')

total_results2 = pd.read_excel('./exp/tmpres_result2.xlsx')

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
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

name_list = [
    r'$L_{2,|\cdot|_0}+\ell_0$',
    r'$L_{2,|\cdot|}+\ell_1$',
    r'$L_{2,|\cdot|^{1/2}}+\ell_{1/2}$',
    r'$L_{2,|\cdot|^{2/3}}+\ell_{2/3}$',
   #
    r'$L_{1,|\cdot|^{1/2}}$',
    r'$L_{1,|\cdot|^{2/3}}$',
    r'$L_{1,{\rm MCP}}$',
   r'$L_{1,{\rm SCAD}}$',
    r'$L_{1,{\rm TL1}}$'
]

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(hspace=0.5)  #
for index, (name, loss) in enumerate(total_results1.to_dict().items()):
    # linestyle = '--' if 'Joint' in name else '-'
    ax1.plot(total_results1[total_results1.columns[index]], color=plot_color[index], label=name_list[index], linestyle='-')
    ax1.set_yscale('log')
    ax1.set_yticks([10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4, 10 ** -5],
                   [r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$'])

ax1.set_ylabel('Relative Error',fontdict={'fontsize': 12})
ax1.set_xlabel('Iteration',fontdict={'fontsize': 12})

for index, (name, loss) in enumerate(total_results2.to_dict().items()):
    # linestyle = '--' if 'Joint' in name else '-'
    ax2.plot(total_results2[total_results2.columns[index]], color=plot_color[index], label=name_list[index], linestyle='-')
    ax2.set_yscale('log')
    ax2.set_yticks([10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4,10**-5],
               [r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$'])
ax2.set_ylabel('Relative Error',fontdict={'fontsize': 12})
ax2.set_xlabel('Iteration',fontdict={'fontsize': 12})
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.52, 0.97), ncol=9,fontsize=12)

# 显示图形
plt.tight_layout(rect=[0, 0, 1, 0.9])
#
plt.savefig('./exp/joint1e-2.png', dpi=400)
plt.show()
