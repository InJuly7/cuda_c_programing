import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义 block 维度和对应的执行时间
data = [
    ((1, 1), 4671.247237),
    ((1, 16), 735.396594),
    ((1, 32), 598.521632),
    ((1, 64), 609.651617),
    ((1, 128), 623.264519),
    ((1, 256), 661.578814),
    ((1, 512), 890.033589),
    ((1, 1024), 1067.201286),
    ((16, 1), 288.472699),
    ((16, 16), 102.481308),
    ((16, 32), 116.317098),
    ((16, 64), 155.005515),
    ((32, 1), 138.985113),
    ((32, 16), 100.427545),
    ((32, 32), 109.866213),
    ((64, 1), 79.236188),
    ((64, 16), 104.362450),
    ((128, 1), 78.362286),
    ((256, 1), 78.593040),
    ((512, 1), 80.937715),
    ((1024, 1), 87.626326)
]

# 为散点图准备数据
block_dim_x = [x for ((x, y), time) in data]
block_dim_y = [y for ((x, y), time) in data]
execution_times = [time for ((x, y), time) in data]

# 对 X 和 Y 应用对数变换，对 Z 应用平方根变换
log_block_dim_x = np.log2(block_dim_x)
log_block_dim_y = np.log2(block_dim_y)
sqrt_execution_times = np.sqrt(execution_times)

# 创建3D图
fig = plt.figure(figsize=(10, 8))  # 调整图像尺寸为10x8英寸
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
sc = ax.scatter(log_block_dim_x, log_block_dim_y, sqrt_execution_times,
                c=sqrt_execution_times, cmap='viridis', s=100, edgecolors='k', alpha=0.6)  # 点的大小为100


# 添加颜色条
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Square Root of Execution Time (ms)')

for i, txt in enumerate(sqrt_execution_times):
    ax.text(log_block_dim_x[i], log_block_dim_y[i], sqrt_execution_times[i], '%.2f' % txt, 
            color='red', fontsize=8)  # 调整字体大小为8

ax.set_xlabel('Log2 Block Dim X')
ax.set_ylabel('Log2 Block Dim Y')
ax.set_zlabel('Square Root of Execution Time (ms)')

plt.title('3D Scatter Plot of Kernel Execution Time')

# 显示图像
# plt.show()

# 可以选择保存图像
plt.savefig('3d_scatter_plot_log_sqrt.png')
plt.close(fig)
