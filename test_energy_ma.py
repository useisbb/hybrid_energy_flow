# 本文件编写代码，用于验证计算的功率参数是否正确；这个一个能源管理系统，连接了电网，储能，光伏和负载，记录了每一时刻的总电能量（负载无电能量表）
# 读取2026-02-24\GRID_METER.csv文件，'time'列为时间，分辨率为15秒，但只用读取整数分钟的数据；'正向有功总电能'列为电网正向总电能；'反向有功总电能'为列电网反向总电能，电网总电能='正向有功总电能'-'反向有功总电能'
# 读取2026-01-24\PCS_METER.csv文件，'time'列为时间，分辨率为15秒，但只用读取整数分钟的数据；'正向有功总电量累计'列为储能正向总电能；'反向有功总电量累计'为列储能反向总电能，储能总电能='正向有功总电量累计'-'反向有功总电量累计'
# 增加读取2026-01-24\EMS.csv文件，'time'列为时间，分辨率为15秒，但只用读取整数分钟的数据；‘储能SOC(%)’ 为储能SOC曲线。
# 然后，基于以上数据，可以计得到'虚拟负载电能'=电网总电能+储能总电能

# 首先，编写代码，通过以上数据，读取指定时间段（2026年1月29日14：00至2026年1月29日15：00）的以上所有数据，只保留整数分钟的数据.
# 可以计算得到'计算储能功率'=储能总电能与上分钟储能总电能的差值*60;
# 增加一个需量计量周期参数demand_time = 30,代表需量计量周期为30分钟，每小时的0，30分钟作为计量周期起始点
# 那么'计算电网功率'每分钟的功率数值=当前时刻电网总电能与计量周期起点电网总电能的差值*60/(当前分钟-计量周期起始分钟)
# 同理，计算负载功率'=当前时刻'虚拟负载电能'与计量周期起点电能的差值*60/(当前分钟-计量周期起始分钟)
# 将以上读取的数据和计算的结果，一起另存为result.csv文件。
# 绘图显示以上计算的结果曲线。
# 保留这些文字，在下面区域编程。先理解我的需求，不清楚的问我，得到肯定回答后再编程。

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 需量计量周期参数（分钟）
demand_time = 30

# 读取数据文件
print("正在读取数据文件...")
grid_df = pd.read_csv('2026-02-25/GRID_METER.csv')
pcs_df = pd.read_csv('2026-02-25/PCS_METER.csv')
ems_df = pd.read_csv('2026-02-25/EMS.csv')

# 转换时间列为datetime格式
grid_df['time'] = pd.to_datetime(grid_df['time'])
pcs_df['time'] = pd.to_datetime(pcs_df['time'])
ems_df['time'] = pd.to_datetime(ems_df['time'])

# 筛选时间范围：2026-01-29 14:00-15:00，只保留整数分钟的数据
start_time = pd.to_datetime('2026-02-25 20:00:00')
end_time = pd.to_datetime('2026-02-25 22:30:00')

# 筛选时间范围内的数据
grid_df = grid_df[(grid_df['time'] >= start_time) & (grid_df['time'] <= end_time)]
pcs_df = pcs_df[(pcs_df['time'] >= start_time) & (pcs_df['time'] <= end_time)]
ems_df = ems_df[(ems_df['time'] >= start_time) & (ems_df['time'] <= end_time)]

# 重置索引，保持CSV文件的原始行顺序
grid_df = grid_df.reset_index(drop=True)
pcs_df = pcs_df.reset_index(drop=True)
ems_df = ems_df.reset_index(drop=True)

# 创建分钟级时间列（去掉秒），用于去重
grid_df['time_minute'] = grid_df['time'].dt.floor('min')
pcs_df['time_minute'] = pcs_df['time'].dt.floor('min')
ems_df['time_minute'] = ems_df['time'].dt.floor('min')

# 按原始顺序,保留每分钟的第一条数据（去重）
grid_df = grid_df.drop_duplicates(subset='time_minute', keep='first').reset_index(drop=True)
pcs_df = pcs_df.drop_duplicates(subset='time_minute', keep='first').reset_index(drop=True)
ems_df = ems_df.drop_duplicates(subset='time_minute', keep='first').reset_index(drop=True)

# 删除辅助列，保留原始time列
grid_df = grid_df.drop('time_minute', axis=1)
pcs_df = pcs_df.drop('time_minute', axis=1)
ems_df = ems_df.drop('time_minute', axis=1)

print(f"电网数据: {len(grid_df)} 条")
print(f"储能数据: {len(pcs_df)} 条")
print(f"EMS数据: {len(ems_df)} 条")

# 计算总电能
# 电网总电能 = 正向 - 反向
grid_df['电网总电能'] = grid_df['正向有功总电能'] - grid_df['反向有功总电能']

# 储能总电能 = 正向 - 反向
pcs_df['储能总电能'] = pcs_df['正向有功总电能'] - pcs_df['反向有功总电能']

# 合并数据（基于时间）
result_df = pd.merge(grid_df[['time', '电网总电能']], pcs_df[['time', '储能总电能']], on='time', how='outer')
result_df = pd.merge(result_df, ems_df[['time', '储能SOC(%)']], on='time', how='outer')
result_df = result_df.sort_values('time').reset_index(drop=True)

# 计算虚拟负载电能
result_df['虚拟负载电能'] = result_df['电网总电能'] + result_df['储能总电能']

# 计算储能功率（逐分钟差值 × 60）
result_df['计算储能功率'] = result_df['储能总电能'].diff() * 60

# 计算电网功率和负载功率（基于30分钟需量计量周期）
result_df['分钟'] = result_df['time'].dt.minute
result_df['小时'] = result_df['time'].dt.hour

# 确定每个时刻所属的计量周期起点（0, 30分钟）
def get_demand_period_start(minute):
    """获取需量计量周期起点分钟数"""
    if minute < 30:
        return 0
    else:
        return 30

result_df['计量周期起点分钟'] = result_df['分钟'].apply(get_demand_period_start)

# 初始化功率列
result_df['计算电网功率'] = 0.0
result_df['计算负载功率'] = 0.0

# 按小时分组计算
for hour in result_df['小时'].unique():
    hour_mask = result_df['小时'] == hour

    # 对每个计量周期起点进行计算
    for period_start in [0, 30]:
        # 找到该周期的数据
        period_mask = hour_mask & (result_df['计量周期起点分钟'] == period_start)
        period_data = result_df[period_mask]

        if len(period_data) > 0:
            # 获取周期起点的电能值
            start_idx = period_data.index[0]
            grid_energy_start = result_df.loc[start_idx, '电网总电能']
            load_energy_start = result_df.loc[start_idx, '虚拟负载电能']

            # 计算该周期内每个时刻的功率
            for idx in period_data.index:
                current_minute = result_df.loc[idx, '分钟']
                time_diff = current_minute - period_start

                if time_diff > 0:
                    # 电网功率 = (当前电能 - 起点电能) * 60 / 时间差
                    grid_energy_current = result_df.loc[idx, '电网总电能']
                    result_df.loc[idx, '计算电网功率'] = (grid_energy_current - grid_energy_start) * 60 / time_diff

                    # 负载功率 = (当前电能 - 起点电能) * 60 / 时间差
                    load_energy_current = result_df.loc[idx, '虚拟负载电能']
                    result_df.loc[idx, '计算负载功率'] = (load_energy_current - load_energy_start) * 60 / time_diff
                else:
                    # 周期起点时刻，计算与上一分钟的差值
                    if idx > 0:
                        # 电网功率 = (当前电能 - 上一分钟电能) * 60
                        grid_energy_current = result_df.loc[idx, '电网总电能']
                        grid_energy_prev = result_df.loc[idx - 1, '电网总电能']
                        result_df.loc[idx, '计算电网功率'] = (grid_energy_current - grid_energy_prev) * 60

                        # 负载功率 = (当前电能 - 上一分钟电能) * 60
                        load_energy_current = result_df.loc[idx, '虚拟负载电能']
                        load_energy_prev = result_df.loc[idx - 1, '虚拟负载电能']
                        result_df.loc[idx, '计算负载功率'] = (load_energy_current - load_energy_prev) * 60
                    else:
                        # 第一个数据点，功率为0
                        result_df.loc[idx, '计算电网功率'] = 0.0
                        result_df.loc[idx, '计算负载功率'] = 0.0

# 删除辅助列
result_df = result_df.drop(['分钟', '小时', '计量周期起点分钟'], axis=1)

# 保存结果到CSV文件
output_file = 'result.csv'
result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到: {output_file}")

# 计算需量（每隔30分钟）
# 筛选出每隔30分钟的数据点（0, 30分钟）
result_df['分钟'] = result_df['time'].dt.minute
demand_df = result_df[result_df['分钟'].isin([0, 30])].copy()
demand_df = demand_df.sort_values('time').reset_index(drop=True)

# 计算电网侧需量 = 电网总电能差值 * 2
demand_df['电网侧需量'] = demand_df['电网总电能'].diff() * 2

# 计算负载侧需量 = 虚拟负载电能差值 * 2
demand_df['负载侧需量'] = demand_df['虚拟负载电能'].diff() * 2

# 删除第一行（没有差值）
demand_df = demand_df.dropna(subset=['电网侧需量', '负载侧需量']).reset_index(drop=True)

print(f"\n需量数据点: {len(demand_df)} 个")

# 找到最大值
grid_max_idx = demand_df['电网侧需量'].idxmax()
load_max_idx = demand_df['负载侧需量'].idxmax()

grid_max_time = demand_df.loc[grid_max_idx, 'time']
grid_max_value = demand_df.loc[grid_max_idx, '电网侧需量']

load_max_time = demand_df.loc[load_max_idx, 'time']
load_max_value = demand_df.loc[load_max_idx, '负载侧需量']
print(demand_df)
print(f"电网侧需量最大值: {grid_max_value:.2f} kW，时间: {grid_max_time}")
print(f"负载侧需量最大值: {load_max_value:.2f} kW，时间: {load_max_time}")

# 绘制需量曲线图
fig_demand = plt.figure(figsize=(15, 6))
ax_demand = fig_demand.add_subplot(111)

# 绘制电网侧需量
ax_demand.plot(demand_df['time'], demand_df['电网侧需量'], 'g-', linewidth=2.5,
               label='电网侧需量', marker='o', markersize=6)

# 绘制负载侧需量
ax_demand.plot(demand_df['time'], demand_df['负载侧需量'], 'r-', linewidth=2.5,
               label='负载侧需量', marker='s', markersize=6)

# 标记电网侧需量最大值
ax_demand.plot(grid_max_time, grid_max_value, 'g*', markersize=20,
               markeredgecolor='darkgreen', markeredgewidth=2, zorder=5)
ax_demand.annotate(f'最大值: {grid_max_value:.2f} kW\n{grid_max_time.strftime("%H:%M")}',
                   xy=(grid_max_time, grid_max_value),
                   xytext=(20, 20), textcoords='offset points',
                   fontsize=11, fontweight='bold', color='darkgreen',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                   color='darkgreen', lw=2))

# 标记负载侧需量最大值
ax_demand.plot(load_max_time, load_max_value, 'r*', markersize=20,
               markeredgecolor='darkred', markeredgewidth=2, zorder=5)
ax_demand.annotate(f'最大值: {load_max_value:.2f} kW\n{load_max_time.strftime("%H:%M")}',
                   xy=(load_max_time, load_max_value),
                   xytext=(20, -40), textcoords='offset points',
                   fontsize=11, fontweight='bold', color='darkred',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                   color='darkred', lw=2))

ax_demand.set_xlabel('时间', fontsize=13)
ax_demand.set_ylabel('需量 (kW)', fontsize=13)
ax_demand.set_title('电网侧与负载侧需量分析 (30分钟计量周期)', fontsize=15, fontweight='bold')
ax_demand.grid(True, alpha=0.3, linestyle='--')
ax_demand.legend(fontsize=12, loc='best')
ax_demand.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('demand_analysis_result.png', dpi=300, bbox_inches='tight')
print(f"\n需量图表已保存到: demand_analysis_result.png")


# 绘图显示结果
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

fig.suptitle('能源管理系统功率分析', fontsize=16, y=0.995)

# 子图1：电网功率和负载功率（占50%篇幅）
ax1 = fig.add_subplot(gs[0])
ax1.plot(result_df['time'], result_df['计算电网功率'], 'g-', linewidth=2, label='电网功率', marker='o', markersize=3)
ax1.plot(result_df['time'], result_df['计算负载功率'], 'r-', linewidth=2, label='负载功率', marker='s', markersize=3)
ax1.set_xlabel('时间', fontsize=12)
ax1.set_ylabel('功率 (kW)', fontsize=12)
ax1.set_title('电网功率与负载功率 (30分钟需量计量)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='best')
ax1.tick_params(axis='x', rotation=45)

# 子图2：储能功率
ax2 = fig.add_subplot(gs[1])
ax2.plot(result_df['time'], result_df['计算储能功率'], 'b-', linewidth=2, label='储能功率', marker='o', markersize=3)
ax2.set_xlabel('时间', fontsize=12)
ax2.set_ylabel('功率 (kW)', fontsize=12)
ax2.set_title('储能功率', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=10, loc='best')
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)  # 添加零线

# 子图3：储能SOC
ax3 = fig.add_subplot(gs[2])
ax3.plot(result_df['time'], result_df['储能SOC(%)'], 'purple', linewidth=2, label='储能SOC', marker='o', markersize=3)
ax3.set_xlabel('时间', fontsize=12)
ax3.set_ylabel('SOC (%)', fontsize=12)
ax3.set_title('储能SOC变化曲线', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(fontsize=10, loc='best')
ax3.tick_params(axis='x', rotation=45)
ax3.set_ylim([0, 100])  # SOC范围0-100%

plt.savefig('energy_analysis_result.png', dpi=300, bbox_inches='tight')
print(f"图表已保存到: energy_analysis_result.png")
plt.show()

print("\n数据统计:")
print(f"储能功率范围: {result_df['计算储能功率'].min():.2f} ~ {result_df['计算储能功率'].max():.2f} kW")
print(f"电网功率范围: {result_df['计算电网功率'].min():.2f} ~ {result_df['计算电网功率'].max():.2f} kW")
print(f"负载功率范围: {result_df['计算负载功率'].min():.2f} ~ {result_df['计算负载功率'].max():.2f} kW")
