#绘制散点图
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
tips = sns.load_dataset("tips")

# style(圆，叉) , hue（颜色）, col（）  = smoker
# smoker=Yes的是圆点；smoker=No是星号
# sns.relplot(x="total_bill",y="tip",data=tips, style='smoker')
# plt.show()

#绘制线图
df = pd.DataFrame({'a':range(10), 'b':np.random.randn(10)})
# sns.relplot(x='a',y='b',kind='line',data=df)


#1.3 特殊的线图
fmri = sns.load_dataset("fmri")
# sns.relplot(x='timepoint', y='signal', kind='line', data=fmri)

# sns.relplot(x='timepoint', y='signal', kind='line', ci='sd', data=fmri)

#histplot() 直方图，
# kdeplot()，核密度估计图
# ecdfplot() 经验累积分布图
# rugplot() 和垂直刻度

penguins = pd.read_csv('datasets/penguins_size.csv')

#直方图
sns.displot(penguins, y="flipper_length_mm")
#
# sns.displot(penguins, y='flipper_length_mm', hue='species', multiple='stack')
#
# sns.displot(penguins, y='flipper_length_mm', hue='species', stat="probability")

# # 直方图的目的是通过分箱和计数观察来近似生成数据的潜在概率密度函数。核密度估计 (KDE) 是为这一问题提供了不同的解决方案。
# sns.displot(penguins, x='flipper_length_mm', kind='kde')
# # 设置bw_adjust参数可以让 KDE 图更平滑。
# sns.displot(penguins, x='flipper_length_mm', kind='kde', bw_adjust=2)
# 设置kde=True而不是kind="kde"，可以同时绘制直方图和 KDE 图
# sns.displot(penguins, x='flipper_length_mm', kde=True)

# 经验累积分布函数（ECDF） 通过每个数据点绘制了一条单调递增的曲线，使得曲线的高度反映了具有较小值的观测值的比例。
# sns.displot(penguins, x='flipper_length_mm',hue='species', kind='ecdf')

# # 二元分布图
# sns.displot(penguins, x='culmen_length_mm', y='culmen_depth_mm')
# # 二元核密度估计图，画出来的图形是等高线。
# sns.displot(penguins, x='culmen_length_mm', y='culmen_depth_mm', kind='kde')

# #jointplot() 函数为二元变量同时绘制不同图形
# sns.jointplot(data=penguins, x='culmen_length_mm', y='culmen_depth_mm')

# # jointplot()默认绘制两变量散点图和单变量直方图。
# sns.jointplot(data=penguins, x='culmen_length_mm', y='culmen_depth_mm'
#               , hue='species',
#               kind='kde')

# # pairplot()函数，为更多变量绘图。
# sns.pairplot(penguins)
# sns.pairplot(penguins, kind='kde')


#之前我们绘制的关系图都是数值变量，当数据中有类别数据（离散值）时，就需要用分类图来绘制。
# seaborn 提供 catplot() 函数来绘制分类图，有以下3种类别
# 分类散点图
# kind="strip"（默认） 等价于 stripplot()
# kind="swarm" 等价于 swarmplot()

# 分类分布图
# kind="box" 等价于 boxplot()
# kind="violin" 等价于 violinplot()
# kind="boxen" 等价于 boxenplot()

# 分类估计图
# kind="point" 等价于 pointplot()
# kind="bar" 等价于 barplot()
# kind="count" 等价于 countplot()

# #catplot()默认使用stripplot()绘图，它会用少量随机"抖动"调整分类轴上的点位置，避免所有的点都重叠在一起。
# sns.catplot(x='day', y='total_bill', data=tips)

# maybe could use this one.
# sns.catplot(y='day', x='total_bill', data=tips, jitter=False)
# equals to below one
# sns.relplot(x="day", x="total_bill", data=tips)

# 虽然jitter可以设置“抖动”，但也有可能造成数据重叠。而kind="swarm"可以绘制非重叠的分类散点图。
# sns.catplot(y='day', x='total_bill', data=tips,kind='swarm')

# # kind="box" 可以绘制箱线图
# sns.catplot(y='day', x='total_bill', data=tips, kind='box')
# # kind="boxen" 可以绘制增强箱线图。
# sns.catplot(y='day', x='total_bill', data=tips, kind='boxen')
# diamonds = sns.load_dataset("diamonds", data_home='seaborn-data', cache=True)
# sns.catplot(x="color", y="price", kind="boxen", data=diamonds.sort_values("color"))

# kind="violin" 可以绘制小提琴图。
# sns.catplot(x="day", y="total_bill", hue="sex", kind="violin", split=True, data=tips)
# sns.catplot(x="day", y="total_bill", hue="sex", kind="violin", data=tips)

titanic = sns.load_dataset("titanic", data_home='seaborn-data', cache=True)
# # kind="bar" 以矩形条的方式展示数据点估值(默认取平均值)和置信区间,该置信区间使用误差线绘制。
# sns.catplot(x="sex", y="survived",hue='class', kind='bar', data=titanic)

# # kind="count" 是常见的柱状图，统计x坐标对应的数据量。
# sns.catplot(x="deck", kind="count", data=titanic)

# # kind="point" 绘制点图，展示数据点的估计值（默认平均值）和置信区间，并连接来自同一hue类别的点。
# sns.catplot(x="sex", y="survived", hue="class", kind="point", data=titanic)

#seaborn 提供线性回归函数对数据拟合，包括regplot()和lmplot()，它俩大部分功能是一样的，只是输入的数据和输出图形稍有不同。
# 用lmplot()函数可以绘制两个变量x、y的散点图，拟合回归模型并绘制回归线和该回归的 95% 置信区间。
# sns.lmplot(x="total_bill", y="tip", data=tips)

# anscombe = sns.load_dataset("anscombe", data_home='seaborn-data', cache=True)
#
# sns.lmplot(x="x", y="y",data=anscombe.query('dataset=="II"'))
# sns.lmplot(x="x", y="y",data=anscombe.query('dataset=="II"'),order=2)

# logistic=True参数可以拟合逻辑回归模型 , have problem
# sns.lmplot(x="total_bill", y="big_tip", data=tips, logistic=True, y_jitter=.03)

# FacetGrid类可以同时绘制多图
# g = sns.FacetGrid(tips, row='sex', col='smoker')
# g.map(sns.scatterplot, 'total_bill','tip')
# # equals to below
# sns.relplot(x='total_bill', y='tip', row="sex", col="smoker", data=tips)

# # 当然用FaceGrid的好处是可以像 matplotlib 那样设置很多图形属性。
# g = sns.FacetGrid(tips, row='sex', col='smoker')
# g.map(sns.scatterplot, 'total_bill','tip')
# g.set_axis_labels('Total bill', 'Tip')
# g.set(xticks=[10,30,50], yticks=[2,6,10])
# g.figure.subplots_adjust(wspace=.02, hspace=.02)

#PairGrid，可以为多变量同时绘图，且图形种类可以不同。
# iris = sns.load_dataset("iris", data_home='seaborn-data', cache=True)
# g = sns.PairGrid(iris)
# g.map_upper(sns.scatterplot)
# g.map_lower(sns.kdeplot)
# g.map_diag(sns.kdeplot, lw=3, legend=False)
#
#

#
# sns.set_theme(style="white")
# # Load the example mpg dataset
# mpg = sns.load_dataset("mpg")
#
# # Plot miles per gallon against horsepower with other semantics
# sns.relplot(x="horsepower", y="mpg", size="origin", hue="weight",
#             sizes=(40, 400), alpha=.5, palette="muted",
#             height=6, data=mpg)
#
# sns.catplot(y='day', x='total_bill', data=tips, jitter=False, hue='total_bill'
#             ,sizes=(40, 400), alpha=.5,)


plt.show()
