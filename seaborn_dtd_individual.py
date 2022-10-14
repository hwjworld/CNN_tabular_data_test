
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dtd_result = pd.read_csv('datasets/dtd_result.csv')
dtd_result = pd.read_csv('datasets/dtds/p2.csv')

# sns.set_theme(style="darkgrid")
# sns.relplot(x="time",y="feature",data=dtd_result, style='smoker')
# sns.displot(dtd_result, x='time', y='feature')
# sns.jointplot(dtd_result, x='time', y='feature')
# sns.jointplot(data=dtd_result, x='time', y='feature', hue='value_level')


# sns.catplot(x='time', y='feature', data=dtd_result, hue='value_level')
# sns.catplot(x='time', y='feature', data=dtd_result, jitter=False
# sns.catplot(x='time', y='feature', data=dtd_result, hue='value_level', kind='box')
# sns.catplot(x='time', y='feature', kind="boxen", data=dtd_result.sort_values("relevance"))
# sns.catplot(x='time', y='feature', kind="bar")
# sns.catplot(x='time', y='feature', data=dtd_result, hue='value_level', kind='point')


# mpg = sns.load_dataset("mpg")

# sns.relplot(x="horsepower", y="mpg", size="origin", hue="weight",
#             sizes=(40, 400), alpha=.5, palette="muted",
#             height=6, data=mpg)

dtd_result = dtd_result.sort_values(by='mean_relevance', ascending=False)
top10features = []
for idx, result in enumerate(dtd_result.values):
    if len(top10features)>=10:
        break
    feature = result[0]
    if not top10features.__contains__(feature):
        top10features.append(feature)

# dtd_result = dtd_result.drop(labels='SM6_L',axis=1, inplace=False)
dtd_result = dtd_result.loc[dtd_result['feature'].isin(top10features)]
g = sns.relplot(y="time", x="feature", size="relevance", hue="value_level",
# g = sns.relplot(="feature", size="relevance", hue="value_level",
            sizes=(30, 300), alpha=.8, palette="muted",
            height=2, aspect=5, data=dtd_result)

# g.xti
# g.tick_params(axis='x')
# sns.relplot(x='time', y='feature', col="value", data=dtd_result)
plt.show()
