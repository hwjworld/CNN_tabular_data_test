
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dtd_result = pd.read_csv('datasets/dtd_result.csv')

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



#
# dtd_result = dtd_result.sort_values(by='mean_relevance', ascending=False)
# top10features = []
# for idx, result in enumerate(dtd_result.values):
#     if len(top10features)>=10:
#         break
#     feature = result[0]
#     if not top10features.__contains__(feature):
#         top10features.append(feature)
#
# # dtd_result = dtd_result.drop(labels='SM6_L',axis=1, inplace=False)
# dtd_result = dtd_result.loc[dtd_result['feature'].isin(top10features)]
# # dtd_result.pop('SM6_L')
# sns.relplot(x="time", y="feature", size="relevance", hue="value_level",
#             sizes=(40, 400), alpha=.8, palette="muted",
#             height=8, data=dtd_result)


df_groupd = dtd_result.groupby(by='feature')['mean_relevance'].sum().nlargest(10)
print(df_groupd)

top10features = df_groupd.index.values.tolist()

dtd_result = dtd_result.loc[dtd_result['feature'].isin(top10features)]


dtd_result = dtd_result.sort_values(by='mean_relevance', ascending=False)
sns.barplot(dtd_result, y='feature', x='mean_relevance')
# sns.histplot(dtd_result, y='feature', x='mean_relevance')

# sns.displot(dtd_result, y='feature',  stat="percent")
# sns.catplot(x="mean_relevance",y='feature', data=dtd_result)

sns.catplot(y='feature', x='relevance', data=dtd_result, jitter=False,
            hue='value_level',
            # size='total_relevance',
            sizes=(20, 80), alpha=.5, height=5)

# sns.relplot(x='time', y='feature', col="value", data=dtd_result)
plt.show()
