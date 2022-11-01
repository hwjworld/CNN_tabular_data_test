import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

dtd_result = pd.read_csv('datasets/dtd_result.csv')

df_groupd = dtd_result.groupby(by='feature')['mean_relevance'].sum().nlargest(10)
print(df_groupd)

top10features = df_groupd.index.values.tolist()
dtd_result = dtd_result.loc[dtd_result['feature'].isin(top10features)]

dtd_result = dtd_result.sort_values(by='mean_relevance', ascending=False)
sns.barplot(dtd_result, y='feature', x='mean_relevance')

norm = plt.Normalize(dtd_result['value'].min(), dtd_result['value'].max())
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])

ax = sns.catplot(y='feature', x='relevance', data=dtd_result, jitter=False,
                 hue='value',
                 palette="coolwarm",
                 # size='total_relevance',
                 sizes=(20, 80), alpha=.5, height=5)

ax.legend.remove()
ax.figure.colorbar(sm, location="right")  # 'left', 'right', 'top', 'bottom'

# sns.relplot(x='time', y='feature', col="value", data=dtd_result)
plt.show()
