import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dtd_result = pd.read_csv('datasets/dtds/p12.csv')

# s7 start ---- below is the simulated 7 rows,(one patient, 7 timestamps)
# dtd_result = dtd_result.sort_values(by='mean_relevance', ascending=False)
# s7 end

dtd_result = dtd_result.sort_values(by='relevance', ascending=False)
top10features = []
for idx, result in enumerate(dtd_result.values):
    if len(top10features) >= 10:
        break
    feature = result[0]
    if not top10features.__contains__(feature):
        top10features.append(feature)

dtd_result = dtd_result.loc[dtd_result['feature'].isin(top10features)]

norm = plt.Normalize(dtd_result['value'].min(), dtd_result['value'].max())
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])

ax = sns.relplot(y="time", x="feature", size="relevance", hue="value",
                 sizes=(30, 300), alpha=.8, palette="coolwarm",
                 height=2, aspect=5, data=dtd_result)

ax.legend.remove()
ax.figure.colorbar(sm)

plt.show()
