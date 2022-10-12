import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_simu_dtd = pd.read_csv("datasets/simulate_dtd.csv")
sns.set_theme(style="darkgrid")

sns.relplot(x="time", y="feature", size="relevance", hue="value",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=df_simu_dtd)
# sns.relplot(y='feature',x='time',kind='')
plt.show()