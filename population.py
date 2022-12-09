# using OECD population data for analysis
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import WSJ

hist = pd.read_csv('oecd-historic-population.csv')
proj = pd.read_csv('oecd-projection-population.csv')

# 美国、英国、法国、德国、日本、意大利和加拿大
G7 = ['United States', 'United Kingdom', 'France', 'Germany', 'Italy', 'Canada', 'Japan']

# display the aging trend
hi = hist[(hist.Sex == 'Total') & hist.Country.isin(G7)]
pr = proj[(proj.Sex == 'Total') & proj.Country.isin(G7)]

array = None
for c, sub in hi.groupby('Country'):
    sub_hi = sub[sub.AGE == '65_OVER_SHARE']
    if array is None:
        array = sub_hi[['Country', 'Time', 'Value']].values
    else:
        array = np.vstack([array, sub_hi[['Country', 'Time', 'Value']].values])

    sub_proj = pr[(pr.AGE == '65_OVER_SHARE') & (pr.Country == c) & (pr.Time >= 2021)]
    array = np.vstack([array, sub_proj[['Country', 'Time', 'Value']].values])

# draw the pic (lineplot)
data = pd.DataFrame(array, columns=['country', 'year', 'over65_share'])

color=['#EA423A','royalblue','#B6C438','gray', '#F5B243', 'darkgreen', '#C3842D']
markers = ['o', '+', '^', '*', 's', 'v', 'x']

x_range = np.arange(26)
fig = plt.figure(figsize=(10, 6), dpi=80)
for idx, (c, sub) in enumerate(data.groupby('country')):
    plt.plot(x_range[:16], sub.over65_share.values[:16], marker='', linewidth=4, color=color[idx], label=c)
    plt.plot(x_range[15:], sub.over65_share.values[15:], marker='', linewidth=4, linestyle='--', color=color[idx])

plt.xticks(x_range, data.year.unique(), size=12, rotation=45)
plt.legend(fontsize=12)

plt.xlabel('Year', size=14)
plt.ylabel('Over 65 population share (%)', size=14)  # the higher, the better
plt.grid(axis="y", color="grey", alpha=0.3)
plt.show()
