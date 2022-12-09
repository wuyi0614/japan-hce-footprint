# statistics for different types of households after clustering
#

# Reminder: use `clustered-cache-<ts>.csv` as the source data
#

from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cluster import get_timestamp
from config import PREFECTURE_MAPPING, PREFECTURE_NAME_MAPPING

# env vars
CONFIG_PATH = Path('data')
IMG_PATH = CONFIG_PATH / 'img'

CLUSTER_MAPPING = {0: 'UHBF', 1: 'MCK', 2: 'UHFK', 3: 'LES', 4: 'HFK', 5: 'ULS'}
ORDERED_CLUSTER = ['UHFK', 'UHBF', 'MCK', 'ULS', 'LES', 'HFK']
DATA = pd.read_csv(Path('data') / 'clustered-cache-12091616.csv')

# emission vars
BY_USAGE = ["em_appliance", "em_car", "em_water", "em_heating", "em_kitchen", "em_cooling"]
BY_FUEL = ["em_elec", "em_city_gas", "em_lp_gas", "em_kerosene", "em_gasoline", "em_diesel"]


def export_emission_percap_distribution(data: pd.DataFrame):
    # pre-analysis on features of our interests
    pre = data[["emits_per", "cluster"]].groupby("cluster").mean().reset_index()
    sd = data[["emits_per", "cluster"]].groupby("cluster").std().reset_index()

    pre["sd"] = sd["emits_per"]
    pre = pre.sort_values("emits_per")

    # our segments
    x_range = range(len(data['cluster'].unique()))
    fig = plt.figure(figsize=(10, 3), dpi=80)
    plt.bar(x_range, pre.emits_per.values, color="darkgreen", alpha=0.3)

    for index in pre.index:
        lowy = pre.emits_per.values[index] - pre.sd.values[index] * 1
        highy = pre.emits_per.values[index] + pre.sd.values[index] * 1

        plt.plot([index, index], [lowy, highy], color="black")
        plt.scatter(index, pre.emits_per.values[index], color="black")

    plt.xticks(x_range, [CLUSTER_MAPPING[x] for x in pre.cluster.values], size=14)
    plt.yticks(size=14)
    plt.ylim(0, 6)
    plt.xlabel("Cluster", size=14)
    plt.ylabel("Emission per capita", size=14)
    plt.tight_layout()
    fig.savefig(IMG_PATH / f"dist-emission-by-cluster-{get_timestamp()}.png", format="png", dpi=300)

    # socio-economic segments: city class
    pre = data[["emits_per", "city_class"]].groupby("city_class").mean().reset_index()
    sd = data[["emits_per", "city_class"]].groupby("city_class").std().reset_index()
    pre["sd"] = sd["emits_per"]
    pre = pre.sort_values("emits_per")

    x_range = range(len(pre))
    fig = plt.figure(figsize=(10, 3), dpi=80)
    plt.bar(x_range, pre.emits_per.values, color="darkblue", alpha=0.3)

    for index in pre.index:
        lowy = pre.emits_per.values[index] - pre.sd.values[index] * 1
        highy = pre.emits_per.values[index] + pre.sd.values[index] * 1

        plt.plot([index, index], [lowy, highy], color="black")
        plt.scatter(index, pre.emits_per.values[index], color="black")

    plt.xticks(x_range, ["class-%s" % i for i in pre["city_class"].values], size=14)
    plt.yticks(size=14)
    plt.xlabel("City class", size=14)
    plt.ylabel("Emission per capita", size=14)
    plt.tight_layout()
    fig.savefig(IMG_PATH / f"dist-emission-by-city-class-{get_timestamp()}.png", format="png", dpi=300)

    # socio-economic segments: prefecture
    pre = data[["emits_per", "prefecture"]].groupby("prefecture").mean().reset_index()
    sd = data[["emits_per", "prefecture"]].groupby("prefecture").std().reset_index()
    pre["sd"] = sd["emits_per"]
    pre = pre.sort_values("emits_per")

    x_range = range(len(pre))
    fig = plt.figure(figsize=(10, 3), dpi=80)
    plt.bar(x_range, pre.emits_per.values, color="darkblue", alpha=0.3)

    for index in pre.index:
        lowy = pre.emits_per.values[index] - pre.sd.values[index] * 1
        highy = pre.emits_per.values[index] + pre.sd.values[index] * 1

        plt.plot([index, index], [lowy, highy], color="black")
        plt.scatter(index, pre.emits_per.values[index], color="black")

    plt.xticks(x_range, [PREFECTURE_NAME_MAPPING[PREFECTURE_MAPPING[i]] for i in pre["prefecture"].values],
               size=8, rotation=90)
    plt.yticks(size=14)
    plt.xlabel("Prefecture", size=14)
    plt.ylabel("Emission per capita", size=14)

    plt.margins(0.01)
    plt.tight_layout()
    fig.savefig(IMG_PATH / f"dist-emission-by-prefecture-{get_timestamp()}.png", format="png", dpi=300)

    # emission percap distribution by income
    bins = [100, 250, 500, 1000, 2000]
    data["inc_group"] = pd.cut(data["res_income"].fillna(0), bins=bins).apply(lambda x: str(x))

    pre = data[["emits_per", "inc_group"]].groupby("inc_group").mean().reset_index()
    sd = data[["emits_per", "inc_group"]].groupby("inc_group").std().reset_index()
    pre["sd"] = sd["emits_per"]

    pre = pre.sort_values("inc_group")
    x_range = range(len(pre))
    fig = plt.figure(figsize=(10, 3), dpi=80)
    plt.bar(x_range, pre.emits_per.values, color="darkblue", alpha=0.3)

    for index in pre.index:
        lowy = pre.emits_per.values[index] - pre.sd.values[index] * 1
        highy = pre.emits_per.values[index] + pre.sd.values[index] * 1

        plt.plot([index, index], [lowy, highy], color="black")
        plt.scatter(index, pre.emits_per.values[index], color="black")

    plt.xticks(x_range, ["Income-%s" % i for i in pre["inc_group"].values], size=12, rotation=0)
    plt.yticks(size=14)
    plt.ylim(0, 3.5)
    plt.xlabel("Income group", size=14)
    plt.ylabel("Emission per capita", size=14)
    plt.margins(0.01)
    plt.tight_layout()
    fig.savefig(IMG_PATH / f"dist-emission-by-income-{get_timestamp()}.png", format="png", dpi=300)

    # emission percap by household size
    pre = data[["emits_per", "res_size"]].groupby("res_size").mean().reset_index()
    sd = data[["emits_per", "res_size"]].groupby("res_size").std().reset_index()
    pre["sd"] = sd["emits_per"]

    pre = pre.sort_values("res_size")
    x_range = range(len(pre))
    fig = plt.figure(figsize=(10, 3), dpi=80)
    plt.bar(x_range, pre.emits_per.values, color="darkblue", alpha=0.3)

    for index in pre.index:
        lowy = pre.emits_per.values[index] - pre.sd.values[index] * 1
        highy = pre.emits_per.values[index] + pre.sd.values[index] * 1

        plt.plot([index, index], [lowy, highy], color="black")
        plt.scatter(index, pre.emits_per.values[index], color="black")

    plt.xticks(x_range, ["%s" % i for i in pre["res_size"].values], size=12, rotation=0)
    plt.yticks(size=14)
    plt.ylim(0, 5)
    plt.xlabel("Household size", size=14)
    plt.ylabel("Emission per capita", size=14)
    plt.margins(0.01)
    plt.tight_layout()
    fig.savefig(IMG_PATH / f"dist-emission-by-household-size-{get_timestamp()}.png", format="png", dpi=300)

    # emission percap by age
    bins = range(int(data.avg_age.min()) - 1, int(data.avg_age.max()) + 1, 10)
    data["age_group"] = pd.cut(data["avg_age"].fillna(0), bins=bins).apply(lambda x: str(x))

    pre = data[["emits_per", "age_group"]].groupby("age_group").mean().reset_index()
    sd = data[["emits_per", "age_group"]].groupby("age_group").std().reset_index()
    pre["sd"] = sd["emits_per"]
    pre = pre.sort_values("emits_per")

    x_range = range(len(pre))
    fig = plt.figure(figsize=(10, 3), dpi=80)
    plt.bar(x_range, pre.emits_per.values, color="darkblue", alpha=0.3)

    for index in pre.index:
        lowy = pre.emits_per.values[index] - pre.sd.values[index] * 1
        highy = pre.emits_per.values[index] + pre.sd.values[index] * 1

        plt.plot([index, index], [lowy, highy], color="black")
        plt.scatter(index, pre.emits_per.values[index], color="black")

    plt.xticks(x_range, ["%s" % i for i in pre["age_group"].values], size=12, rotation=0)
    plt.yticks(size=14)
    plt.ylim(0, 4)
    plt.xlabel("Age group", size=14)
    plt.ylabel("Emission per capita", size=14)
    plt.margins(0.01)
    plt.tight_layout()
    fig.savefig(IMG_PATH / f"dist-emission-by-age-{get_timestamp()}.png", format="png", dpi=300)


def stats_hh_distribution_by_cluster_prefecture():
    """Make emission percap distribution stats by prefecture"""
    rows, cols = [], []
    for c, df in DATA.groupby("cluster"):
        prec = {PREFECTURE_NAME_MAPPING[PREFECTURE_MAPPING[i]]: 0 for i in data["prefecture"].unique()}
        count = Counter(df["prefecture"].values)
        count = {PREFECTURE_NAME_MAPPING[PREFECTURE_MAPPING[k]]: v for k, v in count.items()}

        prec.update(count)
        rows += [prec]
        cols += [CLUSTER_MAPPING[c]]

    rows = pd.DataFrame(rows).T
    rows.columns = cols
    # weighted by sampling population from each prefecture
    weights = data[["KEY", "prefecture"]].groupby("prefecture").count()["KEY"].to_dict()
    weights = np.array(list(weights.values()))
    rows = rows.apply(lambda x: x / weights, axis=0).round(3)

    rows.to_excel(CONFIG_PATH / f'household-by-prefecture-cluster-{get_timestamp()}.xlsx')
    return rows


def stats_emission_by_cluster(save=True):
    sub_keys = ["emits_per", "cluster"]

    # by fuel
    by_fuel = DATA[BY_FUEL + sub_keys].copy(True)
    by_fuel = by_fuel.groupby("cluster").mean().reset_index()
    by_fuel["cluster"] = [CLUSTER_MAPPING[i] for i in by_fuel["cluster"]]
    by_fuel = by_fuel.sort_values("cluster")

    # by usage
    by_usage = DATA[BY_USAGE + sub_keys].copy(True)
    by_usage = by_usage.groupby("cluster").mean().reset_index()
    by_usage["cluster"] = [CLUSTER_MAPPING[i] for i in by_usage["cluster"]]
    by_usage = by_usage.sort_values("cluster")

    if save:
        by_fuel.round(3).to_excel(CONFIG_PATH / f'emission-by-fuel-{get_timestamp()}.xlsx')
        by_usage.round(3).to_excel(CONFIG_PATH / f'emission-by-usage-{get_timestamp()}.xlsx')

    return by_fuel, by_usage


def stats_emissions_by_potential(data: pd.DataFrame):
    """Make statistics by potentials"""

    potentials = {
        'pv': ['em_appliance', 'em_heating', 'em_water', 'em_kitchen', 'em_cooling', 'em_ex_car'],
        'nev': ['em_car'],
        'er_light': ['em_appliance'],
        'er_heat': ['em_heating'],
        'er_hot_water': ['em_water'],
        'er_cook': ['em_kitchen']
    }
    # household from the cluster with PV or not
    proc = {p: data[[p, 'cluster'] + potentials[p]].groupby(['cluster', p]).mean().reset_index()
            for p in potentials.keys()}

    # export tables into different sheets
    writer = pd.ExcelWriter(Path('data') / f'figure2-emission-by-potential-{get_timestamp()}.xlsx')
    for p, keys in potentials.items():
        df = pd.DataFrame()
        for key in keys:
            piv = proc[p][[p, "cluster", key]].pivot(p, "cluster")
            piv = piv.droplevel(None, axis=1)

            diff = piv.loc[1, :] - piv.loc[0, :]
            piv = pd.concat([piv, pd.DataFrame(diff.to_frame().T)])
            piv.index = [f"without {p}", f"with {p}", 'diff']
            piv["emission"] = key
            df = pd.concat([df, piv], axis=0)
            # write into sheets
            df = df[ORDERED_CLUSTER + ['emission']].round(3)
            df.to_excel(writer, sheet_name=p)
                                                          
    # statistics about NEV and driving needs of households
    drive_keys = ["drive_freq", "dist_all", "car_num", "em_car"]
    drive = data[['nev', 'cluster'] + drive_keys].groupby(['cluster', 'nev']).mean().reset_index()

    dr = pd.DataFrame()
    for key in drive_keys:
        piv = drive[['nev', 'cluster', key]].pivot('nev', 'cluster')
        piv = piv.droplevel(None, axis=1)

        diff = piv.loc[1, :] - piv.loc[0, :]
        piv = pd.concat([piv, pd.DataFrame(diff.to_frame().T)])
        piv.index = ["without nev", "with nev", 'diff']
        piv["type"] = key
        dr = pd.concat([dr, piv], axis=0)

    dr = dr[ORDERED_CLUSTER + ['type']]
    # output
    dr.round(3).to_excel(writer, sheet_name='driving')
    writer.save()
    return writer


if __name__ == '__main__':
    data = DATA.copy(True)
    export_emission_percap_distribution(data)
    stats_hh_distribution_by_cluster_prefecture()

    by_fuel, by_use = stats_emission_by_cluster()
    # import the raw data
    data.loc[data.index, 'nev'] = data['nev_num'].apply(lambda x: 1 if x > 0 else 0)
    data.loc[data.index, 'pv'] = data['PV'].apply(lambda x: -x+2)
    data.loc[data.index, 'cluster'] = data['cluster'].apply(lambda x: CLUSTER_MAPPING[x])

    drive = stats_emissions_by_potential(data)

