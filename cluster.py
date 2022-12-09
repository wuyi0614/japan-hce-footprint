# new clustering method for nature energy revision
#

import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from preprocess import main, preprocessing
from config import VAR_LABELS


# default saving filepath
CONFIG_PATH = Path('data')
ALL_VARS = {'prefecture': 0,
            'city_area': 0,
            'city_class': 0,
            'grid_type': 0,
            'build_space': 1,
            'build_type': 0,
            'double_window': 0,
            'heat_room_num': 1,
            'res_size': 0,
            'avg_age': 1,
            'child_num': 0,
            'elderly_num': 0,
            'if_single': 0,
            'if_single_kids': 0,
            'if_couple': 0,
            'if_couple_kids': 0,
            'if_single_elderly': 0,
            'if_couple_elderly': 0,
            'if_big_family': 0,
            'if_others': 0,
            'res_income_percap': 1,
            'car_num': 0,
            'nev_num': 0,
            'dist_all': 1,
            'drive_freq': 1,
            'car_avg_fuel': 1,
            'tv_num': 0,
            'fridge_num': 0,
            'ac_num': 0,
            'app_1': 0,
            'app_2': 0,
            'app_3': 0,
            'app_4': 0,
            'app_5': 0,
            'app_6': 0,
            'app_7': 0,
            'app_8': 0,
            'app_9': 0,
            'app_10': 0,
            'app_11': 0,
            'app_12': 0,
            'app_13': 0,
            'app_14': 0,
            'app_15': 0,
            'app_16': 0,
            'app_17': 0,
            'app_18': 0,
            'app_19': 0,
            'app_20': 0,
            'app_21': 0,
            'heater_1': 0,
            'heater_2': 0,
            'heater_3': 0,
            'heater_4': 0,
            'heater_5': 0,
            'heater_6': 0,
            'heater_7': 0,
            'light_type': 0,
            'app_variety': 1,
            'app_intensity': 1,
            'avg_app_age': 1,
            'daytime_at_home': 0,
            'ener_saving_rate': 1,
            'tv_time': 0,
            'ac_time': 0,
            'light_time': 0,
            'heat_time': 0
            }

TEMP_VARS = [
    'grid_type', 'build_space', 'build_type', 'build_time',
    'res_size', 'avg_age',
    'if_single', 'if_couple', 'if_single_kids', 'if_couple_kids',
    'if_single_elderly', 'if_couple_elderly', 'if_big_family',
    'res_income', 'res_income_percap',
    'child_num', 'elderly_num',
    'car_avg_fuel', 'drive_freq', 'dist_all', 'car_num',
    'app_1', 'app_2', 'app_3', 'app_4', 'app_5', 'app_6', 'app_7', 'app_8', 'app_9', 'app_10', 'app_11', 'app_12',
    'app_13', 'app_14', 'app_15', 'app_16', 'app_17', 'app_18', 'app_19', 'app_20', 'app_21',
    'heater_1', 'heater_2', 'heater_3', 'heater_4', 'heater_5', 'heater_6', 'heater_7',
    'light_type', 'daytime_at_home', 'ener_saving_rate',
    'tv_time', 'ac_time', 'light_time', 'heat_time'
]


def get_timestamp():
    return datetime.now().strftime('%m%d%H%M')


def lasso_modelling(data: pd.DataFrame, vars: list, dep_var: str = 'emits_per',
                    min_weight=None, alpha_range: list = None,
                    max_iteration=10e3, display=True):
    if alpha_range is None:
        alpha_range = np.linspace(0.001, 0.1, 1000)

    if min_weight is None:
        min_weight = 0.05

    x = data[vars].fillna(0).copy(True).values
    y = data[dep_var].copy(True).values

    model = Lasso()
    result = GridSearchCV(model, param_grid={'alpha': alpha_range, 'max_iter': [max_iteration]}, cv=5,
                          scoring='neg_mean_absolute_error', n_jobs=2)
    result.fit(x, y)
    print('MAE: %.5f' % result.best_score_)
    print('Optimal param：\n', result.best_params_)
    alpha = result.best_params_

    # with optimal model
    la = Lasso(**alpha).fit(x, y)  # ... find the best alpha
    la_coef = pd.DataFrame(la.coef_, columns=["coef"])
    la_coef["vars"] = vars
    la_coef = la_coef.sort_values("coef", ascending=False)
    la_coef["colors"] = "#639DBC"
    la_coef.loc[la_coef.coef < 0, "colors"] = "#B6C438"
    var_after_lasso = la_coef[la_coef.coef.abs() >= min_weight]
    print(f"{len(var_after_lasso)} variables are filtered with weight={min_weight}")

    # output distribution of weights of variables
    if display:
        fig = plt.figure(figsize=(8, 10), dpi=120)
        x_range = range(len(la_coef))
        plt.barh(x_range, la_coef.coef.values, color=la_coef.colors.values, label="Lasso alpha=0.04")

        ticks = [VAR_LABELS[k] for k in la_coef.vars.values]
        plt.yticks(x_range, labels=ticks, size=9, rotation=0)
        plt.ylabel("Household features")
        plt.xlabel("Feature importance")

        plt.margins(0.01)
        plt.tight_layout()
        plt.grid(axis="x", color="grey", alpha=0.3)
        fig.savefig(CONFIG_PATH / f"variable-importance-figure-{get_timestamp()}.png", format="png", dpi=300)
        plt.show()

    return var_after_lasso


def clustering_modelling(data: pd.DataFrame, vars: list, epoch=10, display=True):
    """Modelling process for clustering"""
    # reconcile with features
    x = data[vars].fillna(0).values
    train = x.copy()  # could use fitted(post-lasso) or X (raw)

    # clustering pipeline
    param = {
        "n_clusters": 1,
        "init": "k-means++",
        "algorithm": "elkan",
        "random_state": 0
    }

    eva = []
    for n in tqdm(range(2, epoch, 1)):
        # baseline: K-Means, use n_cluster = 3 as default
        param["n_clusters"] = n
        km = KMeans(**param).fit(train)
        y_pred = km.predict(train)
        eva += [[silhouette_score(train, y_pred), davies_bouldin_score(train, y_pred)]]

    exp = pd.DataFrame(eva, columns=["silhouette_score", "calinski_harabasz_score"])
    print(exp)

    # for K-means, select the biggest sihouette_score
    n_clusters = exp.silhouette_score.values.argmax() + 2
    print(f"The finalised number of clusters is: {n_clusters}")

    # plot the iteration process
    if display:
        x_range = range(len(exp))
        fig = plt.figure(figsize=(10, 5), dpi=80)
        plt.plot(x_range, exp.silhouette_score.values, marker="^", color="darkgreen")
        plt.xticks(x_range, range(2, epoch, 1), size=12)

        plt.axvspan(n_clusters-2.5, n_clusters-1.5, 0, 0.975,
                    facecolor="none", edgecolor="red", linewidth=2, linestyle="--")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette score")  # the higher, the better
        plt.grid(axis="y", color="grey", alpha=0.3)
        plt.savefig(str(CONFIG_PATH / f'cluster-{get_timestamp()}.png'), dpi=200)
        plt.show()

    # rerun the clustering model
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(train)
    y_pred = km.predict(train)

    # make sure you're passing `cluster` back to the raw dataset
    data["cluster"] = y_pred

    # statistics
    counted = data[["KEY", "cluster"]].groupby("cluster").count()
    print(f"Counted households: {counted}")

    # check the averaged percap emissions
    counted = data[["emits_per", "cluster"]].groupby("cluster").mean()
    print(f"Counted emission percap: {counted}")

    # output
    return data.copy(True)


VALID_KEYS = ['if_single',
              'if_couple',
              'if_single_kids',
              'if_couple_kids',
              'if_single_elderly',
              'if_couple_elderly',
              'if_big_family',
              'if_others',
              "child_num",
              "elderly_num",
              "res_size",
              "res_income",
              "avg_age",
              "drive_freq",
              "emits_per"]


def cluster_validator(data: pd.DataFrame, validate_keys: list = VALID_KEYS, save=True):
    """Validate clustered data with certain criteria"""
    compare = pd.DataFrame()
    exclude = [f'app_{i}' for i in range(1, 22, 1)] + [f'heater_{i}' for i in range(1, 8, 1)]
    for i, (_, sub) in enumerate(data.groupby('cluster')):
        keys = validate_keys + ['PV'] + \
               list(set(ALL_VARS).difference(set(validate_keys)).difference(set(exclude)))
        count = sub[keys].astype(float).apply(lambda x: x.mean(), axis=0).to_frame().T
        count.loc[count.index, 'count'] = len(sub)

        # reorder the keys by, 1) validate_keys; 2) the other keys
        count['cluster'] = i
        compare = pd.concat([compare, count], axis=0)

    if save:
        compare.T.to_excel(CONFIG_PATH / f'cluster-validation-{get_timestamp()}.xlsx')

    return compare


def config_cache(config: dict, tag: str = None):
    """Make caches of configured params in modelling"""
    if tag is None:
        tag = get_timestamp()

    file = CONFIG_PATH / f'cache-config-{tag}.json'
    file.write_text(json.dumps(config))


def make_standardized_keys(keys: dict, domains: list, exclude: list, **kwargs):
    """Make keys for standardization with exclusions"""
    vars = []
    for key in domains:
        foo = keys[key]
        if set(foo).intersection(set(exclude)):
            foo = list(set(foo).difference(set(exclude)))

        # TODO: it's better to standardize every variable
        # foo = [k for k in foo if ALL_VARS.get(k)]
        vars += foo

    return vars


if __name__ == '__main__':
    # prepare the data
    if 'data' not in locals():
        data, keys = main()

    # NB. the best param is `cache-config-cluster6-weight004-no-emits-std.json`
    config = {
        'domains': ['geo', 'resident', 'house', 'relation', 'income', 'transport', 'app', 'behave'],
        'exclude': ['build_time', 'room_num'] + [f'rel_{i}' for i in range(1, 11, 1)],
        'std_keys': [],
        'lasso_param': {'min_weight': 0.04,
                        'dep_var': 'emits_per',
                        'alpha_range': list(np.linspace(0.001, 0.1, 100)),
                        'max_iteration': 10000},
        'selected_vars': []
    }
    config['std_keys'] = make_standardized_keys(keys, **config)
    scale = preprocessing(data, vars=config['std_keys'])
    # select_vars = lasso_modelling(scale, list(ALL_VARS.keys()), **config['lasso_param'])
    select_vars = lasso_modelling(scale, config['std_keys'], **config['lasso_param'])

    # clustering process›
    config['selected_vars'] = list(select_vars.vars.values)
    clustered = clustering_modelling(scale, config['selected_vars'], epoch=12)

    tag = 'cluster6-weight004-no-emits-std'
    config_cache(config, tag=tag)

    # clustered statistics validation
    data.loc[data.index, 'cluster'] = clustered['cluster']
    data.to_csv(CONFIG_PATH / f"clustered-cache-{get_timestamp()}.csv")

    # NB. DO NOT trust the validation results before you import the exact clustered cache data
    # data = pd.read_csv(Path('data') / 'clustered-cache-10231648.csv')
    compare = cluster_validator(data, save=True)

    # make cache of scaled data for propensity calculation
    scale.loc[scale.index, 'cluster'] = clustered['cluster']
    scale.loc[data.index, 'nev'] = scale['nev_num'].apply(lambda x: 1 if x > 0 else 0)
    scale.loc[data.index, 'pv'] = scale['PV'].apply(lambda x: -x+2)
    scale = scale.drop(columns=['PV'])
    scale.to_csv(Path('data') / f'cache-scaled-data-{get_timestamp()}.csv', index=False)
