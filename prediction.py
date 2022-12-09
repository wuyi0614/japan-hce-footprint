# prediction codes for the latest version of revision on Nature Energy
#

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from preprocess import CONFIG_PATH
from stats import CLUSTER_MAPPING, get_timestamp, BY_USAGE
from config import FUTURE_ENERGY_INTENSITY, FUTURE_LINEAR_PENETRATION_RATE, FUTURE_AMBITIOUS_PENETRATION_RATE, \
    FUTURE_GRID_EMISSION_FACTOR, HOUSEHOLD_NUM_Y2018

# env vars
DATA = pd.read_csv(CONFIG_PATH / 'clustered-cache-12091616.csv')
CONFIG = json.loads((CONFIG_PATH / 'cache-config-cluster6-weight004-no-emits-std.json').read_text())
CONTROL_VARS = CONFIG['selected_vars'] + ['ln_percap_inc']
HOUSEHOLD_TYPES = ['if_single', 'if_couple', 'if_couple_kids', 'if_couple_elderly',
                   'if_single_elderly', 'if_single_kids', 'if_others', 'if_big_family']
ENERGY_USES = ['eu_appliance', 'eu_cooling', 'eu_heating', 'eu_water', 'eu_kitchen']


# regression part
def regress_by_potential(data: pd.DataFrame,
                         dep_var: str,
                         indep_var: list,
                         control_vars: list = None,
                         save: bool = True):
    """Get regression coefs of specific potential emission reduction options"""
    if control_vars is None:
        control_vars = CONTROL_VARS

    est_pv = pd.DataFrame()
    for ekey in indep_var:
        formula = f"{ekey} ~ "
        formula += f"{dep_var} + "
        formula += " + ".join(control_vars)
        row = []
        for i, cluster in CLUSTER_MAPPING.items():
            est = smf.ols(formula=formula, data=data[data['cluster'] == i]).fit()
            item = {'cons': est.params.loc["Intercept"],
                    'lower': est.conf_int(alpha=0.05).loc[dep_var].values[0],
                    'upper': est.conf_int(alpha=0.05).loc[dep_var].values[1],
                    f'{dep_var}_coef': est.params.loc[dep_var],
                    f'{dep_var}_tval': est.tvalues.loc[dep_var],
                    f'{dep_var}_pval': est.pvalues.loc[dep_var]}
            for k in ['ln_percap_inc']:
                add = {f'{k}_coef': est.params.loc[k],
                       f'{k}_tval': est.tvalues.loc[k],
                       f'{k}_pval': est.pvalues.loc[k]}
                item.update(add)

            item.update({'r2': est.rsquared_adj, 'cluster': cluster})
            row += [item]

        row = pd.DataFrame(row, index=(range(len(row))))
        row["emtype"] = ekey
        row = row.sort_index()
        est_pv = pd.concat([est_pv, row], axis=0)

    # output est. results
    if save:
        est_pv.to_excel(CONFIG_PATH / f'est-{dep_var}-coef-{get_timestamp()}.xlsx')

    return est_pv


def projection_population(pop: pd.ExcelFile, data: pd.DataFrame):
    """Projection future population of Japan based on Japanese population data"""

    # 1st step: confirm the prob distribution and population by age and household structure by cluster
    def fold_df_last_row(df: pd.DataFrame):
        d = df.iloc[1:15, 1:].copy(True)
        d.loc[14, :] = df.loc[14, :] + df.loc[16, :]
        for k in d.columns:
            d.loc[d.index, k] = d[k] * 1000

        d["age"] = ['(14, 19]', '(19, 24]', '(24, 29]', '(29, 34]', '(34, 39]', '(39, 44]', '(44, 49]',
                    '(49, 54]', '(54, 59]', '(59, 64]', '(64, 69]', '(69, 74]', '(74, 79]', '(79, 84]']
        return d

    # 2nd step: extract population according to demographic structure
    def get_pop(df: pd.DataFrame):
        proj = fold_df_last_row(df)
        data["reltype"] = data[HOUSEHOLD_TYPES].apply(lambda x: HOUSEHOLD_TYPES[list(x).index(1)], axis=1)

        bins = range(14, 85, 5)
        data["age_group"] = pd.cut(data["avg_age"].fillna(0), bins=bins).apply(lambda x: str(x))

        if_couple_elderly = proj.loc[10: 14, ["age", "couple"]]
        if_couple_elderly["couple"] = if_couple_elderly["couple"].values + proj.loc[10:14, "couple_with_kids"].values
        if_couple_elderly["rel"] = "if_couple_elderly"
        if_couple_elderly.columns = ["age", "pop", "rel"]

        pop_by_age_cluster = pd.DataFrame()
        pop_by_age_cluster = pd.concat([pop_by_age_cluster, if_couple_elderly], axis=0)

        if_single_elderly = proj.loc[10: 14, ["age", "single"]]
        if_single_elderly["single"] = if_single_elderly["single"].values + proj.loc[10:14, "single_with_kids"].values
        if_single_elderly["rel"] = "if_single_elderly"
        if_single_elderly.columns = ["age", "pop", "rel"]
        pop_by_age_cluster = pd.concat([pop_by_age_cluster, if_single_elderly], axis=0)

        if_couple = proj.loc[:9, ["age", "couple"]]
        if_couple["rel"] = "if_couple"
        if_couple.columns = ["age", "pop", "rel"]

        if_single = proj.loc[:9, ["age", "single"]]
        if_single["rel"] = "if_single"
        if_single.columns = ["age", "pop", "rel"]

        pop_by_age_cluster = pd.concat([pop_by_age_cluster, if_couple], axis=0)
        pop_by_age_cluster = pd.concat([pop_by_age_cluster, if_single], axis=0)

        if_couple_kids = proj.loc[:9, ["age", "couple_with_kids"]]
        if_couple_kids["rel"] = "if_couple_kids"
        if_couple_kids.columns = ["age", "pop", "rel"]
        if_single_kids = proj.loc[:9, ["age", "single_with_kids"]]
        if_single_kids["rel"] = "if_single_kids"
        if_single_kids.columns = ["age", "pop", "rel"]

        pop_by_age_cluster = pd.concat([pop_by_age_cluster, if_couple_kids], axis=0)
        pop_by_age_cluster = pd.concat([pop_by_age_cluster, if_single_kids], axis=0)

        counted = data[["KEY", "reltype"]].groupby("reltype").count()
        bf_others_count = counted.T["if_big_family"].values[0] + counted.T["if_others"].values[0]
        bf_prob = counted.T["if_big_family"].values[0] / bf_others_count
        others_prob = counted.T["if_others"].values[0] / bf_others_count

        if_big_family = bf_prob * proj[["others"]]
        if_big_family["age"] = proj["age"].values
        if_big_family["rel"] = "if_big_family"
        if_big_family.columns = ["pop", "age", "rel"]

        others = others_prob * proj[["others"]]
        others["age"] = proj["age"].values
        others["rel"] = "if_others"
        others.columns = ["pop", "age", "rel"]

        pop_by_age_cluster = pd.concat([pop_by_age_cluster, if_big_family], axis=0)
        pop_by_age_cluster = pd.concat([pop_by_age_cluster, others], axis=0)
        return pop_by_age_cluster

    # 3rd, map the distribution of household types to clusters
    def mapping_pop():
        # compute the percentages of age and household types in each cluster
        data["reltype"] = data[HOUSEHOLD_TYPES].apply(lambda x: HOUSEHOLD_TYPES[list(x).index(1)], axis=1)
        bins = range(14, 85, 5)
        data["age_group"] = pd.cut(data["avg_age"].fillna(0), bins=bins).apply(lambda x: str(x))

        sub_keys = ["KEY", "reltype", "cluster", "age_group"]
        sub = data[sub_keys].copy(True)
        sub2 = sub[sub_keys].groupby(["cluster", "reltype", "age_group"]).count().fillna(0).reset_index()

        prop = pd.DataFrame()
        for rel, g in sub2.groupby("reltype"):
            g["ratio"] = g["KEY"].values / g["KEY"].sum()
            prop = pd.concat([prop, g], axis=0)

        prop.columns = ["cluster", "rel", "age", "KEY", "ratio"]
        return prop

    def merge_population(tar: pd.DataFrame, proportion: pd.DataFrame):
        result = pd.DataFrame()
        for c, g in proportion.groupby("rel"):
            g = g.merge(tar, on=["rel", "age"], how="left")
            g = g.fillna(0)
            tot = tar.loc[tar["rel"] == c, "pop"].sum()
            g["total"] = g["pop"].values * g["ratio"].values
            if g["total"].sum() < tot:
                g["total"] = g["total"].values + (tot - g["total"].sum()) * g["ratio"].values

            result = pd.concat([result, g], axis=0)

        result["cluster"] = [CLUSTER_MAPPING[i] for i in result["cluster"]]
        return result

    def estimate_total_emission(pop_by_cluster, mean_cluster):
        result, indexes = [], []
        for c, g in pop_by_cluster.groupby("cluster"):
            total = g["total"].sum()
            emit = mean_cluster.loc[c].values[0]
            row = dict(total_emits=emit * total)
            result += [row]
            indexes += [c]

        result = pd.DataFrame(result)
        result.index = indexes
        return result

    # load the population prediction datasets, sheets: 2025, 2030, 2035, 2040, unit: 1000 household
    mean_emission_cluster = data[["emits_per", "cluster"]].groupby("cluster").mean()
    mean_emission_cluster.index = [CLUSTER_MAPPING[i] for i in mean_emission_cluster.index]

    emissions, population = pd.DataFrame(), pd.DataFrame()
    for year in [2025, 2030, 2035, 2040]:
        df = pop.parse(f'{year}')
        df.columns = ["ages", "total", "single", "couple", "couple_with_kids", "single_with_kids", "others"]
        df = df.loc[1:16, ]  # column 2~15 represents 15~84
        del df["ages"]

        # start population projection
        pop_by_age_cluster = get_pop(df)
        prop = mapping_pop()
        pop_by_cluster = merge_population(pop_by_age_cluster, prop)

        foo = pop_by_cluster[['cluster', 'total']].groupby('cluster').sum()
        foo = foo.T.reset_index(drop=True)
        foo['year'] = year
        population = pd.concat([population, foo], axis=0)

        est_emi = estimate_total_emission(pop_by_cluster, mean_emission_cluster)
        est_emi["year"] = year
        emissions = pd.concat([emissions, est_emi], axis=0)

    # show results
    output = emissions.reset_index().pivot("year", "index")
    return population, output


def projection_emission_reduction_by_activity(data: pd.DataFrame,
                                              pscore: pd.DataFrame,
                                              pv: dict,
                                              nev: dict,
                                              years: list = [2018, 2025, 2030, 2035, 2040]):
    """Projection emission changes from activities including lighting, heating, water heating, cooling and kitchen

    However, according to the literature, they can be categorized into:
    Appliance: heating, cooling, water heating
    Lighting: lighting
    Kitchen: cooking
    """

    # first, post-processing emissions per household from clusters
    def get_er_from_energy_intensity(emission: dict,
                                     pv: dict,
                                     nev: dict,
                                     pvf: int,
                                     nevf: int,
                                     cluster: str,
                                     year: int,
                                     tag: str):
        """Get emission reductions from energy intensity changes by scenario levels

        The formula is: E1 = E0 - ΣRA - △EFECxPVxEEC - (1-NEV)xENEV + EFECxΣRA
        """
        em_elec = emission.pop('em_elec')
        ef_delta = FUTURE_GRID_EMISSION_FACTOR[year] - FUTURE_GRID_EMISSION_FACTOR[2018]
        # NB. change in energy intensity
        efce = (ef_delta + pv[cluster]) * em_elec if pvf else ef_delta * em_elec
        ce = (1 + nev[cluster]) * emission['em_car'] if nevf else 0

        scenarios = {}
        for scenario, intensity in FUTURE_ENERGY_INTENSITY.items():
            ra = 0
            for key, ratio in intensity.items():
                ra += ratio[int((year - 2025) / 5)] * emission[key] if year != 2018 else 0
                # assign zero if the activity isn't defined
                # TODO: due to the difference between
                # raec += ratio[level] * energy.get(f'eu_{key.split("_")[-1]}', 0) * FUTURE_GRID_EMISSION_FACTOR[year]
            assert ra >= 0, f'Invalid emission reduction value: {ra}'
            scenarios[f'{scenario}-ra'] = -ra
            scenarios[f'{scenario}-post'] = sum(list(emission.values())) - ra + efce - ce

        scenarios[f'{tag}-efce'] = efce
        scenarios[f'{tag}-nev'] = nev[cluster] * emission['em_car'] if nevf else 0
        return scenarios

    def find_adopter(penetration):
        pvq = penetration['pv'][year]
        nevq = penetration['nev'][year]

        # filter households
        filtered = data.copy(True)
        pvpscore = pscore.copy(True)
        pvpscore = pvpscore[pvpscore['build_type'] == 1]

        filtered.loc[filtered.index, 'pv_filter'] = 0
        filtered.loc[
            pvpscore[pvpscore['pv_pscore'] >= pvpscore['pv_pscore'].quantile(q=1 - pvq)].index, 'pv_filter'] = 1
        filtered.loc[filtered.index, 'nev_filter'] = 0
        filtered.loc[pscore['nev_pscore'] >= pscore['nev_pscore'].quantile(q=1 - nevq), 'nev_filter'] = 1
        return filtered

    def projection(filtered, tag: str):
        series = []
        for c, group in filtered.groupby('cluster'):
            # transform emission types into a dictionary
            for i, row in group[BY_USAGE].iterrows():
                pvf = filtered.loc[i, 'pv_filter']
                nevf = filtered.loc[i, 'nev_filter']

                emission = row.to_dict()
                raw = sum(list(emission.values()))
                emission['em_elec'] = data.loc[i, 'em_elec']

                # energy data (embodied energy consumption from activities)
                # energy = data.loc[i, ENERGY_USES].to_dict()
                post_emission = get_er_from_energy_intensity(emission, pv, nev, pvf, nevf, CLUSTER_MAPPING[c], year,
                                                             tag)
                post_emission['baseline'] = raw
                post_emission['cluster'] = c
                series += [post_emission]

        # compute by-cluster mean values
        foo = pd.DataFrame(series)
        foo = foo.groupby('cluster').mean().T.reset_index()
        foo.columns = ['scenario'] + list(CLUSTER_MAPPING.values())
        foo['year'] = year
        return foo

    # second, calculate each household by cluster
    pred = pd.DataFrame()
    for count, year in enumerate(years):
        # NB. use penetration rate here and `build_type` to filter the selected households
        linear = find_adopter(FUTURE_LINEAR_PENETRATION_RATE)
        ambitious = find_adopter(FUTURE_AMBITIOUS_PENETRATION_RATE)
        li = projection(linear, tag='LINEAR')
        am = projection(ambitious, tag='ambitious')
        pred = pd.concat([pred, li, am.loc[6:7, :]], axis=0)

    pred = pred.reset_index(drop=True)
    return pred


def projection_emission(data: pd.DataFrame, reduction: pd.DataFrame, population: pd.DataFrame, save: bool = True):
    """Projection emissions for scenarios: baseline, low, mid, and high"""
    raw = reduction.loc[8, list(CLUSTER_MAPPING.values())].T  # raw emission percap in 2018
    # cluster distribution without demographic change
    weight = data[['KEY', 'cluster']].groupby('cluster').count().T.values / len(data)

    project = []
    for year, group in reduction.groupby('year'):
        p = population.loc[population['year'] == year]
        # without demographic change
        row_without_demo = raw[list(CLUSTER_MAPPING.values())] * p.values.flatten()[:6].sum() * weight.flatten()
        project += [row_without_demo.tolist() + [year, 'baseline']]

        # with demographic change
        row_with_demo = raw[list(CLUSTER_MAPPING.values())].values.flatten()[:6] * p.values.flatten()[:6]
        project += [row_with_demo.tolist() + [year, 'demographic']]

        # demographic + scenario
        for s, sub in group.groupby('scenario'):
            row = sub[list(CLUSTER_MAPPING.values())].values.flatten()[:6] * p.values.flatten()[:6]
            project += [row.tolist() + [year, s]]

    project = pd.DataFrame(project, columns=list(CLUSTER_MAPPING.values()) + ['year', 'scenario'])
    project['total'] = project.iloc[:, :6].sum(axis=1)
    if save:
        project.to_excel(CONFIG_PATH / f'figure3-projection-emission-{get_timestamp()}.xlsx', index=False)

    return project


if __name__ == '__main__':
    # tests on regressions
    data = DATA.copy(True)
    data.loc[data.index, 'pv'] = - (data['PV'].values - 2)
    data.loc[data.index, 'build_type'] = - (data['build_type'].values - 2)
    data['ln_percap_inc'] = np.log(data['res_income_percap'].values + 1)
    est_pv = regress_by_potential(data, 'pv', ['em_ex_car'], save=True)

    # test on NEV
    data.loc[data.index, 'nev'] = data['nev_num'].apply(lambda x: 1 if x > 0 else 0)
    est_nev = regress_by_potential(data, 'nev', ['em_car'], save=True)

    # load up the population dataset
    reader = pd.ExcelFile(CONFIG_PATH / 'japan-pop-predict.xlsx')
    pop, emission = projection_population(reader, data)
    pop = pop[list(CLUSTER_MAPPING.values()) + ['year']]
    # add 2018 to the population dataset
    foo = data[['KEY', 'cluster']].groupby('cluster').count()
    foo = (foo / foo.sum() * HOUSEHOLD_NUM_Y2018).T.reset_index(drop=True)
    foo.columns = CLUSTER_MAPPING.values()
    foo['year'] = 2018
    population = pd.concat([foo, pop], axis=0)
    population.to_excel(CONFIG_PATH / f'japan-pop-projection-cluster-{get_timestamp()}.xlsx')

    # configure grid emission factor data and projection specification
    pv = pd.read_excel(Path('data') / 'est-pv-coef-11061347.xlsx')
    pv.index = pv['cluster']
    pv = pv['coef'].to_dict()

    nev = pd.read_excel(Path('data') / 'est-nev-coef-11061347.xlsx')
    nev.index = nev['cluster']
    nev = nev['coef'].to_dict()

    ef = pd.read_excel(Path('data') / 'grid-emission-factor.xlsx')
    pscore = pd.read_csv(Path('data') / 'cache-pscore-data.csv')
    pscore.loc[pscore.index, 'nev_pscore'] = pscore['nev_pscore'].fillna(0)
    pscore.loc[pscore.index, 'pv_pscore'] = pscore['pv_pscore'].fillna(0)
    reduction = projection_emission_reduction_by_activity(data, pscore, pv, nev)

    # make projection
    projection = projection_emission(data, reduction, pop, save=False)
