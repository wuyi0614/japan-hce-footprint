# Classes for data processing
#

from pathlib import Path
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np

from loguru import logger
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from config import QS2_2_MAPPING, INCOME_MAPPING, CAR_FUEL_MAPPING, CAR_USE_MAPPING


# default data path
DATA_PATH = Path('data-local')
CONFIG_PATH = Path('data')

RAW = pd.read_csv(DATA_PATH / "japan_18.csv", header=[1], dtype="object")


def get_timestamp():
    return datetime.now().strftime('%m%d%H%M')


# create class-based data generators
class BaseLoader(object):
    """base data loader for all categories"""

    def __init__(self):
        self.codes = []
        self.vars = []
        self.data = None

    def run(self, *args):
        pass

    def validate(self):
        name = self.__class__.__name__.replace("Loader", "")
        check = self.data.dropna(how="any")
        assert check.size > 0, f"[{name}] NaN value issues"
        logger.warning(f'[{name}] {len(self.data)}/{len(RAW)} after processing.')


class GeographyLoader(BaseLoader):
    """load geographic information about households"""

    def __init__(self):
        super().__init__()
        # NB. add `E_F11_2`, grid company info
        self.codes = ["PREFECTUER", "AREA_FLG", "CITY_CLASS", "E_F11_2"]
        self.vars = ["prefecture", "city_area", "city_class", "grid_type"]

    def run(self):
        dat = RAW[self.codes].astype(int)
        dat.columns = self.vars
        dat["KEY"] = RAW["KEY"]
        self.data = dat
        self.validate()
        return dat


class HouseBasicLoader(BaseLoader):
    """load household basic data"""

    def __init__(self):
        super().__init__()
        self.codes = ["H_F4", "H_F5", "NH_F8", "H_F9", "NQW7"]
        self.vars = ["build_type", "build_time", "room_num", "double_window", "heat_room_num"]

    def run(self):
        space_codes = ["NH_F7_1", "NH_F7_2"]
        space = RAW[space_codes]
        # scaler source: https://baike.baidu.com/item/%E5%9D%AA/11027235?fr=aladdin
        space = space.apply(lambda x: float(x[0]) if float(x[0]) else float(x[1]) * 3.3057, axis=1).to_frame()

        house = pd.concat([space, RAW[self.codes].fillna(0).astype(int)], axis=1)  # -1 means NaNs
        house.columns = ["build_space"] + self.vars
        house['KEY'] = RAW['KEY']

        # skip those without building areas
        house = house.dropna(how='any')
        self.data = house
        self.validate()
        return house


class ResidentLoader(BaseLoader):
    """load household type data"""
    codes = [f'H_F2_A_{i}' for i in range(1, 11, 1)]
    vars = [f'res_{i}' for i in range(1, 11, 1)]

    def __init__(self):
        pass

    def run(self):
        resident = RAW[self.codes]
        resident = resident.fillna(0).astype(int)
        resident.columns = self.vars
        resident['KEY'] = RAW['KEY']

        # proceed with household size
        size = RAW['H_F1'].astype(int).to_frame()
        size['KEY'] = RAW['KEY']
        size.columns = ["res_size", "KEY"]
        size = size[size.res_size > 0]

        # proceed with resident age
        convert_opt_age = lambda seq: [np.mean(QS2_2_MAPPING.get(int(x), [0, 0])) for x in seq]
        age = resident.apply(convert_opt_age, axis=0)
        age["avg_age"] = age[self.vars].apply(lambda x: x.astype(float)[x.astype(float) > 0].mean(), axis=1)
        age['KEY'] = RAW['KEY']
        age = age[age.avg_age > 0]

        # add child_num and elderly_num
        child_rule = lambda x: len([i for i in x.values if i in [1, 2]])
        elderly_rule = lambda x: len([i for i in x.values if i in [7, 8, 9]])
        child_elderly = RAW[['KEY']].copy(True)
        child_elderly["child_num"] = resident[self.vars].apply(child_rule, axis=1)
        child_elderly["elderly_num"] = resident[self.vars].apply(elderly_rule, axis=1)
        child_elderly = child_elderly.fillna(0)

        # merge
        dat = size.merge(age, on='KEY', how='left') \
            .merge(child_elderly, on='KEY', how='left')
        dat = dat.dropna(how='all')
        self.data = dat
        self.validate()
        return self.data


class RelationshipLoader(BaseLoader):
    """load relationship data between residents"""

    def __init__(self):
        super().__init__()
        self.codes = [f'H_F2_{i}' for i in range(1, 11, 1)]
        self.vars = [f'rel_{i}' for i in range(1, 11, 1)]
        self.types = []

    def run(self, resident: pd.DataFrame):
        relation = RAW[self.codes]
        relation.columns = self.vars
        relation = relation.fillna(0).astype(int)

        # rule 1: single
        single = lambda x: 1 if x[1] == 0 else 0
        relation["if_single"] = relation[self.vars].apply(single, axis=1)

        # rule 1.1: single with kids
        def single_with_kids(x):
            # NB: updated rule for `single_with_kids` with unlimited number of kids + single parent
            r1 = set(x[:2]) == {1, 3} or set(x[:2]) == {1, 6} or set(x[:2]) == {5, 6}
            r2 = set(x[:2]) == {2, 3} or set(x[:2]) == {2, 6}
            r3 = set(x[2:]) == {0, 3} or set(x[2:]) == {0, 6} or set(x[2:]) == {0}
            return 1 if any([r1, r2]) and r3 else 0

        relation["if_single_kids"] = relation[self.vars].apply(single_with_kids, axis=1)

        # rule 2: couple
        def couple(x):
            r1 = set(x[:2]) == {1, 2} and x[2] == 0  # host couple
            r2 = set(x[:2]) == {3}  # children couple
            r3 = set(x[:2]) == {4}  # parents
            r4 = set(x[:2]) == {5}  # grandparents
            return 1 if any([r1, r2, r3, r4]) else 0

        relation["if_couple"] = relation[self.vars].apply(couple, axis=1)

        # rule 3: couple with kids
        def couple_with_kids(x):
            x = x[x > 0]
            counted = Counter(x)
            r1 = set(x) == {1, 2, 3} or set(x) == {1, 2, 6}
            r2 = counted[5] > 1 and counted[6] > 0
            return 1 if any([r1, r2]) else 0

        relation["if_couple_kids"] = relation[self.vars].apply(couple_with_kids, axis=1)

        # cross `resident size` and `resident age`
        cross = relation[self.vars].copy(True).values.astype(int)
        cross[cross > 1] = 1
        # ... multiplied by age labels, elderly is defined to be over 60
        cross = cross * resident[ResidentLoader.vars].fillna(0).values.astype(int)
        cross = pd.DataFrame(cross)
        cross.columns = ResidentLoader.vars

        # rule 4: single elderly
        single_elderly = lambda x: 1 if x[0] > 60 else 0
        relation["if_single_elderly"] = cross[ResidentLoader.vars].apply(single_elderly, axis=1) * \
                                        relation["if_single"].values

        # rule 5: elderly couple no kids
        couple_elderly = lambda x: 1 if (x[0] > 60 and x[1] > 60) else 0
        relation["if_couple_elderly"] = cross[ResidentLoader.vars].apply(couple_elderly, axis=1) * \
                                        relation["if_couple"].values

        # adjust `single`, `couple` and `elderly` overlapping
        relation["if_single"][relation["if_single_elderly"] == 1] = 0
        relation["if_couple"][relation["if_couple_elderly"] == 1] = 0

        # rule 6: big family (families with 5 or more members)
        rel_types = ["if_single", "if_couple", "if_single_kids", "if_couple_kids",
                     "if_single_elderly", "if_couple_elderly"]
        big_family = lambda x: 1 if len(list(filter(lambda i: i > 0, x))) > 4 else 0
        relation["if_big_family"] = relation[self.vars].apply(big_family, axis=1)
        for key in rel_types:
            relation[key][relation["if_big_family"] == 1] = 0

        # rule 7: others
        rel_types += ["if_big_family"]
        others = lambda x: 1 if sum(x) == 0 else 0
        relation["if_others"] = relation[rel_types].apply(others, axis=1)

        # multiple choices check:
        check = relation[rel_types].apply(lambda x: x.sum(), axis=1)
        assert check[check > 1].empty, "having multiple binary values"
        rel_types += ["if_others"]
        self.types = rel_types

        relation.loc[RAW.index, "KEY"] = RAW["KEY"]
        # rel_types = [k for k in self.types if k.startswith("if")] + ["KEY"]
        # relation = relation[rel_types]
        self.data = relation.dropna(how="any")
        self.validate()
        return relation


class IncomeLoader(BaseLoader):
    """load income specific information"""

    def __init__(self):
        super(IncomeLoader, self).__init__()
        self.codes = ['QW13']
        self.vars = ['res_income']

    def run(self):
        income = RAW[self.codes].copy(True).astype(float).fillna(0)
        convert_opt_income = lambda x: np.mean(INCOME_MAPPING.get(int(x), [0, 0]))
        res_income = income.apply(convert_opt_income, axis=1).to_frame()
        res_income["KEY"] = RAW["KEY"]
        res_income.columns = ["res_income", "KEY"]

        # convert labor status
        labor_codes = [f'H_F2_B_{i}' for i in range(1, 11, 1)]
        labor = RAW[labor_codes].fillna(0).astype(int)
        labor.columns = [f'labor_{i}' for i in range(1, 11, 1)]

        labor["labor_num"] = labor.apply(lambda x: x[x == 1].size if x[x == 1].size > 0 else x[x > 1].size, axis=1)
        assert labor[labor["labor_num"] == 0].size == 0, "invalid labor number"
        res_income["res_income_percap"] = res_income["res_income"].values / labor["labor_num"].values
        res_income = res_income[res_income.res_income >= 0]

        self.data = res_income
        self.validate()
        return res_income


class ExpenditureLoader(BaseLoader):
    """load expenditure based data"""

    def __init__(self):
        super(ExpenditureLoader, self).__init__()
        self.codes = ["NEL_YEN", "NUG_YEN", "NLPG_YEN", "NKER_YEN", "NGSL_YEN", "NDSL_YEN"]
        self.vars = ["bill_elec", "bill_gas", "bill_lpg", "bill_kero", "bill_gaso", "bill_dies"]

    def run(self):
        exp = RAW[["KEY"]]
        for code, var in zip(self.codes, self.vars):
            # every code has 12 items from Jan to Dec
            keys = [f'{code}_{i}' for i in range(1, 13, 1)]
            exp.loc[RAW.index, var] = RAW[keys].astype(float).sum(axis=1)

        self.data = exp
        self.validate()
        return exp


class TransportLoader(BaseLoader):
    """load transport-related information"""

    def __init__(self):
        super().__init__()
        self.codes = [f'QS23_{i}' for i in range(1, 4, 1)]
        self.vars = [f'cartype_{i}' for i in range(1, 4, 1)]

    def run(self):
        car = RAW[self.codes]
        car.columns = self.vars
        car = car.fillna(0).astype(int)
        # create variable: car_num
        car_num = car[self.vars].apply(lambda x: x[x > 0].size, axis=1).to_frame()
        car_num.columns = ['car_num']

        # NB: 2021-02-10, add NEV
        car_num["nev_num"] = car[self.vars].apply(lambda x: 1 if (set(x).intersection({3, 4, 5})) else 0,
                                                  axis=1).values
        car_num["KEY"] = RAW["KEY"]
        car_num.columns = ["car_num", "nev_num", "KEY"]

        # create fuel consumption features
        fuel_codes = [f'QS23_B_{i}' for i in range(1, 4, 1)]
        fuel_vars = [f'fuel_{i}' for i in range(1, 4, 1)]
        fuel = RAW[fuel_codes].fillna(0).astype(float)
        fuel.columns = fuel_vars

        fuel = fuel.apply(lambda x: [np.mean(CAR_FUEL_MAPPING.get(int(i), [0, 0])) for i in x.values], axis=1)
        fuel = np.array([i for i in fuel.values])

        # make sure `car_types` have been created!
        non_fuel = car[self.vars].values
        non_fuel[non_fuel > 3] = 0  # non-fuel vehicles
        non_fuel[(non_fuel < 3) & (non_fuel > 0)] = 1

        fuel = pd.DataFrame(fuel, columns=["fuel_1", "fuel_2", "fuel_3"])
        fuel["car_avg_fuel"] = pd.DataFrame(non_fuel * fuel).apply(lambda x: x[x > 0].mean(), axis=1).fillna(0)

        # create use frequency variable for vehicles
        freq_codes = [f'QS23_C_{i}' for i in range(1, 4, 1)]
        freq_vars = [f'freq_{i}' for i in range(1, 4, 1)]
        freq = RAW[freq_codes].fillna(0).astype(int)
        freq.columns = freq_vars

        freq = freq.apply(lambda x: [np.mean(CAR_USE_MAPPING.get(int(i), [0, 0])) for i in x.values], axis=1)
        freq = np.array([i for i in freq.values])
        freq = pd.DataFrame(freq).apply(lambda x: x[x > 0].mean(), axis=1).fillna(0).to_frame()
        freq.columns = ["drive_freq"]

        # create driving distance variable
        dist_codes = [f'NQS23_D_{i}' for i in range(1, 4, 1)]
        dist_vars = [f'dist_{i}' for i in range(1, 4, 1)]
        dist = RAW[dist_codes].fillna(0).astype(float)
        dist.columns = dist_vars
        dist["dist_all"] = dist.sum(axis=1)

        # proceed with all car data
        car_num["dist_all"] = dist["dist_all"]
        car_num["drive_freq"] = freq["drive_freq"]
        car_num["car_avg_fuel"] = fuel["car_avg_fuel"]

        self.data = car_num
        self.validate()
        return car_num


class ApplianceLoader(BaseLoader):
    """load appliance related data"""

    def __init__(self):
        super().__init__()
        # Note: we suspend H_F10A~D because they're sparse one-hot encodings
        self.vars = ["tv_num", "fridge_num", "ac_num"] + ["app_%s" % i for i in range(1, 22, 1)] + \
                    ["heater_%s" % i for i in range(1, 8, 1)] + ["light_type"]
        # NQS10_A_? are the numbers of appliance the household has
        self.codes = ["NQS1", "NQS4", "NQS7"] + \
                     ["NQS10_A_%s" % i for i in range(1, 22, 1)] + \
                     ["NQW4_A_%s" % i for i in range(1, 8, 1)] + ["QS14_A_1"]

    def run(self):
        app = RAW[self.codes].fillna(0).astype(int)
        app.columns = self.vars
        # appliance statisitcs
        apps = ["app_%s" % i for i in range(1, 22, 1)] + \
               ["heater_%s" % i for i in range(1, 8, 1)] + ["light_type"]

        variety = lambda x: x[x > 0].size / x.size
        app["app_variety"] = app[apps].apply(variety, axis=1)
        intensity = lambda x: x.sum() / x.size
        app["app_intensity"] = app[apps].apply(intensity, axis=1)
        app["KEY"] = RAW["KEY"]

        # a mapping from appliance built-year to real year gap
        age_app_map = {0: [0, 0], 1: [25, 25], 2: [20, 25], 3: [15, 20], 4: [10, 15], 5: [5, 10], 6: [0, 5], 7: [0, 0]}
        # transform aging of appliances into variables
        app_age_codes = ["QS8_A_%s" % i for i in range(1, 6, 1)]
        aged = RAW[app_age_codes].fillna(0).astype(int)
        aged["avg_app_age"] = aged.apply(lambda x: np.array([np.array(age_app_map[i]).mean() for i in x]).mean(),
                                         axis=1)
        app["avg_app_age"] = aged["avg_app_age"].apply(lambda x: x if x > 0 else np.nan)

        self.data = app
        self.validate()
        return app


class PVLoader(BaseLoader):
    """load PV related data"""

    def __init__(self):
        super(PVLoader, self).__init__()
        self.codes = ["PV", "NPVG_KWH", "NPVS_KWH"]
        self.vars = ["pv", "pv_gen", "pv_sell"]

    def run(self):
        pv = RAW[["PV"]].astype(int).fillna(2)
        for code, var in zip(self.codes[1:], self.vars[1:]):
            # every code has 12 items from Jan to Dec
            keys = [f'{code}_{i}' for i in range(1, 13, 1)]
            pv[var] = RAW[keys].astype(float).sum(axis=1)

        # classify 'self-consume' and 'on-grid: 1' households
        def classifier(x):
            r1 = x[0] == 1 and x[1] > 0
            if not r1:
                return 0

            r2 = abs((x[1] / (x[2] + 0.01) - 1)) < 1  # fully on-grid
            r3 = x[2] < 1  # fully self-consume: 2

            if r2:
                return 1
            elif r3:
                return 2
            else:
                return 3  # hybrid: 3

        pv["pv_role"] = pv.apply(classifier, axis=1)
        pv["KEY"] = RAW["KEY"]
        self.data = pv
        self.validate()
        return pv


class BehaviorLoader(BaseLoader):
    """load behavioral data"""

    def __init__(self):
        self.codes = ["H_F3", "DEN_SAVING_3", "QS2_C_1", "QS8_B_1", "QS15", "QW6_3"]
        self.vars = ["daytime_at_home", "ener_saving_rate", "tv_time", "ac_time", "light_time", "heat_time"]

    def run(self):
        behavior = RAW[self.codes].fillna(0).astype(float)
        behavior.columns = self.vars
        behavior["KEY"] = RAW["KEY"]
        self.data = behavior
        self.validate()
        return behavior


class EmissionLoader(BaseLoader):
    """load emission related information"""

    def __init__(self):
        super().__init__()
        self.source_codes = ["N6_4_%s" % i for i in range(5, 8, 1)]  # N6_4_5, N6_4_6, N6_4_7
        self.usage_codes = ["N7_2_%s" % i for i in range(1, 7, 1)]

    def run(self, resident: pd.DataFrame):
        emission = RAW[self.source_codes].fillna(0).astype(float).apply(lambda x: x[x >= 0].sum(), axis=1).to_frame()
        emission.columns = ["emits"]

        usage = RAW[self.usage_codes].fillna(0).astype(float)
        or_types = ["N6_4_%s" % i for i in range(1, 5, 1)] + ["N6_4_6", "N6_4_7"]
        sub = RAW[or_types].fillna(0).astype(float)
        emission["KEY"] = RAW["KEY"]

        mask = (emission["emits"] - usage.apply(lambda x: x[x > 0].sum(), axis=1)).abs() < 0.01
        emission = pd.concat([emission[mask], sub[mask]], axis=1)
        emission = pd.concat([emission[mask], usage[mask]], axis=1)
        del usage, sub

        res_size = resident.loc[resident["KEY"].isin(emission["KEY"]), "res_size"].values
        emission["emits_per"] = emission["emits"].values / res_size
        # put in different kinds of emissions
        em_types = ["em_heating", "em_cooling", "em_water", "em_kitchen", "em_appliance", "em_car",
                    "em_elec", "em_city_gas", "em_lp_gas", "em_kerosene", "em_gasoline", "em_diesel",
                    "em_ex_car"]
        emission["em_tot_ex_car"] = (emission["emits"] - emission["N7_2_6"])

        for k, emt in zip((self.usage_codes + or_types + ["em_tot_ex_car"]), em_types):
            emission[emt] = emission[k].values / res_size
            del emission[k]

        emission.columns = ['emits', 'KEY', 'emits_per', 'em_heating', 'em_cooling', 'em_water',
                            'em_kitchen', 'em_appliance', 'em_car', 'em_elec', 'em_city_gas',
                            'em_lp_gas', 'em_kerosene', 'em_gasoline', 'em_diesel', 'em_ex_car']

        self.data = emission
        self.validate()
        return emission


class EmissionReductionLoader(BaseLoader):
    """Load information about emission reduction options
       quote the documentation in `/predict/household-emission-reduction-others.docx`
    """

    def __init__(self):
        super().__init__()
        # lighting, hot-water, heating, cooking, ... PV/NEV
        self.codes = ['QS14_A_1', 'QS17', 'QW2_2', 'H_F10C']
        self.keys = ['er_light', 'er_hot_water', 'er_heat', 'er_cook']

    def run(self):
        out = RAW[['KEY']]
        # QS14_A_1: option 3 is the ES options
        out.loc[RAW.index, 'er_light'] = RAW['QS14_A_1'].astype(int).apply(lambda x: 1 if x == 3 else 0)
        # QS17: option 1,2,6 are the ES options
        out.loc[RAW.index, 'er_hot_water'] = RAW['QS17'].astype(str).apply(lambda x: 1 if x[0] == '1' or x[1] == '1' or x[5] == '1' else 0)
        # QW2_2: option 2 are the ES options
        out.loc[RAW.index, 'er_heat'] = RAW['QW2_2'].fillna(0).astype(int).apply(lambda x: 1 if x == 2 else 0)
        # H_F10C: option 3,4,6 are the ES options
        out.loc[RAW.index, 'er_cook'] = RAW['H_F10C'].astype(str).apply(lambda x: 1 if x[1] == '1' or x[2] == '1' else 0)
        self.data = out
        self.validate()
        return out


class EnergyUsageLoader(BaseLoader):
    """Load data about energy uses by usages (GJ)"""
    def __init__(self):
        super().__init__()
        self.codes = [f'N7_1_{i}' for i in range(1, 6, 1)]
        self.keys = ['eu_heating', 'eu_cooling', 'eu_water', 'eu_kitchen', 'eu_appliance']

    def run(self):
        eu = RAW[['KEY']]
        for code, key in zip(self.codes, self.keys):
            eu[key] = RAW[[code]].fillna(0).astype(float)

        self.data = eu
        self.validate()
        return eu


def merger(*args):
    """merge all the data and do validation"""
    data = RAW[["KEY"]].astype(object)  # the init data
    for obj in args:
        assert 'KEY' in obj.columns, f'InvalidFrame: {obj}'
        obj.loc[obj.index, 'KEY'] = obj['KEY'].astype(object)
        data = data.merge(obj, on='KEY', how='inner')

    logger.warning(f'[Merger] {len(data)}/{len(RAW)} after processing.')
    des = data.describe().T
    des.to_excel(CONFIG_PATH / f"feature-desc-{get_timestamp()}.xlsx")
    data.to_excel(CONFIG_PATH / f"cache-preprocess-data-{get_timestamp()}.xlsx")
    return data


def main():
    geo = GeographyLoader().run()
    house = HouseBasicLoader().run()
    resident = ResidentLoader().run()
    relation = RelationshipLoader().run(resident)
    resident = resident.drop(labels=[f'res_{i}' for i in range(1, 11, 1)], axis=1)

    income = IncomeLoader().run()
    expense = ExpenditureLoader().run()
    transport = TransportLoader().run()

    app = ApplianceLoader().run()
    pv = PVLoader().run()
    behave = BehaviorLoader().run()
    emission = EmissionLoader().run(resident)
    em_reduce = EmissionReductionLoader().run()
    eu = EnergyUsageLoader().run()

    # produce data
    args = [geo, house, resident, relation, income, expense, transport, app,
            pv, behave, emission, em_reduce, eu]
    nkeys = ['geo', 'house', 'resident', 'relation', 'income', 'expense', 'transport',
             'app', 'pv', 'behave', 'emission', 'em_reduce', 'eu']

    out = {}
    for k, arg in zip(nkeys, args):
        keys = list(arg.columns)
        keys.remove('KEY')
        out[k] = keys

    merged = merger(*args)
    return merged, out


EMIT_VARS = ['emits',
             'emits_per',
             'em_heating',
             'em_cooling',
             'em_water',
             'em_kitchen',
             'em_appliance',
             'em_car',
             'em_elec',
             'em_city_gas',
             'em_lp_gas',
             'em_kerosene',
             'em_gasoline',
             'em_diesel',
             'em_ex_car']

STANDARD_VARS = ['build_space',
                 'build_time',
                 'room_num',
                 'double_window',
                 'heat_room_num',
                 'res_size',
                 'avg_age',
                 'res_income_percap',
                 'bill_elec',
                 'bill_gas',
                 'bill_lpg',
                 'bill_kero',
                 'bill_gaso',
                 'bill_dies',
                 'car_num',
                 'nev_num',
                 'dist_all',
                 'drive_freq',
                 'car_avg_fuel',
                 'tv_num',
                 'fridge_num',
                 'ac_num',
                 'light_type',
                 'app_variety',
                 'app_intensity',
                 'avg_app_age',
                 'daytime_at_home',
                 'ener_saving_rate',
                 'tv_time',
                 'ac_time',
                 'light_time',
                 'heat_time']


def preprocessing(data: pd.DataFrame, vars: list = STANDARD_VARS):
    """data preprocessing and standardization"""
    zscore = StandardScaler()

    scaled = data.copy(True)
    for k in tqdm(vars):
        var = scaled[k].fillna(0)
        scored = zscore.fit_transform(var.values.reshape(len(scaled), 1))
        scaled[k] = scored.reshape(len(scaled), )

    return scaled


if __name__ == '__main__':
    # test for all categories
    data, keys = main()
    scale = preprocessing(data)
