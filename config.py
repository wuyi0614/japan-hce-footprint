# The configuration script for the research: "Pattern Recognition of Household Energy Consumption Behaviors in Japan",
#
# created at 2020-11-13 (updating)
#
import numpy as np
from pathlib import Path

# ENVIRONMENT VARIABLE FOR MODEL
DEFAULT_MODEL_PATH = Path("models")

# ENVIRONMENT VARIABLE FOR COLOR PALETTE
WSJ = {
    "lightred": "#D5695D",
    "lightgreen": "#65A479",
    "lightblue": "#5D8CA8",
    "lightyellow": "#D3BA68",
    "darkred": "#B1283A",
    "darkblue": "#016392",
    "darkyellow": "#BE9C2E",
    "darkgreen": "#098154",
    "gray": "#808080"
}

# Color Schema: Continuous color
COLORS = {
    "red": "#B22222",  # brickred
    "gold": "#FFD700",
    "blue": "#4169E1",  # royalblue
    "darkblue": "#000080",  # navyblue
    "lightgreen": "#32CD32",  # limegreen
    "darkgreen": "#006400",  # darkgreen
    "pink": "#DB7093",  # PaleVioletRed
    "lightpurple": "#D8BFD8",  # Thistle
    "purple": "#9370DB",  # medium purple
    "brown": "#A0522D",  # sienna
    "grey": "#696969",
    "black": "#000000"
}

# ENVIRONMENTAL VARIABLES FOR SURVEY
SURVEY_DATES_INFORMAL = ["Oct. 2014", "Nov. 2014", "Dec. 2014", "Jan. 2015",
                         "Feb. 2015", "Mar. 2015", "Apr. 2015", "May. 2015",
                         "Jun. 2015", "Jul. 2015", "Aug. 2015", "Sep. 2015"]

SURVEY_DATES_DIGITS = ["2014-10", "2014-11", "2014-12", "2015-01", "2015-02",
                       "2015-03", "2015-04", "2015-05", "2015-06", "2015-07",
                       "2015-08", "2015-09", "2015-10"]

QS2_2_MAPPING = {
    1: [0, 9],
    2: [10, 19],
    3: [20, 29],
    4: [30, 39],
    5: [40, 49],
    6: [50, 59],
    7: [60, 64],
    8: [65, 74],
    9: [75, 75]  # open infinite
}

PREFECTURE_MAPPING = {
    1: "北海道",
    2: "青森県",
    3: "岩手県",
    4: "宮城県",
    5: "秋田県",
    6: "山形県",
    7: "福島県",
    8: "茨城県",
    9: "栃木県",
    10: "群馬県",
    11: "埼玉県",
    12: "千葉県",
    13: "東京都",
    14: "神奈川県",
    15: "新潟県",
    16: "富山県",
    17: "石川県",
    18: "福井県",
    19: "山梨県",
    20: "長野県",
    21: "岐阜県",
    22: "静岡県",
    23: "愛知県",
    24: "三重県",
    25: "滋賀県",
    26: "京都府",
    27: "大阪府",
    28: "兵庫県",
    29: "奈良県",
    30: "和歌山県",
    31: "鳥取県",
    32: "島根県",
    33: "岡山県",
    34: "広島県",
    35: "山口県",
    36: "徳島県",
    37: "香川県",
    38: "愛媛県",
    39: "高知県",
    40: "福岡県",
    41: "佐賀県",
    42: "長崎県",
    43: "熊本県",
    44: "大分県",
    45: "宮崎県",
    46: "鹿児島県",
    47: "沖縄県"
}

PREFECTURE_NAME_MAPPING = {
    '北海道': 'Hokkaido',
    '青森県': 'Aomori',
    '岩手県': 'Iwate',
    '宮城県': 'Miyagi',
    '秋田県': 'Akita',
    '山形県': 'Yamagata',
    '福島県': 'Fukushima',
    '茨城県': 'Ibaraki',
    '栃木県': 'Tochigi',
    '群馬県': 'Gumma',
    '埼玉県': 'Saitama',
    '千葉県': 'Chiba',
    '東京都': 'Tokyo-to',
    '神奈川県': 'Kanagawa',
    '新潟県': 'Niigata',
    '富山県': 'Toyama',
    '石川県': 'Ishikawa',
    '福井県': 'Fukui',
    '山梨県': 'Yamanashi',
    '長野県': 'Nagano',
    '岐阜県': 'Gifu',
    '静岡県': 'Shizuoka',
    '愛知県': 'Aichi',
    '三重県': 'Mie',
    '滋賀県': 'Shiga',
    '京都府': 'Kyoto-fu',
    '大阪府': 'Osaka-fu',
    '兵庫県': 'Hyogo',
    '奈良県': 'Nara',
    '和歌山県': 'Wakayama',
    '鳥取県': 'Tottori',
    '島根県': 'Shimane',
    '岡山県': 'Okayama',
    '広島県': 'Hiroshima',
    '山口県': 'Yamaguchi',
    '徳島県': 'Tokushima',
    '香川県': 'Kagawa',
    '愛媛県': 'Ehime',
    '高知県': 'Kochi',
    '福岡県': 'Fukuoka',
    '佐賀県': 'Saga',
    '長崎県': 'Nagasaki',
    '熊本県': 'Kumamoto',
    '大分県': 'Oita',
    '宮崎県': 'Miyazaki',
    '鹿児島県': 'Kagoshima',
    '沖縄県': 'Okinawa'
}

CLASS_CLASS_MAPPING = {
    1: "City Level 1",
    2: "City Level 2",
    3: "City Level 3"
}

HOUSEHOLD_NUM_Y2018 = 50991000

INCOME_MAPPING = {
    1: [0, 250],
    2: [250, 500],
    3: [500, 750],
    4: [750, 1000],
    5: [1000, 1500],
    6: [1500, 2000],
    7: [2000, 2000],
    8: [0, 0],  # don't want to answer
    9: [0, 0]  # unknown
}

CAR_EMISSION = {
    1: [0, 660],
    2: [661, 1000],
    3: [1001, 1500],
    4: [1501, 2000],
    5: [2001, 3000],
    6: [3001, 4000],
    7: [4001, 4001],
    8: [0, 0]
}

CAR_FUEL_MAPPING = {
    1: [0, 5],
    2: [5, 10],
    3: [10, 15],
    4: [15, 15],
    5: [0, 0],
    6: [0, 0]
}

CAR_USE_MAPPING = {
    1: [7, 7],
    2: [5, 6],
    3: [3, 4],
    4: [1, 2],
    5: [0, 1],
    6: [0, 0]
}

VAR_LABELS = {
    "prefecture": "Prefecture",
    "city_area": "City area",
    "city_class": "City class",

    'grid_type': 'Grid company',

    "build_time": "Building period",
    "build_space": "Building space",
    "build_type": "Building type",
    "room_num": "Number of rooms",
    "double_window": "Has double window",

    "res_income": "Household gross income",
    "res_income_percap": "Income per capita",

    "avg_age": "Average age",
    "res_size": "Family size",
    "elderly_num": "Number of the elderly",
    "child_num": "Number of children",
    "labor_num": "Number of the employed",

    "if_single": "The single",
    "if_couple": "The couple",
    "if_single_kids": "The single with kids",
    "if_couple_kids": "The couple with kids",
    "if_single_elderly": "The single elderly",
    "if_couple_elderly": "The couple elderly",
    "if_big_family": "The big family",
    "if_others": "The other families",

    "heater_1": "AC heating",
    "heater_2": "Electric stoves",
    "heater_3": "Electric blanket",
    "heater_4": "Heat-storage heater",
    "heater_5": "Gas stoves",
    "heater_6": "Kerosene heater",
    "heater_7": "Wood-fuel stove",
    "heater_8": "Other heating",
    "heat_time": "Service time of heaters",
    "heat_room_num": "Number of rooms with heaters",
    "heater_variety": "Heater variety",
    "heater_intensity": "Heater density",

    "hotwater_1": "Heat-pump water heater",
    "hotwater_2": "Electric water heater",
    "hotwater_3": "Gas water heater",
    "hotwater_4": "Small gas instant water heater",
    "hotwater_5": "Kerosene water heater",
    "hotwater_6": "Solar water heater",
    "hotwater_7": "Gas-engine water heater",
    "hotwater_8": "Household fuel cell",
    "hotwater_9": "Other water heater",
    "hotwater_time": "Service time of hotwater",

    "app_1": "Washing machine no drying",
    "app_2": "Washing machine with drying",
    "app_3": "Electric clothes dryer",
    "app_4": "Gas clothes dryer",
    "app_5": "Bathroom dryer",
    "app_6": "Dishwasher",
    "app_7": "Dish dryer",
    "app_8": "Microwave oven",
    "app_9": "Gas grill",
    "app_10": "Rice cooker",
    "app_11": "Gas stove",
    "app_12": "Electric kettle",
    "app_13": "Water dispenser",
    "app_14": "Heating toilet seat with washing",
    "app_15": "Heating toilet seat no washing",
    "app_16": "Humidifier",
    "app_17": "Dehumidifier",
    "app_18": "Air cleaner",
    "app_19": "Computer",
    "app_20": "DVD",
    "app_21": "Internet modem & router",
    'app_variety': "Appliance variety",
    'app_intensity': "Appliance density",
    "avg_app_age": "Average age of appliances",

    "daytime_at_home": "Daytime at home",
    "ener_saving_rate": "Energy saving rate",

    "ac_num": "Number of air conditioners",
    "fridge_num": "Number of fridges",
    "tv_time": "Service time of TVs",
    "tv_num": "Number of TVs",
    "ac_time": "Service time of air conditioners",
    "light_type": "Type of lighting",
    "light_time": "Service time of lighting",

    "car_num": "Number of vehicles",
    "dist_all": "Total driving distance",
    "drive_freq": "Weekly driving frequency",
    "nev_num": "Number of NEV",
    "ele_moto_num": "Number of electric motors",
    "car_avg_fuel": "Average fuel use",
    "moto_num": "Number of motors"
}


# future energy intensity has its length to be 3 aligned with low, medium, and high scenarios.
SCENARIO_YEAR_GAP = 6  # 2025 - 2050
FUTURE_ENERGY_INTENSITY = {
    'LED': {
        'em_appliance': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .6,
        'em_water': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .6,
        'em_heating': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .6,
        'em_cooling': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .45,
        'em_kitchen': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .65
    },
    'JG-2040': {
        'em_appliance': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .35,
        'em_water': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .37,
        'em_heating': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .25,
        'em_cooling': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .30,
        'em_kitchen': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .10
    },
    'JG-2050': {
        'em_appliance': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .2333,
        'em_water': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .2467,
        'em_heating': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .1667,
        'em_cooling': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .20,
        'em_kitchen': np.arange(1, SCENARIO_YEAR_GAP + 1) / SCENARIO_YEAR_GAP * .067
    }
}

# future penetration rate settings has its length to be 5 aligned with 2018, 2025, 2030, 2035, 2040
FUTURE_LINEAR_PENETRATION_RATE = {
    'pv': {
        2018: .096,
        2025: .118,
        2030: .14,
        2035: .1735,
        2040: .2085
    },  # baseline=9.6%, 2030=14%, 2050=28%
    'nev': {
        2018: .01,
        2025: 0.1667,
        2030: .3234,
        2035: .3919,
        2040: .4604}  # baseline=1%, 2020=6.7%, 2030=32.34%, 2050=59.74%
}

FUTURE_AMBITIOUS_PENETRATION_RATE = {
    'pv': {
        2018: .096,
        2025: .118,
        2030: .14,
        2035: .21,
        2040: .28
    },  # baseline=9.6%, 2030=14%, 2050=28%
    'nev': {
        2018: .01,
        2025: 0.1667,
        2030: .3234,
        2035: .4604,
        2040: .5974}  # baseline=1%, 2020=6.7%, 2030=32.34%, 2050=59.74%
}


# unit: tCO2/GJ
FUTURE_GRID_EMISSION_FACTOR = {
    2018: 0.139475068,
    2025: 0.106065558,
    2030: 0.090407879,
    2035: 0.07104756,
    2040: 0.045800912
}

""" VARIABLES OF INTERESTS
--------- General description
Integers represent the column numbers of data sheet, i.e. `74 - QR2_A11`.
`QR2_A11` indicates the column name in the survey data.

--------- Calculated emissions



--------- Geometric related
KEY: unique ids for households

AREA (PREFECTURE:2017): 1 - 47 cities
AREA2 (AREA_FLG:2017): 1 - 10 大区区分
CITY (CITY_CLASS:2017): 1 - big, 2 - medium, 3 - small
CITY (2017): 定位到城市-区

QS1 (H_F1:2017): [家庭]Q1居住人数(NA), QS2_1_1 ~ QS2_1_10: indicate relationships between the owner and residents
1	世帯主
2	配偶者
3	子（子の配偶者を含む）
4	親
5	祖父母
6	孫
7	他の親族（兄弟姉妹等）
8	親族以外の人

QS2_2 (H_F2_A:2017): [家庭]Q2居住者-年龄(SA), QS2_2_1 ~ QS2_2_10: indicate ages for each resident
1	0～9歳
2	10～19歳
3	20～29歳
4	30～39歳
5	40～49歳
6	50～59歳
7	60～64歳
8	65～74歳
9	75歳以上

QS2_3 (H_F2_B:2017): [家庭]q2_3居住者-职业有无(SA), QS2_3_1 ~ QS2_3_10: indicate occupation of each resident
1	あり	有
2	なし	没有
3	不明	不明

QS4 (QW13:2017): [家庭]Q4家庭年收入(SA)
1	250万円未満	不到250万日元
2	250～500万円未満	250万~ 500万日元以下
3	500～750万円未満	500 ~ 750万日元以下
4	750～1000万円未満	750万~ 1000万日元以下
5	1000～1500万円未満	1000万~ 1500万日元以下
6	1500～2000万円未満	1500万~ 2000万日元以下
7	2000万円以上	2000万日元以上
8	わからない	不明白
9	答えたくない	不想回答 (2014 only)

QS5: [家庭]Q5住宅的建造方法(SA)
1	[戸建]
2	[集合]

QS6: [世帯]Q6 住まいの建築時期(SA) * need conversion into years long
1	1970（昭和45）年以前
2	1971～1980（昭和46～55）年
3	1981～1985（昭和56～60）年
4	1986～1990（昭和61～平成2）年
5	1991～1995（平成3～7）年
6	1996～2000（平成8～12）年
7	2001～2005（平成13～17）年
8	2006～2010（平成18～22）年
9	2011（平成23）年以降
10	わからない
11	不明

QS27: [家庭]Q27车辆-使用台数(NA)  X
    QS27_1: 汽车
    QS27_2: 使用汽油的摩托车和踏板车(包括原动机自行车)
    QS27_3: 电动摩托车、踏板车(电动助力自行车除外)

QS28 (QS23:2017): [家庭]Q28汽车-燃料(SA)  X
    QS28_1: 第一台
    1	汽油(包括混合动力车)
    2	柴油(柴油)
    3	电
    4	汽油和电(插电混合动力车)
    5	其他(LPG车、CNG车等)
    6	不明
    QS28_2: 第二台
    QS28_3: 第三台
    
QS28_A (QS23_A:2017): [家庭]q28_a汽车-排气量(SA)
    QS28_A_1: 第一台
    1	660cc以下(轻型车)
    2	661 ~ 1000cc
    3	1001 ~ 1500cc
    4	1501 ~ 2000cc
    5	2001 ~ 3000cc
    6	3001 ~ 4000cc
    7	4001cc以上
    8	不明
    QS28_A_2: 第二台
    QS28_A_3: 第三台


QS28_B (QS23_B:2017): [家庭]q28_b汽车-油耗(SA)
    QS28_B_1: 第一台
    1	每升不足五公里
    2	每升5 ~ 10公里
    3	每升10 ~ 15公里
    4	每升15公里以上
    5	不明白
    6	不明
    QS28_B_2: 第二台
    QS28_B_3: 第三台


QS28_C (QS23_C:2017): [家庭]q28_c汽车-使用频率(SA)
    QS28_C_1: 第一台
    1	每天
    2	每周5 ~ 6天
    3	每周3 ~ 4天
    4	每周1 ~ 2天
    5	一周不到一天
    6	不明
    QS28_C_2: 第二台
    QS28_C_3: 第三台


QS28_D (NQS23_D:2017): [家庭]q28_d汽车-年行驶距离(km)(NA)
    QS28_D_1: 第一台
    QS28_D_2: 第二台
    QS28_D_3: 第三辆

--------- Energy related
9 - F1A: [10月]F1A使用设备-供暖设备(MA)
10 - F1B: [10月]F1B使用机器-供热水机器(MA)
13 - F1E: [10月]F1E使用设备-车辆(MA)
14 - QR1_A_1: [例月]q1_a_1电使用量(NA), QR1_A_1_1 ~ QR1_A_1_12: 14年10月 ~ 15年9月
xx - QR1_A_2: [例月]q1_a_2电力使用金额(NA), QR1_A_2_1 ~ QR1_A_2_12: 14年10月 ~ 15年9月

74 - QR2_A11: [例月]q2_a_1太阳光发电电量(NA), 1 - 14年10月 ~ 12 - 15年9月
75 - QR2_A12: [例月]q2_a_1太阳光发电卖电力量(NA), same as above
76 - QR2_A22: [例月]q2_a_2太阳光发电卖电接受金额(NA), same as above
xx - QR3_A_1: [例月]q3_a_1煤气使用量(NA), same as above
xx - QR4_A11: [例月]q4_a_1购买量-煤油(NA), same as above
xx - QR4_A12: [例月]q4_a_1购买量-汽油(NA), same as above
xx - QR4_A13: [例月]q4_a_1购买量-柴油(NA)

xx - QS20: [家庭]Q20照明-种类(MA)

xx - QS18: [家庭]Q18各家电-使用台数(NA), the following are the subsets of this variable:
    QS18_1	洗衣机(无干燥功能)
    QS18_2	洗衣机(附带干燥功能)
    QS18_3	衣服烘干机(电)
    QS18_4	衣服烘干机(煤气)
    QS18_5	浴室干燥机
    QS18_6	洗碗机
    QS18_7	餐具烘干机
    QS18_8	微波炉(附带烤箱功能)
    QS18_9	微波炉(无烤箱功能)
    QS18_10	煤气烤箱
    QS18_11	电饭锅
    QS18_12	煤气电饭锅
    QS18_13	电热水壶
    QS18_14	温水洗净坐便器
    QS18_15	暖气座便器(没有温水洗净功能)
    QS18_16	加湿器
    QS18_17	除湿机
    QS18_18	空气净化器
    QS18_19	个人电脑
    QS18_20	蓝光录像机或播放器
    QS18_21	因特网调制解调器、路由器

---------- Events
* Note: from `QR5B_2` to `QR5B_12`, means 14年10月 ~ 15年9月, including the following options:
1	世帯人数が変化した	家庭人数发生了变化
2	転居した	迁居了
3	住宅を増築した	增建了住宅
4	住宅を建て替えた	改建了住宅
5	太陽光発電を導入した	引进了太阳能发电  *
6	ガスエンジン発電・給湯器（エコウィル）を導入した	引进了燃气发动机发电和热水器  *
7	燃料電池（エネファーム）を導入した	引进了燃料电池  *
8	旅行等で5日間以上、世帯員全員が不在の日があった	因为旅行等5天以上，全体家庭成员都有不在的日子
9	その他	其他
10	とくになし	没什么特别的

---------- Sustainability
xx - QS15: [家庭]Q15冰箱-节能行动(SA)
xx - QS15_1: 冰箱的温度设定在夏天是“中”以下，其他季节是“弱”
1	実施している	正在实施
2	実施していない	没有实施
3	不明	不明

xx - QS15_2: 不要把东西塞进冰箱里
1	実施している	正在实施
2	実施していない	没有实施
3	不明	不明

xx - QS17_2: [家庭]q17_2空调-设定温度(SA)
xx - QS17_3: [家庭]q17_3空调-平日使用时间(8月左右)(SA)
1	2時間未満	不到两个小时
2	2時間～4時間未満	2小时~ 4小时以内
3	4時間～8時間未満	4小时~ 8小时以内
4	8時間～12時間未満	8小时~不到12小时
5	12時間～16時間未満	12小时~ 16小时以内
6	16時間～20時間未満	16小时~ 20小时以内
7	20時間～24時間未満	20小时~ 24小时以内
8	24時間（一日中）	24小时(一整天)
9	不明	不明

xx - QS21: [家庭]Q21照明-节能行动(SA), the following variables are the subsets of QS21
QS21_1	根据状况调整照明的亮度(包括使用减灯和自动调光功能)
1	実施している	正在实施
2	実施していない	没有实施
3	調整できない	不能调整
4	不明	不明

QS21_2  即使是短时间离开的时候也要注意关灯
1	実施している	正在实施
2	実施していない	没有实施
3	不明	不明

"""
