# -*- encoding:utf-8 -*-
from __future__ import print_function

import numpy as np
import seaborn as sns
import warnings

import abupy
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd, ABuMarketDrawing
from abupy import EMarketSourceType
from abupy import EMarketDataFetchMode
from abupy import AbuMetricsBase
from abupy import abu
from abupy.CoreBu import ABuEnv
from abupy.CoreBu.ABuStore import store_abu_result_out_put
from abupy.TradeBu.ABuCommission import calc_commission_cn

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})

# 开盘下跌超过限值后，当天放弃买入
abupy.slippage.sbm.g_open_down_rate = 0.04

# 设置选股因子，None为不使用选股因子
stock_pickers = None
buy_factors = [{'past_factor': 4, 'up_deg_threshold': 3, 'poly': 2, 'nb_dev': 2,
                'time_period': 20, 'atr_off': True, 'class': abupy.MyBuyBollBreak,
                'slippage': abupy.AbuSlippageBuyMean,
                'position': {'class': abupy.AbuKellyPosition, 'win_rate': 0.6,
                'gains_mean': 0.19, 'losses_mean': -0.04}},
               {'position': {'class': abupy.AbuKellyPosition}, 'slippage': abupy.AbuSlippageBuyMean, 'class': abupy.FuBuyAppendTrade}
               ]

sell_factors = [
    {'nb_dev': 2, 'time_period': 20,
     'class': abupy.MyBollSellBreak}]

# 手续费计算
commission_dict = {'buy_commission_func': calc_commission_cn}

# 市场，缓存，数据源配置
ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL
ABuEnv.g_market_target = abupy.EMarketTargetType.E_MARKET_TARGET_CN
ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx
# ABuEnv.g_data_cache_type = abupy.EDataCacheType.E_DATA_CACHE_HDF5
ABuEnv.g_data_cache_type = abupy.EDataCacheType.E_DATA_CACHE_CSV


''''经单独回测，招商银行600036, 联通600050, 双汇发展000895[负收益], 南方航空600029, 东方证券600958, 大唐发电601991[负收益],金隅集团601992[负收益]不适用这个策略'''


'''普通模式，跟机器学习一点关系都没'''


def sample_a21():
    # 设置初始资金数
    read_cash = 500000000

    # 择时股票池
    # choice_symbols = ['603993', '601998', '601992', '601991', '601989', '601988', '601985', '601939', '601933', '601919', '601901', '601899', '601898', '601881', '601877', '601857', '601828', '601818', '601808', '601800', '601788', '601766', '601727', '601688', '601669', '601668', '601633', '601628', '601618', '601607', '601601', '601600', '601398', '601390', '601360', '601328', '601288', '601238', '601229', '601225', '601211', '601186', '601169', '601166', '601155', '601111', '601108', '601088', '601018', '601012', '601009', '601006', '600999', '600958', '600919', '600900', '600893', '600887', '600837', '600816', '600795', '600703', '600690', '600688', '600663', '600660', '600606', '600600', '600588', '600585', '600518', '600487', '600406', '600398', '600383', '600362', '600346', '600340', '600309', '600297', '600221', '600196', '600188', '600176', '600115', '600111', '600104', '600061', '600050', '600048', '600036', '600031', '600030', '600029', '600028', '600025', '600023', '600019', '600018', '600016', '600015', '600011', '600010', '600000', '300498', '300433', '300124', '300072', '300070', '300059', '300015', '300003', '002739', '002736', '002714', '002600', '002558', '002493', '002475', '002456', '002450', '002415', '002310', '002252', '002241', '002236', '002202', '002142', '002120', '002044', '002027', '002024', '002010', '001979', '001965', '000895', '000776', '000725', '000617', '000166', '000069', '000063', '000039', '000002', '000001']
    choice_symbols = ['601398']
    # choice_symbols = ['600999']
    # choice_symbols = ['601398', '601988', '601939', '603993', '600999', '300059', '600900', '601328', '601288', '600887', '600029', '000002']

    abu_result_tuple, _ = abu.run_loop_back(read_cash,
                                            buy_factors, sell_factors, stock_pickers, choice_symbols=choice_symbols,
                                            n_folds=6, commission_dict=commission_dict)
    print(abu_result_tuple.orders_pd)

    metrics = AbuMetricsBase(*abu_result_tuple)
    metrics.fit_metrics()

    # plot_simple = abu_result_tuple.orders_pd[abu_result_tuple.orders_pd.profit_cg < 0]
    # save=True保存在本地，耗时操作，需要运行几分钟
    # ABuMarketDrawing.plot_candle_from_order(plot_simple, save=False)
    # 筛出有交易结果的
    orders_pd_atr = abu_result_tuple.orders_pd[abu_result_tuple.orders_pd.result != 0]
    orders_pd_atr.filter(['buy_cnt', 'buy_pos', 'buy_price', 'profit', 'result'])
    metrics.plot_returns_cmp(only_info=True)
    # 保存交易结果
    store_abu_result_out_put(abu_result_tuple)
    # metrics.plot_buy_factors()
    # metrics.plot_sell_factors()
    # metrics.plot_effect_mean_day()
    # plt.show()
    # metrics.plot_keep_days()
    # plt.show()
    metrics.plot_max_draw_down()
    # plt.show()


'''开始ump主裁识别拦截模式'''


def sample_a22():
    # 设置初始资金数
    read_cash = 5000000

    abupy.env.g_enable_ump_main_deg_block = True
    abupy.env.g_enable_ump_main_jump_block = True
    abupy.env.g_enable_ump_main_price_block = True
    abupy.env.g_enable_ump_main_wave_block = True

    # 择时股票池
    # choice_symbols = ['603993', '601998', '601992', '601991', '601989', '601988', '601985', '601939', '601933', '601919', '601901', '601899', '601898', '601881', '601877', '601857', '601828', '601818', '601808', '601800', '601788', '601766', '601727', '601688', '601669', '601668', '601633', '601628', '601618', '601607', '601601', '601600', '601398', '601390', '601360', '601328', '601288', '601238', '601229', '601225', '601211', '601186', '601169', '601166', '601155', '601111', '601108', '601088', '601018', '601012', '601009', '601006', '600999', '600958', '600919', '600900', '600893', '600887', '600837', '600816', '600795', '600703', '600690', '600688', '600663', '600660', '600606', '600600', '600588', '600585', '600518', '600487', '600406', '600398', '600383', '600362', '600346', '600340', '600309', '600297', '600221', '600196', '600188', '600176', '600115', '600111', '600104', '600061', '600050', '600048', '600036', '600031', '600030', '600029', '600028', '600025', '600023', '600019', '600018', '600016', '600015', '600011', '600010', '600000', '300498', '300433', '300124', '300072', '300070', '300059', '300015', '300003', '002739', '002736', '002714', '002600', '002558', '002493', '002475', '002456', '002450', '002415', '002310', '002252', '002241', '002236', '002202', '002142', '002120', '002044', '002027', '002024', '002010', '001979', '001965', '000895', '000776', '000725', '000617', '000166', '000069', '000063', '000039', '000002', '000001']
    # choice_symbols = ['601398']
    # choice_symbols = ['600036']
    choice_symbols = None
    # choice_symbols = ['601398', '601988', '601939', '601328', '601288', '600887', '600029', '000002']
    #choice_symbols = ['601398', '601988', '601939', '603993', '600999', '300059', '600900', '601328', '601288',
    #                  '600887', '600029', '000002', '600196', '002024', '002241', '600050', '601989', '601992', '601901']
    # choice_symbols = ['601398', '601988', '601939', '603993', '600196', '600660', '600703', '600887', '600999', '300059', '600900', '601328', '601288', '600887', '600029', '000002']

    abu_result_tuple, _ = abu.run_loop_back(read_cash,
                                            buy_factors, sell_factors, stock_pickers, choice_symbols=choice_symbols,
                                            n_folds=1, commission_dict=commission_dict)

    # AbuMetricsBase.show_general(*abu_result_tuple, only_show_returns=True)

    metrics = AbuMetricsBase(*abu_result_tuple)
    metrics.fit_metrics()
    # 筛出有交易结果的
    orders_pd_atr = abu_result_tuple.orders_pd[abu_result_tuple.orders_pd.result != 0]
    orders_pd_atr.filter(['buy_cnt', 'buy_pos', 'buy_price', 'profit', 'result'])

    # 保存交易结果
    store_abu_result_out_put(abu_result_tuple)
    metrics.plot_returns_cmp(only_info=True)
    # metrics.plot_buy_factors()
    # metrics.plot_sell_factors()
    metrics.plot_effect_mean_day()
    # plt.show()
    metrics.plot_keep_days()
    # plt.show()
    metrics.plot_max_draw_down()
    # plt.show()


'''机器学习训练数据生成'''


def sample_a23():
    # 设置初始资金数
    read_cash = 50000000

    # abupy.env.g_enable_ump_main_deg_block = True
    # abupy.env.g_enable_ump_main_jump_block = True
    # abupy.env.g_enable_ump_main_price_block = True
    # abupy.env.g_enable_ump_main_wave_block = True

    ###########################################################################################
    # 回测生成买入时刻特征
    abupy.env.g_enable_ml_feature = True
    # 回测将symbols切割分为训练集数据和测试集数据
    abupy.env.g_enable_train_test_split = True
    # 下面设置回测时切割训练集，测试集使用的切割比例参数，默认为10，即切割为10份，9份做为训练，1份做为测试，
    # 由于美股股票数量多，所以切割分为4份，3份做为训练集，1份做为测试集
    abupy.env.g_split_tt_n_folds = 10

    ###########################################################################################

    # 择时股票池
    # choice_symbols = ['603993', '601998', '601992', '601991', '601989', '601988', '601985', '601939', '601933', '601919', '601901', '601899', '601898', '601881', '601877', '601857', '601828', '601818', '601808', '601800', '601788', '601766', '601727', '601688', '601669', '601668', '601633', '601628', '601618', '601607', '601601', '601600', '601398', '601390', '601360', '601328', '601288', '601238', '601229', '601225', '601211', '601186', '601169', '601166', '601155', '601111', '601108', '601088', '601018', '601012', '601009', '601006', '600999', '600958', '600919', '600900', '600893', '600887', '600837', '600816', '600795', '600703', '600690', '600688', '600663', '600660', '600606', '600600', '600588', '600585', '600518', '600487', '600406', '600398', '600383', '600362', '600346', '600340', '600309', '600297', '600221', '600196', '600188', '600176', '600115', '600111', '600104', '600061', '600050', '600048', '600036', '600031', '600030', '600029', '600028', '600025', '600023', '600019', '600018', '600016', '600015', '600011', '600010', '600000', '300498', '300433', '300124', '300072', '300070', '300059', '300015', '300003', '002739', '002736', '002714', '002600', '002558', '002493', '002475', '002456', '002450', '002415', '002310', '002252', '002241', '002236', '002202', '002142', '002120', '002044', '002027', '002024', '002010', '001979', '001965', '000895', '000776', '000725', '000617', '000166', '000069', '000063', '000039', '000002', '000001']
    # choice_symbols = ['601398']
    # choice_symbols = ['600036']
    # choice_symbols = ['601398', '601988', '601939', '601328', '601288', '600887', '600029', '000002']
    choice_symbols = ['601398', '601988', '601939', '603993', '600999', '300059', '600900', '601328', '601288',
                      '600887', '600029', '000002', '002024', '002241', '600050', '601989', '601992', '601901']

    abu_result_tuple, _ = abu.run_loop_back(read_cash,
                                            buy_factors, sell_factors, stock_pickers, choice_symbols=choice_symbols,
                                            n_folds=6, start='2012-04-20', end='2018-04-26', commission_dict=commission_dict)

    # 把运行的结果保存在本地，以便之后分析回测使用，保存回测结果数据代码如下所示
    abu.store_abu_result_tuple(abu_result_tuple, n_folds=6, store_type=abupy.EStoreAbu.E_STORE_CUSTOM_NAME,
                               custom_name='18_6_top_train_cn')

    abu.store_abu_result_tuple(abu_result_tuple, n_folds=6, store_type=abupy.EStoreAbu.E_STORE_CUSTOM_NAME,
                               custom_name='18_6_top_test_cn')

    AbuMetricsBase.show_general(*abu_result_tuple, only_show_returns=True)

    metrics = AbuMetricsBase(*abu_result_tuple)
    metrics.fit_metrics()
    # 筛出有交易结果的
    orders_pd_atr = abu_result_tuple.orders_pd[abu_result_tuple.orders_pd.result != 0]
    orders_pd_atr.filter(['buy_cnt', 'buy_pos', 'buy_price', 'profit', 'result'])
    metrics.plot_returns_cmp(only_info=True)
    # metrics.plot_buy_factors()
    # metrics.plot_sell_factors()
    metrics.plot_effect_mean_day()
    # plt.show()
    metrics.plot_keep_days()
    # plt.show()
    metrics.plot_max_draw_down()
    # plt.show()

if __name__ == "__main__":
    # sample_a21()
    sample_a22()
    # sample_a23()
