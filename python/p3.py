# -*- encoding:utf-8 -*-
from __future__ import print_function

import ast

import numpy as np
import seaborn as sns
import warnings

from sklearn.metrics import accuracy_score

import abupy
from sklearn import metrics
from abupy import ABuSymbolPd, metrics, AbuUmpMainJump, AbuUmpMainPrice, AbuUmpMainWave
from abupy import EMarketSourceType
from abupy import EMarketDataFetchMode
from abupy import AbuMetricsBase
from abupy import AbuUmpMainDeg
from abupy import abu
from abupy import ml
from abupy.CoreBu import ABuEnv
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
ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_bd
# ABuEnv.g_data_cache_type = abupy.EDataCacheType.E_DATA_CACHE_HDF5
ABuEnv.g_data_cache_type = abupy.EDataCacheType.E_DATA_CACHE_CSV


def sample_a21():
    # 设置初始资金数
    read_cash = 50000

    ###########################################################################################
    # 回测生成买入时刻特征
    abupy.env.g_enable_ml_feature = True
    # 回测将symbols切割分为训练集数据和测试集数据
    abupy.env.g_enable_train_test_split = True
    # 下面设置回测时切割训练集，测试集使用的切割比例参数，默认为10，即切割为10份，9份做为训练，1份做为测试，
    # 由于美股股票数量多，所以切割分为4份，3份做为训练集，1份做为测试集
    abupy.env.g_split_tt_n_folds = 4

    ###########################################################################################

    # 择时股票池
    # choice_symbols = ['603993', '601998', '601992', '601991', '601989', '601988', '601985', '601939', '601933', '601919', '601901', '601899', '601898', '601881', '601877', '601857', '601828', '601818', '601808', '601800', '601788', '601766', '601727', '601688', '601669', '601668', '601633', '601628', '601618', '601607', '601601', '601600', '601398', '601390', '601360', '601328', '601288', '601238', '601229', '601225', '601211', '601186', '601169', '601166', '601155', '601111', '601108', '601088', '601018', '601012', '601009', '601006', '600999', '600958', '600919', '600900', '600893', '600887', '600837', '600816', '600795', '600703', '600690', '600688', '600663', '600660', '600606', '600600', '600588', '600585', '600518', '600487', '600406', '600398', '600383', '600362', '600346', '600340', '600309', '600297', '600221', '600196', '600188', '600176', '600115', '600111', '600104', '600061', '600050', '600048', '600036', '600031', '600030', '600029', '600028', '600025', '600023', '600019', '600018', '600016', '600015', '600011', '600010', '600000', '300498', '300433', '300124', '300072', '300070', '300059', '300015', '300003', '002739', '002736', '002714', '002600', '002558', '002493', '002475', '002456', '002450', '002415', '002310', '002252', '002241', '002236', '002202', '002142', '002120', '002044', '002027', '002024', '002010', '001979', '001965', '000895', '000776', '000725', '000617', '000166', '000069', '000063', '000039', '000002', '000001']
    # choice_symbols = ['601398']
    choice_symbols = ['601398', '601988', '601939', '601328', '601288', '600887', '600029', '000002']

    abu_result_tuple, _ = abu.run_loop_back(read_cash,
                                            buy_factors, sell_factors, stock_pickers, choice_symbols=choice_symbols,
                                            n_folds=4, commission_dict=commission_dict)

    # 把运行的结果保存在本地，以便之后分析回测使用，保存回测结果数据代码如下所示
    abu.store_abu_result_tuple(abu_result_tuple, n_folds=4, store_type=abupy.EStoreAbu.E_STORE_CUSTOM_NAME,
                               custom_name='tt_train_cn')

    abu.store_abu_result_tuple(abu_result_tuple, n_folds=4, store_type=abupy.EStoreAbu.E_STORE_CUSTOM_NAME,
                               custom_name='tt_test_cn')

    # AbuMetricsBase.show_general(*abu_result_tuple, only_show_returns=True)

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


def sample_ump_train():
    abu_result_tuple = \
        abu.load_abu_result_tuple(n_folds=4, store_type=abupy.EStoreAbu.E_STORE_CUSTOM_NAME,
                                  custom_name='tt_train_cn')
    # 需要在有缓存的情况下运行
    abupy.env.g_enable_train_test_split = False
    # 使用切割好的测试数据
    abupy.env.g_enable_last_split_test = True

    from abupy.UmpBu.ABuUmpMainMul import UmpMulFiter
    mul = UmpMulFiter(orders_pd=abu_result_tuple.orders_pd, scaler=False)
    print('mul.df.head():\n', mul.df.head())

    # 默认使用svm作为分类器
    print('decision_tree_classifier cv please wait...')
    mul.estimator.decision_tree_classifier()
    mul.cross_val_accuracy_score()

    # 默认使用svm作为分类器
    print('knn_classifier cv please wait...')
    # 默认使用svm作为分类器, 改分类器knn
    mul.estimator.knn_classifier()
    mul.cross_val_accuracy_score()

    from abupy.UmpBu.ABuUmpMainDeg import UmpDegFiter
    deg = UmpDegFiter(orders_pd=abu_result_tuple.orders_pd)
    print('deg.df.head():\n', deg.df.head())

    print('xgb_classifier cv please wait...')
    # 分类器使用GradientBoosting
    deg.estimator.xgb_classifier()
    deg.cross_val_accuracy_score()

    print('adaboost_classifier cv please wait...')
    # 分类器使用adaboost
    deg.estimator.adaboost_classifier(base_estimator=None)
    deg.cross_val_accuracy_score()

    print('train_test_split_xy please wait...')
    deg.train_test_split_xy()


'''加载机器学习训练数据'''


def load_abu_result_tuple():
    abu_result_tuple_train = abu.load_abu_result_tuple(n_folds=7, store_type=abupy.EStoreAbu.E_STORE_CUSTOM_NAME,
                                                       # custom_name='tt_train_cn')
                                                       custom_name='all_top_train_cn')
    abu_result_tuple_test = abu.load_abu_result_tuple(n_folds=7, store_type=abupy.EStoreAbu.E_STORE_CUSTOM_NAME,
                                                      # custom_name='tt_test_cn')
                                                      custom_name='all_top_train_cn')
    metrics_train = AbuMetricsBase(*abu_result_tuple_train)
    metrics_train.fit_metrics()
    metrics_test = AbuMetricsBase(*abu_result_tuple_test)
    metrics_test.fit_metrics()

    return abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test


def sample_112():
    """
    11.2.1 角度主裁, 11.2.2 使用全局最优对分类簇集合进行筛选
    :return:
    """

    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    orders_pd_train = abu_result_tuple_train.orders_pd
    # 参数为orders_pd
    ump_deg = AbuUmpMainDeg(orders_pd_train)
    # df即由之前ump_main_make_xy生成的类df，表11-1所示
    print('ump_deg.fiter.df.head():\n', ump_deg.fiter.df.head())

    # 耗时操作，大概需要10几分钟，具体根据电脑性能，cpu情况
    _ = ump_deg.fit(brust_min=False, show=False)
    print('ump_deg.cprs:\n', ump_deg.cprs)
    max_failed_cluster = ump_deg.cprs.loc[ump_deg.cprs.lrs.argmax()]
    print('失败概率最大的分类簇{0}, 失败率为{1:.2f}%, 簇交易总数{2}, 簇平均交易获利{3:.2f}%'.format(
        ump_deg.cprs.lrs.argmax(), max_failed_cluster.lrs * 100, max_failed_cluster.lcs, max_failed_cluster.lms * 100))

    cpt = int(ump_deg.cprs.lrs.argmax().split('_')[0])
    print('cpt:\n', cpt)
    ump_deg.show_parse_rt(ump_deg.rts[cpt])

    max_failed_cluster_orders = ump_deg.nts[ump_deg.cprs.lrs.argmax()]

    print('max_failed_cluster_orders:\n', max_failed_cluster_orders)

    abupy.ml.show_orders_hist(max_failed_cluster_orders,
                              ['buy_deg_ang21', 'buy_deg_ang42', 'buy_deg_ang60', 'buy_deg_ang252'])
    print('分类簇中deg_ang60平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_deg_ang60.mean()))

    print('分类簇中deg_ang21平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_deg_ang21.mean()))

    print('分类簇中deg_ang42平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_deg_ang42.mean()))

    print('分类簇中deg_ang252平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_deg_ang252.mean()))

    abupy.ml.show_orders_hist(orders_pd_train, ['buy_deg_ang21', 'buy_deg_ang42', 'buy_deg_ang60', 'buy_deg_ang252'])
    print('训练数据集中deg_ang60平均值为{0:.2f}'.format(
        orders_pd_train.buy_deg_ang60.mean()))

    print('训练数据集中deg_ang21平均值为{0:.2f}'.format(
        orders_pd_train.buy_deg_ang21.mean()))

    print('训练数据集中deg_ang42平均值为{0:.2f}'.format(
        orders_pd_train.buy_deg_ang42.mean()))

    print('训练数据集中deg_ang252平均值为{0:.2f}'.format(
        orders_pd_train.buy_deg_ang252.mean()))

    """
        11.2.2 使用全局最优对分类簇集合进行筛选
    """
    brust_min = ump_deg.brust_min()
    print('brust_min:', brust_min)

    llps = ump_deg.cprs[(ump_deg.cprs['lps'] <= brust_min[0]) & (ump_deg.cprs['lms'] <= brust_min[1]) & (
        ump_deg.cprs['lrs'] >= brust_min[2])]
    print('llps:\n', llps)

    print(ump_deg.choose_cprs_component(llps))
    ump_deg.dump_clf(llps)


"""
    11.2.3 跳空主裁
"""


def sample_1123():
    """
    11.2.3 跳空主裁
    :return:
    """
    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    orders_pd_train = abu_result_tuple_train.orders_pd
    ump_jump = AbuUmpMainJump.ump_main_clf_dump(orders_pd_train, save_order=False)
    print(ump_jump.fiter.df.head())

    print('失败概率最大的分类簇{0}'.format(ump_jump.cprs.lrs.argmax()))
    # 拿出跳空失败概率最大的分类簇
    max_failed_cluster_orders = ump_jump.nts[ump_jump.cprs.lrs.argmax()]
    # 显示失败概率最大的分类簇，表11-6所示
    print('max_failed_cluster_orders:\n', max_failed_cluster_orders)

    ml.show_orders_hist(max_failed_cluster_orders, feature_columns=['buy_diff_up_days', 'buy_jump_up_power',
                                                                    'buy_diff_down_days', 'buy_jump_down_power'])

    print('分类簇中jump_up_power平均值为{0:.2f}， 向上跳空平均天数{1:.2f}'.format(
        max_failed_cluster_orders.buy_jump_up_power.mean(), max_failed_cluster_orders.buy_diff_up_days.mean()))

    print('分类簇中jump_down_power平均值为{0:.2f}, 向下跳空平均天数{1:.2f}'.format(
        max_failed_cluster_orders.buy_jump_down_power.mean(), max_failed_cluster_orders.buy_diff_down_days.mean()))

    print('训练数据集中jump_up_power平均值为{0:.2f}，向上跳空平均天数{1:.2f}'.format(
        orders_pd_train.buy_jump_up_power.mean(), orders_pd_train.buy_diff_up_days.mean()))

    print('训练数据集中jump_down_power平均值为{0:.2f}, 向下跳空平均天数{1:.2f}'.format(
        orders_pd_train.buy_jump_down_power.mean(), orders_pd_train.buy_diff_down_days.mean()))


"""
    11.2.4 价格主裁
"""


def sample_1124():
    """
    11.2.4 价格主裁
    :return:
    """
    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    orders_pd_train = abu_result_tuple_train.orders_pd
    ump_price = AbuUmpMainPrice.ump_main_clf_dump(orders_pd_train, save_order=False)
    print('ump_price.fiter.df.head():\n', ump_price.fiter.df.head())

    print('失败概率最大的分类簇{0}'.format(ump_price.cprs.lrs.argmax()))

    # 拿出价格失败概率最大的分类簇
    max_failed_cluster_orders = ump_price.nts[ump_price.cprs.lrs.argmax()]
    # 表11-8所示
    print('max_failed_cluster_orders:\n', max_failed_cluster_orders)


"""
    11.2.5 波动主裁
"""


def sample_1125():
    """
    11.2.5 波动主裁
    :return:
    """
    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    orders_pd_train = abu_result_tuple_train.orders_pd
    # 文件保存在~/abu/data/save_png/中
    ump_wave = abupy.AbuUmpMainWave.ump_main_clf_dump(orders_pd_train, save_order=True)
    print('ump_wave.fiter.df.head():\n', ump_wave.fiter.df.head())

    print('失败概率最大的分类簇{0}'.format(ump_wave.cprs.lrs.argmax()))
    # 拿出波动特征失败概率最大的分类簇
    max_failed_cluster_orders = ump_wave.nts[ump_wave.cprs.lrs.argmax()]
    # 表11-10所示
    print('max_failed_cluster_orders:\n', max_failed_cluster_orders)

    ml.show_orders_hist(max_failed_cluster_orders, feature_columns=['buy_wave_score1', 'buy_wave_score3'])

    print('分类簇中wave_score1平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_wave_score1.mean()))

    print('分类簇中wave_score3平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_wave_score3.mean()))

    ml.show_orders_hist(orders_pd_train, feature_columns=['buy_wave_score1', 'buy_wave_score1'])

    print('训练数据集中wave_score1平均值为{0:.2f}'.format(
        orders_pd_train.buy_wave_score1.mean()))

    print('训练数据集中wave_score3平均值为{0:.2f}'.format(
        orders_pd_train.buy_wave_score1.mean()))


def sample_ump():
    ump_deg = AbuUmpMainDeg(predict=True)
    ump_jump = AbuUmpMainJump(predict=True)
    ump_price = AbuUmpMainPrice(predict=True)
    ump_wave = AbuUmpMainWave(predict=True)

    def apply_ml_features_ump(order, predicter, need_hit_cnt):
        if not isinstance(order.ml_features, dict):
            # 低版本pandas dict对象取出来会成为str
            ml_features = ast.literal_eval(order.ml_features)
        else:
            ml_features = order.ml_features

        return predicter.predict_kwargs(need_hit_cnt=need_hit_cnt, **ml_features)

    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    # 选取有交易结果的数据order_has_result
    order_has_result = abu_result_tuple_test.orders_pd[abu_result_tuple_test.orders_pd.result != 0]
    # 角度主裁开始裁决
    order_has_result['ump_deg'] = order_has_result.apply(apply_ml_features_ump, axis=1, args=(ump_deg, 2,))
    # 跳空主裁开始裁决
    order_has_result['ump_jump'] = order_has_result.apply(apply_ml_features_ump, axis=1, args=(ump_jump, 2,))
    # 波动主裁开始裁决
    order_has_result['ump_wave'] = order_has_result.apply(apply_ml_features_ump, axis=1, args=(ump_wave, 2,))
    # 价格主裁开始裁决
    order_has_result['ump_price'] = order_has_result.apply(apply_ml_features_ump, axis=1, args=(ump_price, 2,))

    block_pd = order_has_result.filter(regex='^ump_*')
    block_pd['sum_bk'] = block_pd.sum(axis=1)
    block_pd['result'] = order_has_result['result']

    block_pd = block_pd[block_pd.sum_bk > 0]
    print('四个裁判整体拦截正确率{:.2f}%'.format(
        block_pd[block_pd.result == -1].result.count() / block_pd.result.count() * 100))
    print('block_pd.tail():\n', block_pd.tail())

    def sub_ump_show(block_name):
        sub_block_pd = block_pd[(block_pd[block_name] == 1)]
        # 如果失败就正确 －1->1 1->0
        # noinspection PyTypeChecker
        sub_block_pd.result = np.where(sub_block_pd.result == -1, 1, 0)
        return accuracy_score(sub_block_pd[block_name], sub_block_pd.result)

    print('角度裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_deg') * 100))
    print('跳空裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_jump') * 100))
    print('波动裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_wave') * 100))
    print('价格裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_price') * 100))

    print('角度裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_deg') * 100))
    # print('跳空裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_jump') * 100))
    # print('波动裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_wave') * 100))
    # print('价格裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_price') * 100))


'''机器学习训练启动'''


def start_ump_train():
    sample_112()
    sample_1123()
    sample_1124()
    sample_1125()

if __name__ == "__main__":
    start_ump_train()
    # sample_a21()
    # sample_ump()
    # sample_112()
    # sample_1123()
    # sample_1124()
    # sample_1125()
