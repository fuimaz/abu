# -*- encoding:utf-8 -*-
from __future__ import print_function

import urllib2

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings

# noinspection PyUnresolvedReferences
import sys

from abupy.MarketBu.ABuMarketDrawing import plot_candle_from_symbol

from abupy.IndicatorBu.ABuNDBase import ECalcType

import abu_local_env

import abupy
from abupy import AbuFactorBuyBreak
from abupy import AbuFactorSellBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
from abupy import AbuBenchmark
from abupy import AbuPickTimeWorker
from abupy import AbuCapital
from abupy import AbuKLManager
from abupy import ABuTradeProxy
from abupy import ABuTradeExecute
from abupy import ABuPickTimeExecute
from abupy import AbuMetricsBase
from abupy import ABuMarket
from abupy import AbuPickTimeMaster
from abupy import ABuRegUtil
from abupy import AbuPickRegressAngMinMax
from abupy import AbuPickStockWorker
from abupy import ABuPickStockExecute
from abupy import AbuPickStockPriceMinMax
from abupy import AbuPickStockMaster
from abupy.CoreBu import ABuEnv

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})

# 市场，缓存，数据源配置
ABuEnv.g_data_fetch_mode = abupy.EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL
ABuEnv.g_market_target = abupy.EMarketTargetType.E_MARKET_TARGET_CN
ABuEnv.g_market_source = abupy.EMarketSourceType.E_MARKET_SOURCE_tx
# ABuEnv.g_data_cache_type = abupy.EDataCacheType.E_DATA_CACHE_HDF5
ABuEnv.g_data_cache_type = abupy.EDataCacheType.E_DATA_CACHE_CSV
ABuEnv.g_calc_type = ECalcType.E_FROM_TA

reload(sys)
sys.setdefaultencoding('utf-8')

def sample_821_2():
    """
    8.2.1_2 ABuPickStockExecute
    :return:
    """
    stock_pickers = [{'class': AbuPickRegressAngMinMax,
                      'threshold_ang_min': 0.0, 'threshold_ang_max': 10.0,
                      'reversed': False}]

    choice_symbols = ['601398', '601988', '601939', '603993', '600999', '300059', '600900', '601328', '601288', '600887', '600029', '000002']
    benchmark = AbuBenchmark()
    capital = AbuCapital(1000000, benchmark)
    kl_pd_manager = AbuKLManager(benchmark, capital)

    print('ABuPickStockExecute.do_pick_stock_work:\n', ABuPickStockExecute.do_pick_stock_work(choice_symbols, benchmark,
                                                                                              capital, stock_pickers))

    kl_pd_sfun = kl_pd_manager.get_pick_stock_kl_pd('601398')
    print('sfun 选股周期内角度={}'.format(round(ABuRegUtil.calc_regress_deg(kl_pd_sfun.close), 3)))


def sample_821_3():
    """
    8.2.1_2 ABuPickStockExecute
    :return:
    """
    stock_pickers = [{'class': abupy.FuWeekVolumeBoll,
                      'threshold_ang_min': 0.0, 'threshold_ang_max': 10.0,
                      'reversed': False}]

    choice_symbols = ['601398', '601988', '601939', '603993', '600999', '300059', '600900', '601328', '601288', '600887', '600029', '000002']
    choice_symbols = ['sz000983', 'sh600338', 'sh600511', 'sh600196', 'sh600423', 'sz399136', 'sz002044', 'sh601800', 'sz300132', 'sz300133', 'sh000821', 'sz300003', 'sz300009', 'sz200045', 'sh600998', 'sz300313', 'sh601607', 'sz002644', 'sh600697', 'sz000627', 'sh000003', 'sz399302', 'sh600984', 'sz399301', 'sz000916', 'sz000911', 'sz000912', 'sz000688', 'sh600079', 'sh601101', 'sz000861', 'sz000736', 'sz002053', 'sz000048', 'sh600703', 'sh000814', 'sz300015', 'sh000818', 'sz399352', 'sz399356', 'sh900911', 'sh600395', 'sh000075', 'sz002323', 'sh000101', 'sh600285', 'sh600882', 'sz000789', 'sh601398', 'sz000898', 'sh601390', 'sh601009', 'sh601001', 'sz000525', 'sh600713', 'sh601628', 'sz399299', 'sz399298', 'sh600800', 'sh000808', 'sh900909', 'sh900908', 'sh000061', 'sh000068', 'sh000116', 'sz000617', 'sh600535', 'sz000792', 'sz000889', 'sz000065', 'sh601015', 'sz000089', 'sh600871', 'sz002412', 'sz399400', 'sz399402', 'sz399404', 'sh000057', 'sh900930', 'sh900936', 'sh900934', 'sh900935', 'sh600267', 'sz000650', 'sz399978', 'sh600485', 'sh601021', 'sh601601', 'sh600208', 'sh601288', 'sh600062', 'sh600015', 'sh600016', 'sz300197', 'sz300199', 'sz399413', 'sz399411', 'sz399416', 'sh000134', 'sh000136', 'sh000139', 'sz002007', 'sh600258', 'sh600123', 'sz000511', 'sh601618', 'sh600745', 'sz399170', 'sh000923', 'sz399319', 'sz399554', 'sz399555', 'sz002530', 'sh000145', 'sz002070', 'sh000149', 'sz399220', 'sh601998', 'sh600111', 'sh600023', 'sz000560', 'sh601699', 'sz399305', 'sz399431', 'sz000766', 'sz399436', 'sz399230', 'sz399237', 'sz002661', 'sz002599', 'sh000155', 'sh000152', 'sh000151', 'sh600806', 'sh601988', 'sh600693', 'sh600699', 'sh600582', 'sz000995', 'sh600566', 'sh601318', 'sz399150', 'sz399441', 'sz399200', 'sh000841', 'sh600917', 'sz002128', 'sh600176', 'sz000968', 'sh600771', 'sh600579', 'sh600578', 'sh600572', 'sh600681', 'sh600680', 'sz399140', 'sz000540', 'sz000545', 'sz200022', 'sz200026', 'sz200025', 'sz399210', 'sz200029', 'sz002601', 'sz002656', 'sz002204', 'sz002737', 'sz000748', 'sh600965', 'sz002135', 'sh000934', 'sh601169', 'sh601899', 'sh601898', 'sh600549', 'sh600546', 'sh600545', 'sz000778', 'sh600141', 'sh600145', 'sh601231', 'sz399139', 'sz000630', 'sz000613', 'sz399137', 'sz399130', 'sz399131', 'sz399132', 'sz399133', 'sz200019', 'sz300146', 'sz300144', 'sz399661', 'sz002701', 'sh600971', 'sz002382', 'sz002385', 'sh600085', 'sh603158', 'sz002602', 'sh601939', 'sh600007', 'sz399645', 'sh600000', 'sh601339', 'sh601336', 'sh000125', 'sz399674', 'sh000974', 'sz399160', 'sz002653', 'sz002717', 'sz200726', 'sz399647', 'sz300294', 'sh000100', 'sz300347', 'sh600348', 'sh000933', 'sh600401', 'sh000109', 'sz000034', 'sh600623', 'sz000581', 'sz000672', 'sz300028', 'sh603368', 'sh000023', 'sh000021', 'sz002198', 'sh600432', 'sh603989', 'sz000937', 'sh600508', 'sh600500', 'sh000159', 'sz000732', 'sh600188', 'sz000598', 'sz000029', 'sz000028', 'sz399394', 'sh000832', 'sz300036', 'sz200053', 'sz300326', 'sz002742', 'sz300253', 'sh000013', 'sh000011']
    # choice_symbols = ['002656', '000903']
    benchmark = AbuBenchmark(n_folds=15)
    capital = AbuCapital(1000000, benchmark)
    kl_pd_manager = AbuKLManager(benchmark, capital)

    stock_pickers = ABuPickStockExecute.do_pick_stock_work(None, benchmark,
    # stock_pickers = ABuPickStockExecute.do_pick_stock_work(choice_symbols, benchmark,
                                           capital, stock_pickers)
    print('ABuPickvStockExecute.do_pick_stock_work:\n', stock_pickers)
    for stock_symbol in stock_pickers:
        if ~fetch_stock_base_info(stock_symbol):
            continue
        draw_candle(stock_symbol, 15)


def fetch_stock_base_info(stock_symbol):
    url = "http://qt.gtimg.cn/q=%s" % stock_symbol
    res = None

    try:
        req = urllib2.Request(url)
        res_data = urllib2.urlopen(req)
        res = res_data.read().decode("gbk").encode("utf-8")
    except :
        return False
    if res is None:
        return False

    info = str.split(res, '~')
    if len(info) < 47:
        print(info)
        return False

    name = unicode(info[1])
    if name.find(u"Ｂ") != -1 or name.find(u"指数") != -1 \
            or name.find(u"债") != -1 or name.find(u"退") != -1:
        return False
    print(unicode(info[1]), info[2], "价格", info[3], "市盈率", info[39], "流通市值", info[44], "总市值", info[45], "市净率", info[46])
    return True


def draw_candle(stock_symbols, n_folds = 2):
    plot_candle_from_symbol(stock_symbols, n_folds, save=True)

if __name__ == "__main__":
    # sample_821_2()
    sample_821_3()
    # fetch_stock_base_info('sz000629')
    # sample_a22()
    # sample_a23()