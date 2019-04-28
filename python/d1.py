# -*- encoding:utf-8 -*-
from __future__ import print_function
import seaborn as sns
import warnings

# noinspection PyUnresolvedReferences
from abupy.IndicatorBu.ABuNDBase import ECalcType

import abupy
from abupy.CoreBu import ABuEnv
from abupy.IndicatorBu.ABuNDBoll import plot_boll_from_klpd

from abupy.MarketBu.ABuMarketDrawing import plot_candle_form_klpd, plot_candle_from_symbol

import abu_local_env
from abupy import ABuSymbolPd, ABuScalerUtil
import matplotlib.pyplot as plt
import numpy as np
from abupy import EMarketSourceType
from abupy import EMarketDataFetchMode
from abupy import AbuFactorBuyBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
from abupy import AbuMetricsBase
from abupy import abu

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})

ABuEnv.g_data_fetch_mode = abupy.EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL
ABuEnv.g_market_target = abupy.EMarketTargetType.E_MARKET_TARGET_CN
ABuEnv.g_market_source = abupy.EMarketSourceType.E_MARKET_SOURCE_tx
# ABuEnv.g_data_cache_type = abupy.EDataCacheType.E_DATA_CACHE_HDF5
ABuEnv.g_data_cache_type = abupy.EDataCacheType.E_DATA_CACHE_CSV
ABuEnv.g_calc_type = ECalcType.E_FROM_TA


def sample_a21():
    choice_symbols = '601398'
    plot_candle_from_symbol(choice_symbols, n_folds=15)


def sample_a22():
    choice_symbols = '601398'
    kl_pd = ABuSymbolPd.make_kl_df(choice_symbols, n_folds=15)
    plot_boll_from_klpd(kl_pd)


if __name__ == "__main__":
    # sample_a21()
    sample_a22()

