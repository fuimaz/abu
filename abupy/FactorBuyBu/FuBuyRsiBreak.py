# -*- encoding:utf-8 -*-
"""
    示例买入择时因子
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd

from abupy.IndicatorBu.ABuNDBoll import calc_boll
from abupy.IndicatorBu.ABuNDMacd import calc_macd
from abupy.IndicatorBu.ABuNDRsi import calc_rsi
from .ABuFactorBuyBase import AbuFactorBuyXD, BuyCallMixin
from ..TLineBu.ABuTL import AbuTLine

__author__ = '居尘'
__weixin__ = 'abu_quant'


# noinspection PyAttributeOutsideInit
class FuBuyRsiBreak(AbuFactorBuyXD, BuyCallMixin):
    """示例买入因子： 在AbuFactorBuyBreak基础上进行降低交易频率，提高系统的稳定性处理"""

    def _init_self(self, **kwargs):
        """
            kwargs中可以包含xd: 比如20，30，40天...突破，默认20
            kwargs中可以包含past_factor: 代表长线的趋势判断长度，默认4，long = xd * past_factor->eg: long = 20 * 4
            kwargs中可以包含up_deg_threshold: 代表判断上涨趋势拟合角度阀值，即长线拟合角度值多少决策为上涨，默认3
        """
        if 'xd' not in kwargs:
            # 如果外部没有设置xd值，默认给一个30
            kwargs['xd'] = 20
        super(FuBuyRsiBreak, self)._init_self(**kwargs)
        # 代表长线的趋势判断长度，默认4，long = xd * past_factor->eg: long = 30 * 4
        self.past_factor = kwargs.pop('past_factor', 4)
        # 代表判断上涨趋势拟合角度阀值，即长线拟合角度值多少决策为上涨，默认4
        self.up_deg_threshold = kwargs.pop('up_deg_threshold', 3)

        """
            kwargs: kwargs可选参数poly值，poly在fit_
        """
        self.time_period = kwargs.pop('time_period', 14)

    def fit_day(self, today):
        rsi = calc_rsi(self.kl_pd.close, self.time_period)
        # dif, dea, bar = calc_macd(self.kl_pd.close)

        if pd.isnull(rsi[int(today.key) - 1]):
            return None

        if int(today.key) != 0:
            pre_i = int(today.key) - 1
            now_i = int(today.key)

            # if dif[pre_i] < dea[pre_i] and dif[now_i] >= dea[now_i]:
            #     return self.buy_tomorrow()
            if rsi[pre_i] < rsi[now_i] and rsi[now_i] > 55:
                return self.buy_tomorrow()

