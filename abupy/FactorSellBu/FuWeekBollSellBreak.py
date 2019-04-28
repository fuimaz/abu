# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子：突破卖出择时因子
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd

from abupy.IndicatorBu.ABuNDMacd import calc_macd
from abupy.TLineBu.ABuTLWave import calc_wave_std
from abupy.UtilBu import ABuRegUtil

from abupy.IndicatorBu.ABuNDBoll import calc_boll
from abupy.UtilBu.FuSplitDataFrame import past_today_kl
from .ABuFactorSellBase import AbuFactorSellBase, ESupportDirection

__author__ = '阿布'
__weixin__ = 'abu_quant'


class FuWeekBollSellBreak(AbuFactorSellBase):
    """示例向下突破卖出择时因子"""

    def _init_self(self, **kwargs):
        """kwargs中必须包含: 突破参数xd 比如20，30，40天...突破"""

        self.time_period = kwargs.pop('time_period', 20)
        self.nb_dev = kwargs.pop('nb_dev', 3)
        # self.sell_type_extra = '{}:{}'.format(self.__class__.__name__, self.xd)

    def support_direction(self):
        """支持的方向，只支持正向"""
        return [ESupportDirection.DIRECTION_CAll.value]

    def fit_day(self, today, orders):
        """
        寻找向下突破作为策略卖出驱动event
        :param today: 当前驱动的交易日金融时间序列数据
        :param orders: 买入择时策略中生成的订单序列
        """
        upper, middle, lower = calc_boll(self.kl_pd.close, self.time_period, self.nb_dev)

        if pd.isnull(upper[int(today.key)]):
            return None

        '''达到布林线上端时卖出'''
        if today.close >= upper[int(today.key)]:
            # print(u"跌穿布林带底线卖出", "close=", str(today.close), "lower=", str(lower[int(today.key)]), 'date=', str(today.date))
            for order in orders:
                if order.buy_price < today.close:
                    self.sell_today(order)
                    # break

