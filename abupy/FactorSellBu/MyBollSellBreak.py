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


class MyBollSellBreak(AbuFactorSellBase):
    """示例向下突破卖出择时因子"""

    def _init_self(self, **kwargs):
        """kwargs中必须包含: 突破参数xd 比如20，30，40天...突破"""

        self.time_period = kwargs.pop('time_period', 20)
        self.nb_dev = kwargs.pop('nb_dev', 2)
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
        dif, dea, bar = calc_macd(self.kl_pd.close)

        if pd.isnull(upper[int(today.key)]):
            return None

        # if self.is_wave_period(today):
        #     if today.close < middle[int(today.key)]:
        #         print(u"震荡卖出")
        #         print("close=", str(today.close), "upper=", str(lower[int(today.key)]), 'date=', str(today.date))
        #         for order in orders:
        #             self.sell_tomorrow(order)

        '''快速上涨macd策略卖出'''
        if self._is_fast_up_period(today) and int(today.key) != 0:
            pre_i = int(today.key) - 1
            now_i = int(today.key)
            if dif[pre_i] > dea[pre_i] and dif[now_i] <= dea[now_i]:
                # print(u"快速上涨macd卖出", "close=", str(today.close), "middle=", str(middle[int(today.key)]), 'date=', str(today.date))
                for order in orders:
                    self.sell_tomorrow(order)

        '''原本是达到布林线上端时卖出，但回测结果并不好'''
        '''跌破布林线下端卖出'''
        if today.close < lower[int(today.key)]:
            # print(u"跌穿布林带底线卖出", "close=", str(today.close), "lower=", str(lower[int(today.key)]), 'date=', str(today.date))
            for order in orders:
                self.sell_tomorrow(order)

    def is_wave_period(self, today):
        kl_df = past_today_kl(self.kl_pd, today, 21)
        line = calc_wave_std(kl_df, show=False)
        if line.close >= line.high:
            # print(u'当日振幅', str(line.close), str(line.high))
            return True

        return False

    def _is_fast_up_period(self, today):
        kl_df = past_today_kl(self.kl_pd, today, 21)
        if len(kl_df) == 0:
            return False

        deg = ABuRegUtil.calc_regress_deg(kl_df.close.values, show=False)
        if deg > 7:
            return True
        return False
