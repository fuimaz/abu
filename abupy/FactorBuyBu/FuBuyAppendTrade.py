# -*- encoding:utf-8 -*-
"""
    示例买入择时因子
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd

from abupy.IndicatorBu.ABuNDBoll import calc_boll
from .ABuFactorBuyBase import AbuFactorBuyXD, BuyCallMixin
from ..TLineBu.ABuTL import AbuTLine


class FuBuyAppendTrade(AbuFactorBuyXD, BuyCallMixin):
    def _init_self(self, **kwargs):
        if 'xd' not in kwargs:
            # 如果外部没有设置xd值，默认给一个30
            kwargs['xd'] = 22
        super(FuBuyAppendTrade, self)._init_self(**kwargs)
        # 代表判断上涨趋势拟合角度阀值，即长线拟合角度值多少决策为上涨，默认4
        self.up_deg_threshold = kwargs.pop('up_deg_threshold', 5)
        self.down_deg_threshold = kwargs.pop('down_deg_threshold', 2)

    def fit_day(self, today):
        upper, middle, lower = calc_boll(self.kl_pd.close)

        if pd.isnull(middle[int(today.key) - 1]):
            return None

        '''判断是否处于快速上涨区'''
        long_kl = self.past_today_kl(today, 120)
        tl_long = AbuTLine(long_kl.close, 'long')
        # 判断长周期是否属于上涨趋势
        if tl_long.is_up_trend(up_deg_threshold=self.up_deg_threshold, show=False):
            if today.close == self.xd_kl.close.min() and \
                    AbuTLine(self.xd_kl.close, 'short').is_down_trend(
                        down_deg_threshold=-self.down_deg_threshold, show=False):
                # if middle[int(today.key) - 1] < today.pre_close < middle[int(today.key) - 1] * 1.1 \
                #         and today.close >= today.pre_close:
                #     print(u"补仓，快速上涨区，跌到布林带中间线，未跌穿， 买入", "pre_close=", str(today.pre_close), "middle=",
                #           str(middle[int(today.key) - 1]),
                #           "close=", str(today.close), "middle=", str(middle[int(today.key)]), 'date=', str(today.date))
                # print("补仓")
                return self.buy_today()
