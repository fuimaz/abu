# -*- encoding:utf-8 -*-
"""
    示例买入择时因子
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

from abupy.IndicatorBu.ABuNDBoll import calc_boll
from abupy.IndicatorBu.ABuNDMacd import calc_macd
from abupy.IndicatorBu.ABuNDMa import calc_ma
from .ABuFactorBuyBase import AbuFactorBuyXD, BuyCallMixin
from ..TLineBu.ABuTL import AbuTLine

__author__ = '居尘'
__weixin__ = 'abu_quant'


# noinspection PyAttributeOutsideInit
class FuWeekBuyBollBreak(AbuFactorBuyXD, BuyCallMixin):
    """n周内放量2.5被且涨超过-5%，且达到了boll带顶端后，又达到了boll带底端"""

    def _init_self(self, **kwargs):
        """
            kwargs中可以包含xd: 比如20，30，40天...突破，默认20
            kwargs中可以包含past_factor: 代表长线的趋势判断长度，默认4，long = xd * past_factor->eg: long = 20 * 4
            kwargs中可以包含up_deg_threshold: 代表判断上涨趋势拟合角度阀值，即长线拟合角度值多少决策为上涨，默认3
        """
        if 'xd' not in kwargs:
            # 如果外部没有设置xd值，默认给一个30
            kwargs['xd'] = 2
        super(FuWeekBuyBollBreak, self)._init_self(**kwargs)
        # 代表长线的趋势判断长度，默认4，long = xd * past_factor->eg: long = 30 * 4
        self.past_factor = kwargs.pop('past_factor', 4)
        # 代表判断上涨趋势拟合角度阀值，即长线拟合角度值多少决策为上涨，默认4
        self.up_deg_threshold = kwargs.pop('up_deg_threshold', 3)

        """
            kwargs: kwargs可选参数poly值，poly在fit_month中和每一个月大盘计算的poly比较，
            若是大盘的poly大于poly认为走势震荡，poly默认为2
            atr_off是否大盘走势震荡时停止交易
        """
        # poly阀值，self.poly在fit_month中和每一个月大盘计算的poly比较，若是大盘的poly大于poly认为走势震荡
        self.poly = kwargs.pop('poly', 2)
        # 是否封锁买入策略进行择时交易
        self.lock = False
        self.is_atr_off = kwargs.pop('atr_off', True)
        self.time_period = kwargs.pop('time_period', 20)
        self.nb_dev = kwargs.pop('nb_dev', 2)
        self.status = 0
        self.bear = False

    def fit_month(self, today):
        pass

    def fit_day(self, today):
        # logging.info("enter fit_day, %s %d %f" % (self.kl_pd.name, today.date, today.close))
        self.pause(today)
        if self.lock:
            # 如果封锁策略进行交易的情况下，策略不进行择时
            return None

        upper, middle, lower = calc_boll(self.kl_pd.close, self.time_period, self.nb_dev)

        if pd.isnull(middle[int(today.key)]):
            return None

        if today.close <= lower[int(today.key)]:
            return self.buy_today()

    def pause(self, today):
        today_volume = self.kl_pd.volume[int(today.key)]

        upper, middle, lower = calc_boll(self.kl_pd.close, self.time_period, self.nb_dev)

        if pd.isnull(middle[int(today.key)]):
            return None

        dif = today.pre_close - today.close
        ma_volume = calc_ma(self.kl_pd.volume, 20)
        if pd.isnull(ma_volume[int(today.key)]):
            return
        if today_volume / ma_volume[int(today.key)] > 2.5 and dif < 0 and dif / today.pre_close < -0.1:
            self.lock = True
            self.status = 0

        if today_volume / ma_volume[int(today.key)] > 2.5 and dif / today.pre_close > -0.05:
            self.status = 1

        if self.status == 1 and today.close >= upper[int(today.key)] * 1.05:
            self.lock = False
            self.status = 0

        if today.close >= upper[int(today.key)]:
            self.lock = False
            self.status = 0

    def is_bear(self, today):
        ma_close = calc_ma(self.kl_pd.close, 48)
        if pd.isnull(ma_close[int(today.key)]):
            return
        if today.close < ma_close[int(today.key)]:
            self.bear = True
        else:
            self.bear = False
