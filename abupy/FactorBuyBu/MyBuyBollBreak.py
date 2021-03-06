# -*- encoding:utf-8 -*-
"""
    示例买入择时因子
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import logging

from abupy.CoreBu.FuTodayRecord import FuTodayCanBuyRecord
from abupy.IndicatorBu.ABuNDBoll import calc_boll
from abupy.IndicatorBu.ABuNDMacd import calc_macd
from .ABuFactorBuyBase import AbuFactorBuyXD, BuyCallMixin
from ..TLineBu.ABuTL import AbuTLine

__author__ = '居尘'
__weixin__ = 'abu_quant'


# noinspection PyAttributeOutsideInit
class MyBuyBollBreak(AbuFactorBuyXD, BuyCallMixin):
    """示例买入因子： 在AbuFactorBuyBreak基础上进行降低交易频率，提高系统的稳定性处理"""

    def _init_self(self, **kwargs):
        """
            kwargs中可以包含xd: 比如20，30，40天...突破，默认20
            kwargs中可以包含past_factor: 代表长线的趋势判断长度，默认4，long = xd * past_factor->eg: long = 20 * 4
            kwargs中可以包含up_deg_threshold: 代表判断上涨趋势拟合角度阀值，即长线拟合角度值多少决策为上涨，默认3
        """
        if 'xd' not in kwargs:
            # 如果外部没有设置xd值，默认给一个30
            kwargs['xd'] = 2
        super(MyBuyBollBreak, self)._init_self(**kwargs)
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

    def fit_month(self, today):
        if self.is_atr_off:
            return False

        # fit_month即在回测策略中每一个月执行一次的方法
        # 策略中拥有self.benchmark，即交易基准对象，AbuBenchmark实例对象，benchmark.kl_pd即对应的市场大盘走势
        benchmark_df = self.benchmark.kl_pd
        # 拿出大盘的今天
        benchmark_today = benchmark_df[benchmark_df.date == today.date]
        if benchmark_today.empty:
            return 0
        # 要拿大盘最近一个月的走势，准备切片的start，end
        end_key = int(benchmark_today.iloc[0].key)
        start_key = end_key - 20
        if start_key < 0:
            return False

        # 使用切片切出从今天开始向前20天的数据
        benchmark_month = benchmark_df[start_key:end_key + 1]
        # 通过大盘最近一个月的收盘价格做为参数构造AbuTLine对象
        benchmark_month_line = AbuTLine(benchmark_month.close, 'benchmark month line')
        # 计算这个月最少需要几次拟合才能代表走势曲线
        least = benchmark_month_line.show_least_valid_poly(show=False)

        if least >= self.poly:
            # 如果最少的拟合次数大于阀值self.poly，说明走势成立，大盘非震荡走势，解锁交易
            self.lock = False
        else:
            # 如果最少的拟合次数小于阀值self.poly，说明大盘处于震荡走势，封锁策略进行交易
            self.lock = True

    def fit_day(self, today):
        # logging.info("enter fit_day, %s %d %f" % (self.kl_pd.name, today.date, today.close))
        if self.lock:
            # 如果封锁策略进行交易的情况下，策略不进行择时
            return None

        short_kl = self.past_today_kl(today, 21)
        sl_short = AbuTLine(short_kl.close, 'short')
        # 判断长周期是否属于上涨趋势
        # if not tl_long.is_up_trend(up_deg_threshold=self.up_deg_threshold, show=False):
        #     return None

        upper, middle, lower = calc_boll(self.kl_pd.close, self.time_period, self.nb_dev)
        dif, dea, bar = calc_macd(self.kl_pd.close)

        # if pd.isnull(middle[int(today.key) - 1]):
        if pd.isnull(middle[int(today.key)]):
            return None

        # long_kl = self.past_today_kl(today, 21)
        # long_line = AbuTLine(long_kl.close, 'long')
        # if long_line.is_down_trend(down_deg_threshold=-3, show=False):
        #     return None

        '''处于快速上涨区, 跌到布林带中间线买入效果不好'''
        # if tl_short.is_up_trend(6, show=False):
        #     if middle[int(today.key) - 1] < today.pre_close < middle[int(today.key) - 1] * 1.05 and today.close >= today.pre_close:
        #         print(u"快速上涨区，跌到布林带中间线，未跌穿， 买入", "pre_close=", str(today.pre_close), "middle=", str(middle[int(today.key) - 1]),
        #               "close=", str(today.close), "middle=", str(middle[int(today.key)]), 'date=', str(today.date))
        #         return self.buy_tomorrow()

        if int(today.key) != 0:
            pre_i = int(today.key) - 1
            now_i = int(today.key)
            '''0值之上不操作买入, 增加这个逻辑后整体收益不好，去掉'''

            # if not sl_short.is_up_trend(up_deg_threshold=2, show=False):
            #     return None

            if dif[pre_i] < dea[pre_i] and dif[now_i] >= dea[now_i]:
                # print(u"macd金叉， 买入", "pre_close=", str(today.pre_close), "middle=", str(middle[int(today.key) - 1]),
                #       "close=", str(today.close), "middle=", str(middle[int(today.key)]), 'date=', str(today.date))
                # order =
                # if order is not None:
                #     today_record = FuTodayCanBuyRecord()
                #     today_record.record_today_can_buy_stock(today, self.kl_pd.name, 1)
                return self.buy_today()

        if today.pre_close <= middle[int(today.key) - 1] and today.close >= middle[int(today.key)]:
            # print(u"穿过布林带中间线， 买入", "pre_close=", str(today.pre_close), "middle=", str(middle[int(today.key) - 1]),
            #       "close=", str(today.close), "middle=", str(middle[int(today.key)]), 'date=', str(today.date))
            # order = self.buy_today()
            return self.buy_today()
