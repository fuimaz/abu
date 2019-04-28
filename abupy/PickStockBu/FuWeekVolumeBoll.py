# -*- encoding:utf-8 -*-
"""
    选股示例因子：价格拟合角度选股因子
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import pandas as pd
import numpy as np
from abupy.IndicatorBu.ABuNDMa import calc_ma

from abupy.IndicatorBu.ABuNDBoll import calc_boll

from ..UtilBu import ABuRegUtil
from .ABuPickStockBase import AbuPickStockBase, reversed_result

__author__ = '阿布'
__weixin__ = 'abu_quant'


class FuWeekVolumeBoll(AbuPickStockBase):
    """n周内放量2.5被且涨超过-5%，且达到了boll带顶端后，又达到了boll带底端 """
    def _init_self(self, **kwargs):
        self.xd = 5000
        self.back_count = kwargs.pop('back_count', 90)
        pass

    @reversed_result
    def fit_pick(self, kl_pd, target_symbol):
        upper, middle, lower = calc_boll(kl_pd.close, 20, 2)
        ma_volume = calc_ma(kl_pd.volume, 20)
        status = 0
        lock = False

        if len(kl_pd) < self.back_count:
            return False

        for index in np.arange(len(kl_pd) - self.back_count, len(kl_pd)):
            today_volume = kl_pd.volume[index]

            if pd.isnull(middle[index]):
                continue

            dif = kl_pd.close[index] - kl_pd.close[index - 1]
            if pd.isnull(ma_volume[index]):
                return
            if today_volume / ma_volume[index] > 2 and dif < 0 and dif / kl_pd.close[index - 1] < -0.1:
                lock = True
                status = 0

            '放量且不是大跌的情况，预解除锁定'
            if today_volume / ma_volume[index] > 2.5 and dif / kl_pd.close[index - 1] > -0.05:
                status = 1

            '放量后涨到boll带顶端，解除锁定'
            if status == 1 and kl_pd.close[index] >= upper[index] * 1.05:
                lock = False
                status = 0

            # if kl_pd.close[index] >= upper[index]:
            #     self.lock = False
            #     self.status = 0

        if ~lock:
            # return True
            if kl_pd.close[len(kl_pd.volume) - 1] < lower[len(kl_pd.volume) - 1] * 1.05:
                return True

        return False

    def fit_first_choice(self, pick_worker, choice_symbols, *args, **kwargs):
        raise NotImplementedError('AbuPickRegressAng fit_first_choice unsupported now!')
