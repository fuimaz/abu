# -*- encoding:utf-8 -*-
"""
    日内滑点买入示例实现：均价买入
    最简单的回测买入方式，优点简单，且回测高效，在回测交易
    数量足够多的前提下也能接近实盘
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from .ABuSlippageBuyBase import AbuSlippageBuyBase, slippage_limit_up

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""外部修改默认下跌阀值使用如：abupy.slippage.sbm.g_open_down_rate = 0.02"""
g_open_down_rate = 0.07


class FuSlippageBuyOpen(AbuSlippageBuyBase):
    """示例日内滑点均价买入类"""

    @slippage_limit_up
    def fit_price(self):
        """
        取当天交易日的收市价做为决策价格
        :return: 最终决策的当前交易买入价格
        """

        # TODO 基类提取作为装饰器函数，子类根据需要选择是否装饰，并且添加上根据order的call，put明确细节逻辑
        if self.kl_pd_buy.pre_close == 0 or (self.kl_pd_buy.open / self.kl_pd_buy.pre_close) < (1 - g_open_down_rate):
            # 开盘就下跌一定比例阀值，放弃单子
            return np.inf
        # 买入价格为当天收市价
        self.buy_price = self.kl_pd_buy['open']
        # 返回最终的决策价格
        return self.buy_price
