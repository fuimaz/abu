# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

import pandas as pd
import threading

from abupy.CoreBu import ABuEnv
from abupy.UtilBu import ABuFileUtil


class FuTodayCanBuyRecord:

    canBuyList = pd.DataFrame(columns=['symbol', 'date', 'price', 'type'])
    lock = threading.Lock()

    def record_today_can_buy_stock(self, today, buy_symbol, type):
        if today is None:
            return

        today_date = int(str(datetime.date.today()).replace('-', ''))
        if today_date != today.date:
            return

        # print(buy_symbol, today.date, today.close)

        df_temp = pd.DataFrame([[buy_symbol, today.date, today.close, type]], columns=['symbol', 'date', 'price', 'type'])
        FuTodayCanBuyRecord.lock.acquire()
        FuTodayCanBuyRecord.canBuyList = FuTodayCanBuyRecord.canBuyList.append(df_temp)
        # print(FuTodayCanBuyRecord.canBuyList)
        FuTodayCanBuyRecord.lock.release()

    def store_today_can_buy_stock(self):
        base_dir = 'out_put'
        # 时间字符串
        date_dir = datetime.datetime.now().strftime("%Y_%m_%d")
        fn = os.path.join(ABuEnv.g_project_data_dir, base_dir, date_dir, 'today_actions.csv')
        ABuFileUtil.ensure_dir(fn)
        ABuFileUtil.dump_df_csv(fn, FuTodayCanBuyRecord.canBuyList)