# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

import pandas as pd
import threading
import logging

from abupy.CoreBu import ABuEnv
from abupy.UtilBu import ABuFileUtil


class FuTodayCanBuyRecord:

    canBuyList = pd.DataFrame(columns=['symbol', 'date', 'price', 'type'])
    lock = threading.Lock()

    def record_today_can_buy_stock(self, today, buy_symbol, type):
        if today is None:
            return

        today_date = int(str(datetime.date.today()).replace('-', ''))
        logging.info("before check date, %s %d %f" % (buy_symbol, today.date, today.close))
        # print(today.date, int(today.date), today_date)
        if today_date != int(today.date):
            return

        print(buy_symbol, today.date, today.close)
        logging.info("%s %d %f" % (buy_symbol, today.date, today.close))

        df_temp = pd.DataFrame([[buy_symbol, today.date, today.close, type]], columns=['symbol', 'date', 'price', 'type'])
        # FuTodayCanBuyRecord.lock.acquire()
        FuTodayCanBuyRecord.canBuyList = FuTodayCanBuyRecord.canBuyList.append(df_temp)
        # print(FuTodayCanBuyRecord.canBuyList)
        # FuTodayCanBuyRecord.lock.release()

    def store_today_can_buy_stock(self):
        base_dir = 'out_put'
        # 时间字符串
        date_dir = datetime.datetime.now().strftime("%Y_%m_%d")
        fn = os.path.join(ABuEnv.g_project_data_dir, base_dir, date_dir, 'today_actions.csv')
        ABuFileUtil.ensure_dir(fn)
        ABuFileUtil.dump_df_csv(fn, FuTodayCanBuyRecord.canBuyList)

    def load_today_stock(self):
        base_dir = 'out_put'
        # 时间字符串
        date_dir = datetime.datetime.now().strftime("%Y_%m_%d")
        fn = os.path.join(ABuEnv.g_project_data_dir, base_dir, date_dir, 'today_actions.csv')
        return ABuFileUtil.load_df_csv(fn)

    def load_today_stock_list(self):
        df = self.load_today_stock()
        if df is None or len(df.symbol.tolist()) == 0:
            return None
        stock_list = []
        raw_list = set(df.symbol.tolist())
        for symbol in raw_list:
            stock_list.append(str(symbol))
        return stock_list