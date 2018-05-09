# -*- encoding:utf-8 -*-
from __future__ import print_function

import logging
import os
import time

import datetime

import abupy
import tushare as ts
import pandas as pd

from abupy import AbuSymbolCN
from abupy.CoreBu import ABuEnv
from abupy.IndicatorBu.ABuNDBoll import calc_boll
from abupy.IndicatorBu.ABuNDMacd import calc_macd
from abupy.MarketBu.ABuMarket import all_symbol
from abupy.UtilBu import ABuFileUtil
from abupy.UtilBu.ABuFileUtil import load_df_csv

ABuEnv.g_market_target = abupy.EMarketTargetType.E_MARKET_TARGET_CN

now_df = None
can_buy_df = pd.DataFrame(columns=['symbol', 'price', 'type'])
all_symbols = []


def fetch_all_now():
    global now_df
    while(True):
        now_df = ts.get_today_all()
        if now_df is None or len(now_df) == 0:
            time.sleep(10)
            continue
        return now_df
    # now_df = load_df_csv('/Users/juchen/PycharmProjects/stock/data/day_all/2018-05-08.csv')


def parse_one_stock(his_df, symbol):


    global can_buy_df
    all_close = his_df.close.copy()
    stock_now = now_df[now_df.code == int(symbol)]
    if stock_now is None:
        return

    now_price_s = stock_now.trade.astype(float)
    all_close.add(now_price_s)
    if len(stock_now["trade"].values) == 0:
        logging.error("today price null, %s" % symbol)
        return

    now_price = stock_now["trade"].values.tolist()[0]
    upper, middle, lower = calc_boll(all_close, 20, 2)
    dif, dea, bar = calc_macd(all_close)

    if his_df.close[len(his_df.close) - 1] < middle[len(middle) - 2] \
            and now_price > middle[len(middle) - 1]:
        df_temp = pd.DataFrame([[str(symbol), now_price, 'boll']], columns=['symbol', 'price', 'type'])
        can_buy_df = can_buy_df.append(df_temp)
    if dif[len(dif) - 2] < dea[len(dea) - 2] \
            and dif[len(dif) - 1] >= dea[len(dea) - 1]:
        df_temp = pd.DataFrame([[str(symbol), now_price, 'macd']], columns=['symbol', 'price', 'type'])
        can_buy_df = can_buy_df.append(df_temp)


def for_each_dir():
    for sfile in os.listdir(ABuEnv.g_project_kl_df_data_csv):
        fpath = os.path.join(ABuEnv.g_project_kl_df_data_csv, sfile)
        if os.path.isfile(fpath):
            his_df = load_df_csv(fpath)
            if his_df is None or len(his_df) == 0:
                logging.info("%s is empty" % fpath)
                continue
            if is_stock(sfile) is False:
                continue

            parse_one_stock(his_df,  parse_symbol(sfile))


def parse_symbol(fname):
    return fname.split('_')[0][2:]


def is_stock(fname):
    global all_symbols
    if fname.split('_')[0] in all_symbols:
        return True
    else:
        return False


def store_today_can_buy_stock():
    global can_buy_df
    base_dir = 'out_put'
    # 时间字符串
    date_dir = datetime.datetime.now().strftime("%Y_%m_%d")
    fn = os.path.join(ABuEnv.g_project_data_dir, base_dir, date_dir, 'today_actions.csv')
    ABuFileUtil.ensure_dir(fn)
    ABuFileUtil.dump_df_csv(fn, can_buy_df)

if __name__ == "__main__":
    global all_symbols
    all_symbols = all_symbol()
    fetch_all_now()
    # print(int('000001'))
    # stock_now = now_df[now_df.code == int('000001')]
    # now_price = float(stock_now.trade)
    # print(now_price)
    for_each_dir()
    store_today_can_buy_stock()
