import tushare as ts
import pandas as pd
import os
from datacenter import *
import logging

logger = logging.getLogger(__name__)

datapath = DataCenter.get_datapath()

def get_daily_price(codelist, startdate, enddate):
    pxall = pd.DataFrame()
    for i, code in enumerate(codelist):
        logger.info("Downloading price for %s (%i/%i)", code, i+1, len(codelist))
        px = ts.get_k_data(code, start=startdate, end=enddate) #autype='hfq'
        if px.empty:
            logger.warning('Empty data loaded')
        else:
            pxall = pxall.append(px)
    return pxall

def get_index_composition(index):
    if index == 'sz50':
        filename =  os.path.join(datapath['univ'], 'sz50.csv')
    elif index == 'hs300':
        filename = os.path.join(datapath['univ'], 'hs300.csv')
    elif index == 'zz500':
        filename = os.path.join(datapath['univ'], 'zz500.csv')
    return pd.read_csv(filename, dtype={'code': str})

def get_code_list(index_list):
    dat = pd.DataFrame()
    for index in index_list:
        dat = dat.append(get_index_composition(index))
    codelist = list(set(dat['code']))
    codelist += ['sh', 'sz', 'hs300']
    return codelist

def get_prevclose(df):
    df['prevclose'] = df.close.shift(1)
    return df

def rolling_operation(func, windowlen, col, newcol):
    def f(df):
        df[newcol] = func(arg=df[col], window=windowlen, min_periods=windowlen).tolist()
        return df
    return f

def enrich(pxorig):
    logger.info('Enriching price data')
    startdate = min(pxorig.date)
    dates = DataCenter.get_business_days_within(startdate, 60, 0)
    dc = DataCenter(dates[0], startdate)
    dc.price = dc.price[dc.price['date'] < startdate]
    univ = pd.concat([dc.univ_dict['sz50'][['code','name']],
                      dc.univ_dict['hs300'][['code','name']],
                      dc.univ_dict['zz500'][['code','name']]], ignore_index=True)
    px = pd.merge(pxorig, univ.drop_duplicates(), on='code', how='left')
    logger.info('Enriching return')
    px = px.groupby(px.code).apply(get_prevclose)
    prevpx = dc.price[dc.price.date==max(dc.price.date)][['code','close']]
    pxtmp = pd.merge(px[px.date==startdate][['code']], prevpx, on='code', how='left')
    px.loc[px.date==startdate, 'prevclose'] = pxtmp['close'].tolist()
    px['return'] = (px['close'] - px['prevclose']) / px['prevclose']
    logger.info('Enriching high/low')
    px.date = pd.to_datetime(px.date, format='%Y-%m-%d')
    pxmerged = pd.concat([dc.price, px], ignore_index=True)
    pxmerged = pxmerged.groupby(pxmerged.code).apply(rolling_operation(pd.rolling_max, 21, 'high', 'high21'))
    pxmerged = pxmerged.groupby(pxmerged.code).apply(rolling_operation(pd.rolling_min, 21, 'low', 'low21'))
    pxmerged = pxmerged.groupby(pxmerged.code).apply(rolling_operation(pd.rolling_max, 60, 'high', 'high60'))
    pxmerged = pxmerged.groupby(pxmerged.code).apply(rolling_operation(pd.rolling_min, 60, 'low', 'low60'))
    logger.info('Enriching volatility')
    pxmerged = pxmerged.groupby(pxmerged.code).apply(rolling_operation(pd.rolling_std, 21, 'return', 'volatility21'))
    pxmerged = pxmerged.groupby(pxmerged.code).apply(rolling_operation(pd.rolling_max, 60, 'return', 'volatility60'))
    logger.info('Enriching median volume/turnover')
    pxmerged['vwap'] = 0.25 * (pxmerged['open']+pxmerged['close']+pxmerged['high']+pxmerged['low'])
    pxmerged['turnover'] = 100 * pxmerged['vwap'] * pxmerged['volume']
    pxmerged = pxmerged.groupby(pxmerged.code).apply(rolling_operation(pd.rolling_median, 21, 'volume', 'med21volume'))
    pxmerged = pxmerged.groupby(pxmerged.code).apply(rolling_operation(pd.rolling_median, 21, 'turnover', 'med21turnover'))
    pxmerged = pxmerged.groupby(pxmerged.code).apply(rolling_operation(pd.rolling_median, 60, 'volume', 'med60volume'))
    pxmerged = pxmerged.groupby(pxmerged.code).apply(rolling_operation(pd.rolling_median, 60, 'turnover', 'med60turnover'))
    px = pxmerged[pxmerged.date >= startdate][dc.price.columns]
    logger.info('Finish enriching')
    return px

def save_csv_archive(dat):
    dates = list(set(dat['date']))
    dates.sort()
    for date in dates:
        d = dat[dat['date']==date]
        filename = os.path.join(datapath['dailycache'])
        filename = os.path.join(filename, date[:4]+'/'+date[:4]+date[5:7]+date[8:10]+'.csv')
        d = d.drop(['industry','area','concept','sme','gem'], axis=1)
        logger.info('Saving file to %s', filename)
        d.to_csv(filename, index=False)

def save_csv(dat):
    dates = list(set(dat['date']))
    dates.sort()
    for date in dates:
        d = dat[dat['date']==date]
        filename = os.path.join(datapath['dailycache'])
        filename = os.path.join(filename, str(date.year)+'/'+date.strftime('%Y%m%d')+'.csv')
        logger.info('Saving file to %s', filename)
        d.to_csv(filename, index=False)

def get_trading_calender():
    cal = ts.trade_cal()
    cal.columns = ['date', 'isopen']
    cal.date = pd.to_datetime(cal.date, format='%Y/%m/%d', errors='ignore')
    cal.to_csv(os.path.join(datapath['misc'], 'trading_calender.csv'), index=False)

def get_business_days():
    cal = pd.read_csv(os.path.join(datapath['misc'], 'trading_calender.csv'))
    bdays = cal[cal.isopen==1]
    bdays['date'].to_csv(os.path.join(datapath['misc'], 'business_days.csv'), index=False)

def update_business_days():
    logger.info('Updating business days')
    years = os.listdir(datapath['dailycache'])
    dates = []
    for year in years:
        filenames = os.listdir(os.path.join(datapath['dailycache'], year))
        dates += [x[:8] for x in filenames]
    dat = pd.DataFrame({'index':range(len(dates)), 'date':dates})
    dat.to_csv(os.path.join(datapath['misc'], 'business_days.csv'), index=False)
    