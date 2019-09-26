# -*- coding: utf8 -*-
import time
import numpy as np
import pandas as pd

from common import xparse, ALF
from impdata.names import CFN, LLD
import specs

# --- K-Means clustering algorithm ---
def edist(X, Y):
    d = np.empty((X.shape[0], Y.shape[0]), np.float64)
    for k, y in enumerate(Y):
        d[:, k] = ((X - y) ** 2).sum(axis=1)**0.5
    return d

def kmeans(X, centres, delta=.001, maxiter=10):
    N, dim = X.shape
    k, cdim = centres.shape
    if dim != cdim:
        raise ValueError('kmeans: X %s and centres %s must have the same number of columns' % (X.shape, centres.shape))
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = edist(X, centres)  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx, xtoc]
        avdist = distances.mean()  # median ?
        if (1 - delta) * prevdist <= avdist <= prevdist or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centres[jc] = X[c].mean( axis=0 )
    return centres, xtoc, distances

def randomsample(X, n):
    if len(X) < n:
        return X[np.random.choice(xrange(X.shape[0]), size=int(n), replace=True)]
    else:
        return X[np.random.choice(xrange(X.shape[0]), size=int(n), replace=False)]

def kmeanssample(X, k, nsample=100, **kwargs):
    # merge w kmeans ? mttiw
    # v large N: sample N^1/2, N^1/2 of that
    # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max(2*np.sqrt(N), 10*k)
    Xsample = randomsample(X, int(nsample))
    pass1centres = randomsample(X, int(k))
    samplecentres = kmeans(Xsample, pass1centres, **kwargs)[0]
    return kmeans(X, samplecentres, **kwargs)

def build_statistics(df, currdate):

    wa = lambda x: np.average(x, weights = df.loc[x.index, LLD.Current_Debt])

    agg_dict = {
        LLD.Current_Debt            : np.sum,
        LLD.Mortgage_No             : len,
        LLD.Original_Debt           : np.sum,
        LLD.Interest_Percent        : wa,
        LLD.CLTV                    : wa,
        LLD.LTV                     : wa,
        LLD.Seasoning               : wa,
        LLD.Maturity                : wa,
    }

    df['Bucket'] = 'default'

    res = df.groupby(by=['Bucket']).agg(agg_dict)

    # res[LLD.Original_Debt] *= 10**-6
    # res[LLD.Current_Debt] *= 10**-6
    res['Mortgage_Size'] = res[LLD.Current_Debt] / res[LLD.Mortgage_No]

    to_rename = {
        LLD.CLTV: 'CLTV',
        LLD.LTV: 'LTV',
        LLD.Seasoning: 'Seasoning',
        LLD.Maturity: 'Maturity',
        LLD.Interest_Percent: 'Interest_Percent',
        LLD.Mortgage_No: 'Mortgage_No',
        LLD.Current_Debt: 'Current_Debt',
        LLD.Original_Debt: 'Original_Debt',
    }
    res.rename(columns=to_rename, inplace=True)
    res = res[['Current_Debt','Original_Debt','Mortgage_No','Interest_Percent','CLTV','LTV','Seasoning','Maturity','Mortgage_Size']].T

    return res

# Lk normalization
def l_normalize(a, axis=-1, order=1):
    l = np.atleast_1d(np.linalg.norm(np.copy(a), order, axis))
    l[l == 0] = 1
    return a / np.expand_dims(l, axis)

# Min-max as an alternative
def min_max_scale(data):
    # X_scaled = scale * X + 0 - X.min(axis=0) * scale
    # where scale = (1 - 0) / (X.max(axis=0) - X.min(axis=0))

    scale = 1.0 / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0))
    min_ = np.nanmin(data, axis=0)
    return (data - min_) * scale

def normalize(a, axis=-1, order=1):
    return l_normalize(a, axis, order)

def make_clust(df, n_syn):

    # Filter out matured loans so that clustering be done over actual portfolio
    df = df[df[LLD.Current_Debt] > 0].copy()

    # clustering over credit seasoning and maturity
    X0 = normalize(df[[LLD.Seasoning, LLD.Maturity]].values)
    nc0 = 1
    if n_syn is None:
        arr = df[LLD.Seasoning].values + df[LLD.Maturity].values
        nc0 = min(int(1 + (arr.max() - arr.min()) / 60.0), 5)
    elif n_syn > 1.0:
        nc0 = int(n_syn**0.5) + 1
    df['idx1'] = kmeanssample(X0, nc0)[1]

    X1 = normalize(df[[LLD.Interest_Percent]].values)
    nc1 = 1
    if n_syn is None:
        arr = df[LLD.Interest_Percent].values
        nc1 = min(int(1 + (arr.max() - arr.min()) / 3.0), 5)
    elif n_syn > 1:
        nc1 = int(n_syn**0.5) + 1
    df['idx2'] = kmeanssample(X1, nc1)[1]

    def wa_func(x):
        return np.average(x, weights=df.loc[x.index, LLD.Current_Debt])

    # rename
    df[LLD.Total_Debt] = df[LLD.Current_Debt]
    df.rename(columns={LLD.Mortgage_No: LLD.Mortgage_Num}, inplace=True)

    res = df.groupby(by=['idx1', 'idx2']).agg(
        {
            LLD.Mortgage_Num: len,
            LLD.Original_Debt: np.average,
            LLD.Current_Debt: np.average,
            LLD.Total_Debt: np.average,
            LLD.Interest_Percent: wa_func,
            LLD.Seasoning: wa_func,
            # LLD.Maturity: np.average,
            LLD.Maturity: wa_func,
            LLD.Payment: np.average,
            LLD.LTV: wa_func,
            LLD.MIR0: wa_func,
            LLD.Default: wa_func,
            LLD.Prepayment: wa_func,
            LLD.Partial: wa_func,
            LLD.PartSize: wa_func,
            LLD.Indexing: wa_func,
            # LLD.Product: wa_func,
        }
    )
    res.reset_index(inplace=True)
    return res

def run(df, inputs, newseed=None, makeclust=True):

    np.random.seed(newseed)

    if len(df) == 0:
        raise Exception('empty LLD (PortfolioID=%d, InputFileName=%s' % (inputs.PortfolioID, inputs.InputFileName))

    # # join region data to LLD
    # df = df.join(inputs.Datasets.Regions, how='left', on=LLD.Region)
    df.loc[pd.isnull(df[LLD.Region]), LLD.Region] = 77

    # join MIR at the issue
    df['Start_Date'] = df[LLD.Issue_Date].values.astype('datetime64[M]')
    df = df.join(inputs.Datasets.History[[CFN.MIR]].rename(columns={CFN.MIR:LLD.MIR0}), how='left', on='Start_Date')
    df.loc[pd.isnull(df[LLD.MIR0]), LLD.MIR0] = 12.85
    df[LLD.Rating] = df[LLD.MIR0] - df[LLD.Interest_Percent]

    curr_date = inputs.Parameters.ActualLLDMonth
    quar_date = np.datetime64("%04d-%02d-%02d" % (pd.to_datetime(curr_date).year, pd.to_datetime(curr_date).quarter * 3 - 2, 1), 'M')

    # join HPI from issue to current
    if LLD.Indexing not in df.columns:
        y2000 = np.datetime64('2000-01-01', 'M')
        quarter = np.timedelta64(3, 'M')
        df['Property_Quarter'] = y2000 + quarter * ((df[LLD.Issue_Date].values.astype('datetime64[M]') - y2000) / quarter).astype(int)
        df['Period_Quarter'] = y2000 + quarter * ((inputs.Parameters.ActualLLDMonth - y2000) / quarter).astype(int)
        df = df.join(inputs.Datasets.HPIndex[[LLD.Indexing]], on=[LLD.Region, 'Property_Quarter', 'Period_Quarter'], how='left')

    # join IncomeToLoan data
    df[LLD.IncomeToLoan] = df[LLD.Income] / df[LLD.Original_Debt]
    df.loc[pd.isnull(df[LLD.IncomeToLoan]), LLD.IncomeToLoan] = 0.035

    # join IncomeToLoan data
    df.loc[pd.isnull(df[LLD.Education]), LLD.Education] = 3

    # --- apply simple lld scoring ---
    models = ['Default', 'Prepayment', 'Partial', 'PartSize']
    scores = [LLD.Default, LLD.Prepayment, LLD.Partial, LLD.PartSize]

    models = {
        'Default':      LLD.Default,
        'Prepayment':   LLD.Prepayment,
        'Partial':      LLD.Partial,
        'PartSize':     LLD.PartSize
    }

    # new type of adjustment applying
    for col in models.keys():
        df[models[col]] = 1.0

    # # --- for model debug ---
    # df[LLD.Product] = 0

    for product in df[LLD.Product].unique():

        # select only current product
        L0 = df[LLD.Product]==product

        # get adjustments constant rules
        if product in specs.adjust_dict.keys():
            adjusts = specs.adjust_dict[product].const_adjusts
        else:
            print time.strftime('%Y-%m-%d %H:%M:%S'), 'Model does not support product type = "%s" (product was set to default type).' % product
            adjusts = specs.adjust_dict[0].const_adjusts

        # apply adjustments to selected data
        for key in adjusts:

            expr, kwargs = xparse(key)
            if expr is None:
                cov = df.loc[L0, key].values
            else:
                for i in range(len(kwargs)):
                    # Note: This only works in Python 2.x.
                    locals()[ALF[i]] = df.loc[L0, kwargs[ALF[i]]].values
                cov = eval(expr)

            for col in adjusts[key].columns[1:]:
                df.loc[L0, models[col]] *= np.interp(cov, adjusts[key]['Value'].values, adjusts[key][col].values)
                # --- for model debug ---
                # if col == 'Default':
                #     if key+'_adj' not in df.columns:
                #         df[key+'_adj'] = 0.0
                #     df.loc[L0, key+'_adj'] = np.interp(cov, adjusts[key]['Value'].values, adjusts[key][col].values)

    # --- for model debug ---
    # df.to_csv(r'\\vfs02.ahml1.ru\CFM\WORK_DIR\Queue\Service\dmp\temp_lld.csv', sep='\t', index=False, encoding='utf8')

    currdate = inputs.Parameters.ActualLLDMonth
    # --- Seasoning ---
    df[LLD.Seasoning] = (currdate - df[LLD.Issue_Date].values.astype('datetime64[M]')) / np.timedelta64(1, 'M')
    # check if Seasoning <= 0
    df.loc[df[LLD.Seasoning] <= 0.0, LLD.Seasoning] = 1.0
    # --- Maturity ---
    df[LLD.Maturity] = (df[LLD.Maturity_Date].values.astype('datetime64[M]') - currdate) / np.timedelta64(1, 'M')
    # check if Maturity <= 0
    df.loc[df[LLD.Maturity] <= 0.0, LLD.Maturity] = 1.0

    df[LLD.CLTV] = df[LLD.LTV] * df[LLD.Current_Debt] / df[LLD.Original_Debt]

    # payment function
    def payment(debt, rate, lmat):
        i = rate / 1200.0
        R = (1.0 + i) ** lmat
        K = i * R / (R - 1.0)
        return debt * K

    df[LLD.Payment] = payment(df[LLD.Current_Debt].values, df[LLD.Interest_Percent].values, df[LLD.Maturity].values)

    stats = build_statistics(df, currdate)

    # --- multiproduct support ---
    if (inputs.Parameters.NumberOfSyntheticLoans is not None) and (inputs.Parameters.NumberOfSyntheticLoans > 0):
        res = []
        for product in df[LLD.Product].unique():
            temp = make_clust(df.loc[df[LLD.Product].values == product, :].copy(deep=True), inputs.Parameters.NumberOfSyntheticLoans)
            temp[LLD.Product] = int(product)
            res.append(temp)
        res = pd.concat(res)
    else:
        res = df
        res[LLD.Mortgage_Num] = 1.0
        res[LLD.Total_Debt] = res[LLD.Original_Debt]

        # fix - check that Product is integer
        res.rename(columns={LLD.Product:'temp'}, inplace=True)
        res[LLD.Product] = res['temp'].values.astype(int)
        del res['temp']

    return res[[
        LLD.Mortgage_Num,
        LLD.Total_Debt,
        LLD.Original_Debt,
        LLD.Current_Debt,
        LLD.Interest_Percent,
        LLD.Seasoning,
        LLD.Maturity,
        LLD.Payment,
        LLD.LTV,
        LLD.MIR0,
        LLD.Default,
        LLD.Prepayment,
        LLD.Partial,
        LLD.PartSize,
        LLD.Indexing,
        LLD.Product,
    ]].copy(deep=True), stats
