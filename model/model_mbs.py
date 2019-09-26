# -*- coding: utf8 -*-
import numpy as np
import pandas as pd

# utility subsystems
from common import profiling
from common import debugging

from impdata.names import CFN
from scipy.optimize import minimize

def mbs_npv(cf, spread=0.0, useirr=False, weighted=False, usemkt=True):

    if usemkt:
        df_name = CFN.ZCY
    else:
        df_name = CFN.SCR

    if useirr:
        if weighted:
            return (cf['Period'] * cf['CF'] * (1.0 + float(spread)/10000.0)**-cf['Period']).sum(axis=0, level=0)
        else:
            return (cf['CF'] * (1.0 + float(spread)/10000.0)**-cf['Period']).sum(axis=0, level=0)
    else:
        if weighted:
            return (cf['Period'] * cf['CF'] * (1.0 + cf[df_name] / 100.0 + float(spread)/10000.0)**-cf['Period']).sum(axis=0, level=0)
        else:
            return (cf['CF'] * (1.0 + cf[df_name] / 100.0 + float(spread)/10000.0)**-cf['Period']).sum(axis=0, level=0)

def mbs_rate(cf, fv0, maxscr):

    res = np.zeros((maxscr, ))

    scr_list = cf.index.levels[0].unique()
    for i in range(maxscr):
        j = scr_list[i]
        period = cf.loc[pd.IndexSlice[j, :], 'Period'].values
        cashflow = cf.loc[pd.IndexSlice[j, :] ,'CF'].values
        res[i] = minimize(lambda x, S, cashflow, period: (S - np.sum(cashflow * (1.0 + x / 100.0) ** -period))**2, 0.0, (fv0, cashflow, period)).x[0]

    return res

def fmin((fv0, cashflow, period)):
    return minimize(lambda x, S, cashflow, period: (S - np.sum(cashflow * (1.0 + x / 100.0) ** -period)) ** 2, 0.0, (fv0, cashflow, period)).x[0]

def mbs_irr(cf, fv0=0.0):
    return minimize(lambda x, S, cashflow: (mbs_npv(cashflow, spread=x*100.0, useirr=True).mean() - S)**2, 0.0, (fv0, cf)).x[0]

def mbs_durmac(cf, irr=0.0):
    return mbs_npv(cf, spread=irr*100.0, useirr=True, weighted=True).mean() / mbs_npv(cf, spread=irr*100.0, useirr=True).mean()

def mbs_durmod(cf, spread=0.0, delta=1):
    dx = 0.0001 * float(delta)
    PV0 = mbs_npv(cf, spread=spread).mean()
    PV1 = mbs_npv(cf, spread=spread + delta).mean()
    return -(PV1 - PV0) / (dx * PV0)

def mbs_dureff(PV0, PVm, PVp, dx = 0.01):
    return -0.5 * (PVp - PVm) / (dx * PV0)

def mbs_convex(PV0, PVm, PVp, dx = 0.01):
    return (PVp - 2.0*PV0 + PVm) / (PV0 * dx**2)

def mbs_simple_stats(cf, s0, spread):

    # find irr
    irr = mbs_irr(cf, fv0=s0)

    # find Macoley duration
    mac = mbs_durmac(cf, irr=irr)

    # find modified duration
    mod = mbs_durmod(cf, spread)

    return np.array([irr, mac, mod])

def mbs_complex_stats(cfc, cfm, cfp, spread, delta=100):

    dx = 0.0001 * float(delta)

    PV0 = mbs_npv(cfc, spread=spread, usemkt=False).mean()
    PVm = mbs_npv(cfm, spread=spread, usemkt=False).mean()
    PVp = mbs_npv(cfp, spread=spread, usemkt=False).mean()

    # find effective duration
    eff = mbs_dureff(PV0, PVm, PVp, dx)

    # find convexity
    con = mbs_convex(PV0, PVm, PVp, dx)

    return np.array([eff, con])

def mbs_cpr(cf):

    debt = (cf['Debt'] + cf['Depreciation']).loc[cf[CFN.CPR] > 0.0, :]
    cpr = cf[CFN.CPR].loc[cf[CFN.CPR] > 0.0, :]

    return ((cpr * debt).sum(axis=0, level=0) / debt.sum(axis=0, level=0)).mean()

def mbs_cpr_1i(cf):

    debt = (cf['Debt'] + cf['Depreciation']).loc[cf[CFN.CPR] > 0.0]
    cpr = cf[CFN.CPR].loc[cf[CFN.CPR] > 0.0]

    a_cpr = (cpr * debt).sum() / debt.sum()

    if np.isnan(a_cpr):
        a_cpr = 0.0

    return a_cpr

def recover_cashflow(df, hst_cf, lldmon, recmon, recday, first_coupon=False):

    if isinstance(df.index, pd.core.index.MultiIndex):
        S = len(df.index.levels[0]) # S = par.NumberOfMonteCarloScenarios
        M = len(df.index.levels[1]) # number of cashflow months
    else:
        S = 1 # S = par.NumberOfMonteCarloScenarios
        M = len(df.index) # number of cashflow months

    # allocate memory
    BOP = np.zeros((M, S))
    EOP = np.zeros((M, S))
    YLD = np.zeros((M, S))
    AMT = np.zeros((M, S))
    ADV = np.zeros((M, S))
    DAT = np.zeros(BOP.shape, dtype='datetime64[M]')

    # fill data
    BOP[:] = df[CFN.DBT].values.reshape((S, -1,)).T
    EOP[:] = df[CFN.DBT].values.reshape((S, -1,)).T
    YLD[:] = df[CFN.YLD].values.reshape((S, -1,)).T
    AMT[:] = df[CFN.AMT].values.reshape((S, -1,)).T
    # ADV[:] = (df[CFN.FLL]+df[CFN.PRT]+df[CFN.PDL]).values.reshape((S, -1,)).T
    ADV[:] = df[CFN.ADV].values.reshape((S, -1,)).T

    # reindex dates (the cashflows always start from the LLD-month,
    # but the index in res_c starts from the evaluation month):
    for i in range(BOP.shape[0]):
        DAT[i] = lldmon + i * np.timedelta64(1, 'M')
        EOP[i] = BOP[i] - AMT[i]

    # recover cashflows from history or by interpolation (if recovering is necessary):
    if lldmon > recmon:

        # --- find how many months should be added ---
        R = int((lldmon - recmon) / np.timedelta64(1, 'M'))

        BOP = np.vstack([np.full([R, S], np.nan), BOP])
        EOP = np.vstack([np.full([R, S], np.nan), EOP])
        YLD = np.vstack([np.full([R, S], np.nan), YLD])
        AMT = np.vstack([np.full([R, S], np.nan), AMT])
        ADV = np.vstack([np.full([R, S], np.nan), ADV])
        DAT = np.vstack([np.full([R, S], np.nan, dtype='datetime64[M]'), DAT])

        # --- start recovering recursively:                                                ---
        # --- step 1. check if there is data -- if yes -> recover, if no -> step 2;        ---
        # --- step 2. recover by adding the difference between the two last observations;  ---
        for i in range(R):
            DAT[R-i-1] = recmon + (R-i-1) * np.timedelta64(1, 'M')
            if DAT[R-i-1][0] in hst_cf.index.values:
                BOP[R-i-1] = hst_cf.BOP[DAT[R-i-1][0]]
                YLD[R-i-1] = hst_cf.YLD[DAT[R-i-1][0]]
                AMT[R-i-1] = hst_cf.AMT[DAT[R-i-1][0]]
                ADV[R-i-1] = ADV[R-i] * BOP[R-i-1] / BOP[R-i]
                EOP[R-i-1] = BOP[R-i-1] - AMT[R-i-1]
            else:
                BOP[R-i-1] = 2*BOP[R-i] - BOP[R-i+1]
                YLD[R-i-1] = YLD[R-i] * BOP[R-i-1] / BOP[R-i]
                AMT[R-i-1] = BOP[R-i-1] - BOP[R-i]
                ADV[R-i-1] = ADV[R-i] * BOP[R-i-1] / BOP[R-i]
                EOP[R-i-1] = BOP[R-i-1] - AMT[R-i-1]

    return DAT, BOP, EOP, YLD, AMT, ADV

def find_coupon_period(par):

    cpnmon = par.FirstCouponMonth
    issmon = par.IssuedMonth

    cpnday = par.FirstCouponDate - par.FirstCouponMonth

    # --- find next coupon payment date ---
    nxtcpn = cpnmon
    while nxtcpn < par.EvaluationMonth:
        nxtcpn += par.CouponPeriod

    if (nxtcpn == par.EvaluationMonth) and (nxtcpn + cpnday <= par.EvaluationDate):
        nxtcpn += par.CouponPeriod

    # --- find minimum recuired lld month & recover missed cashflow ---
    if nxtcpn == cpnmon:
        prvcpn = issmon
    else:
        prvcpn = nxtcpn - par.CouponPeriod

    return nxtcpn, prvcpn

def mbs_cf_fixed(params, fct_cf, hst_cf, sc=None):
    #  1) --- get all of the necessary nominal values and coupon day ---
    if (params.Nominal is None) or (params.Nominal == 0.0):
        Nominal = 1000.0
    else:
        Nominal = params.Nominal

    if (params.CurrentNominal is None) or (params.CurrentNominal == 0.0):
        CurrentNominal = Nominal
    else:
        CurrentNominal = params.CurrentNominal

    CurrentAmount = CurrentNominal * params.StartIssueAmount / Nominal
    cpnday = params.FirstCouponDate - params.FirstCouponMonth

    # 2) --- find next coupon payment month ---
    next_cpn_mon = params.FirstCouponMonth
    while next_cpn_mon < params.EvaluationMonth:
        next_cpn_mon += params.CouponPeriod

    if (next_cpn_mon == params.EvaluationMonth) and (next_cpn_mon + cpnday <= params.EvaluationDate):
        next_cpn_mon += params.CouponPeriod

    nearest_cpn_mon = next_cpn_mon.copy() # save it under another name for the future
    last_coupon_mon = nearest_cpn_mon - params.CouponPeriod

    # 3) --- recover past missing pool's cashflows needed for the calculation of the nearest bond's CF ---
    if next_cpn_mon == params.FirstCouponMonth:
        DT0, BOP, EOP, YLD, AMT, ADV = recover_cashflow(fct_cf, hst_cf, lldmon=params.ActualLLDMonth, recmon=params.IssuedMonth,recday=params.IssuedDate - params.IssuedMonth + 1, first_coupon = True)
    else:
        DT0, BOP, EOP, YLD, AMT, ADV = recover_cashflow(fct_cf, hst_cf, lldmon=params.ActualLLDMonth, recmon=last_coupon_mon, recday=np.timedelta64(0, 'D'))

    # 4) --- create an array of scenarios' indeces ---
    SCR = np.zeros((BOP.shape[0]+1, BOP.shape[1]), dtype=int)
    SCR[:, :] = np.array(range(BOP.shape[1]))

    # 5) --- remake dates array s.t. the dates now show to which coupon the pool's cashflows belong to ---
    for i in range(DT0.shape[0]):
        if DT0[i][0] < next_cpn_mon:
            DT0[i] = next_cpn_mon
        else:
            next_cpn_mon += params.CouponPeriod
            DT0[i] = next_cpn_mon

    # 5.1) Now, there may be the case that LLD month is in the very remote past from the evaluation date.
    #      This can happen in two cases:
    #      --- either LLD was not updated for a long time
    #      --- OR the forward estimation is being implemented.
    # Such situation will result in the current pay period (расчетный период) that is longer than params.CouponPeriod.
    # To avoid it, we need to cut all series such that the length of the current pay period equals params.CouponPeriod:

    cut = (DT0 == DT0[0]).sum() - int(params.CouponPeriod / np.timedelta64(1, 'M'))
    if cut > 0:
        DT0, BOP, EOP, YLD, AMT, ADV, SCR = DT0[cut:], BOP[cut:], EOP[cut:], YLD[cut:], AMT[cut:], ADV[cut:], SCR[cut:]

    # 6) --- create an auxiliary dataframe of scenario- & date- multi-indexed pool's cashflows  ---
    DT0 = np.vstack([np.full(BOP.shape[1], params.EvaluationDate) - cpnday, DT0])
    ADV = np.vstack([np.full(BOP.shape[1], np.nan), ADV])
    EOP = np.vstack([np.full(BOP.shape[1], CurrentAmount), EOP])

    aux = pd.DataFrame({CFN.SCN: SCR.T.ravel(),
                        CFN.DAT: DT0.astype('datetime64[D]').T.ravel() + cpnday,
                        'ADV': ADV.T.ravel(),
                        'NOM': EOP.T.ravel()}).set_index([CFN.SCN, CFN.DAT])

    # 7) --- create a table that consists of accumulated depreciation, accumulated ---
    #    --- advance amortization and bond nominals at dates of coupon payments    ---
    res = aux[['ADV']].groupby(level=[0,1]).sum().join(aux.NOM.groupby(level=[0,1]).tail(1))
    res['DEP'] = res.NOM.groupby(level=0).diff(periods=-1).shift(1).fillna(0)

    # 8) --- check for clean-up condition  ---
    clean_up_value = params.CleanUpCall * 0.01 * params.StartIssueAmount
    not_clean_up = (res.NOM > clean_up_value).astype(int).replace(0, np.nan).groupby(level=0).fillna(1, limit=1).fillna(0).astype(bool)
    res[['NOM','DEP']] = res[['NOM','DEP']].mask(~not_clean_up, np.nan)

    # 8a) --- create clean-up as a separate column...
    res['CLN'] = 0.0

    # 9) --- make the last bond amortization equal to the last nominal (clean-up is made)  ---
    def clean_up(scenario):
        dep = scenario.DEP.fillna(scenario.NOM.min(), limit=1)
        # 9a) ...equal to depreciation ---
        scenario.CLN = dep - scenario.DEP.fillna(0.0, limit=1)
        scenario.DEP = dep
        return scenario
    res = res.groupby(level=0).apply(clean_up)

    # 10) --- convert everything irrelevant to nans ---
    res[['ADV']] = res[['ADV']].mask(~res.DEP.notnull(), np.nan)
    res.fillna(0.0, inplace=True)

    # 11) --- add time-variables and pay special attention to the length of the current coupon period ---
    res = res.reset_index()
    res['Period'] = (res[CFN.DAT].values - params.EvaluationDate) / np.timedelta64(365, 'D')
    res['Delta'] = res[CFN.DAT].groupby(res[CFN.SCN]).diff() / np.timedelta64(365, 'D')

    if res[CFN.DAT][1] == params.FirstCouponDate:
        res.loc[res[res[CFN.DAT]==res[CFN.DAT][1]].index, 'Delta'] = (res[CFN.DAT][1] - params.IssuedDate) / np.timedelta64(365, 'D')
    else:
        previous_cpn_date = nearest_cpn_mon - params.CouponPeriod + cpnday
        res.loc[res[res[CFN.DAT]==res[CFN.DAT][1]].index, 'Delta'] = (res[CFN.DAT][1] - previous_cpn_date) / np.timedelta64(365, 'D')

    # 12) --- define fixed coupon  ---
    res = res.set_index([CFN.SCN,CFN.DAT])
    res['CPN'] = res.NOM.groupby(level=0).apply(lambda x: x.shift(+1)) * res.Delta * params.CouponPercentage * 0.01
    res['CF'] = res.DEP + res.CPN
    res = res.reset_index()

    # 16) --- construct resulting dataframe ---
    res = res[[CFN.SCN, CFN.DAT, 'Period', 'Delta', 'NOM', 'CF', 'CPN', 'DEP', 'ADV', 'CLN']]
    res.columns = [CFN.SCN, CFN.DAT, 'Period', 'Delta', 'Debt', 'CF', 'Coupon', 'Depreciation', 'Advance', 'Cleanup']

    # 17) --- add macro-variables
    if sc is not None:
        macro = sc[[CFN.ZCY, CFN.SCR, CFN.SPT, CFN.MS6, CFN.MIR, CFN.HPI, CFN.HPR]].reset_index().copy(deep=True)

        # --- before interpolating take into account that macro variables all start from the evaluation date, not the first day of the month:
        shiftdelta = (params.EvaluationDate - np.datetime64(macro[CFN.DAT].min(), 'D'))
        macro[CFN.DAT] = macro[CFN.DAT].values.astype('datetime64[D]') + shiftdelta

        def interpolate(x):
            coupon_dates = pd.to_numeric(x[CFN.DAT]).values
            macro_scenario = macro[macro[CFN.SCN] == x[CFN.SCN].values[0]]
            macro_dates = pd.to_numeric(macro_scenario[CFN.DAT]).values

            y = x.copy(deep=True)
            for variable in [CFN.ZCY, CFN.SCR, CFN.SPT, CFN.MS6, CFN.MIR, CFN.HPI, CFN.HPR]:
                y[variable] = np.interp(coupon_dates, macro_dates, macro_scenario[variable].values)

            return y

        res = res.groupby(CFN.SCN).apply(interpolate)

        res[CFN.HPI] *= 100.0
        res[CFN.HPR] = (res[CFN.HPR] - 1.0) * 100.0

    # 18) --- some final stuff
    res.set_index([CFN.SCN, CFN.DAT], inplace=True)
    res.sort_index(inplace=True)

    res[CFN.CPR] = (res['Advance'] / (res['Depreciation'] + res['Debt'])) / res['Delta'] * 100
    res[CFN.INT] = (res['Coupon'] / (res['Depreciation'] + res['Debt'])) / res['Delta'] * 100

    res.fillna(0.0, inplace=True)

    return res

def mbs_cf_floating(params, fct_cf, hst_cf, sc=None):
    #  1) --- get all of the necessary nominal values and coupon day ---
    if (params.Nominal is None) or (params.Nominal == 0.0):
        Nominal = 1000.0
    else:
        Nominal = params.Nominal

    if (params.CurrentNominal is None) or (params.CurrentNominal == 0.0):
        CurrentNominal = Nominal
    else:
        CurrentNominal = params.CurrentNominal

    CurrentAmount = (CurrentNominal / Nominal) * params.StartIssueAmount
    cpnday = params.FirstCouponDate - params.FirstCouponMonth

    # 2) --- find next coupon payment month ---
    next_cpn_mon = params.FirstCouponMonth
    while next_cpn_mon < params.EvaluationMonth:
        next_cpn_mon += params.CouponPeriod

    if (next_cpn_mon == params.EvaluationMonth) and (next_cpn_mon + cpnday <= params.EvaluationDate):
        next_cpn_mon += params.CouponPeriod

    nearest_cpn_mon = next_cpn_mon.copy()  # save it under another name for the future
    last_coupon_mon = nearest_cpn_mon - params.CouponPeriod

    # 3) --- recover past missing pool's cashflows needed for the calculation of the nearest bond's CF ---
    if next_cpn_mon == params.FirstCouponMonth:
        DT0, BOP, EOP, YLD, AMT, ADV = recover_cashflow(fct_cf, hst_cf, lldmon=params.ActualLLDMonth, recmon=params.IssuedMonth, recday=params.IssuedDate - params.IssuedMonth + 1, first_coupon=True)
    else:
        DT0, BOP, EOP, YLD, AMT, ADV = recover_cashflow(fct_cf, hst_cf, lldmon=params.ActualLLDMonth, recmon=last_coupon_mon, recday=np.timedelta64(0, 'D'))

    # 4) --- create an array of scenarios' indices ---
    SCR = np.zeros((BOP.shape[0]+1, BOP.shape[1]), dtype=int)
    SCR[:, :] = np.array(range(BOP.shape[1]))

    # 5) --- remake dates array s.t. the dates now show to which coupon the pool's cashflows belong to ---
    for i in range(DT0.shape[0]):
        if DT0[i][0] < next_cpn_mon:
            DT0[i] = next_cpn_mon
        else:
            next_cpn_mon += params.CouponPeriod
            DT0[i] = next_cpn_mon

    # 5.1) Now, there may be the case that LLD month is in the very remote past from the evaluation date.
    #      This can happen in two cases:
    #      --- either LLD was not updated for a long time
    #      --- OR the forward estimation is being implemented.
    # Such situation will result in the current pay period (расчетный период) that is longer than params.CouponPeriod.
    # To avoid it, we need to cut all series such that the length of the current pay period equals params.CouponPeriod:

    cut = (DT0 == DT0[0]).sum() - int(params.CouponPeriod / np.timedelta64(1, 'M'))
    if cut > 0:

        # A very important moment fot the bonds with a floating coupon:
        # Следующие две строчки добавлены в начале сентября 2019 года по результатам анализа оценок для готовящегося
        # выпуска однотраншевой ИЦБ с ежемесячным купоном и плавающей ставкой (ВТБ-4). Возникла проблема, что первый
        # купон по бумаге в расчетах выходил отрицательным, так как величина полученных процентов оказывалась меньше,
        # чем сервисный сбор (включающий крупный единовременный платеж NonRecurringExpenses). По резальтатам общения с
        # коллегами из секьюритизации и анализа проспекта проснилась следующая картина:
        #    -- дата LLD             --> 2019-07-01;
        #    -- дата выкупа пула     --> 2019-07-11;
        #    -- дата оценки          --> 2019-09-05;
        #    -- дата выпуска ИЦБ     --> 2019-09-28;
        #    -- дата первого купона  --> 2019-11-28;
        # В данной ситуации model_mbs должна сформировать первый денежный поток по следующей схеме:
        # 1) Амортизация = StartIssueAmount - значение DBT (ООД) на 2019-11-01;
        # 2) Купон первого потока:
        #
        #    -- 2.1) С точки зрения реальности:
        #            Купон = [Процентные поступления за июль, август, сентябрь, октябрь] - сервисные сборы -
        #                    - величина накопленных, но не выплаченных процентов;
        #    -- 2.2) С точки зрения того, как эта реальность отражена в model_mbs:
        #            Купон = [YLD за июль, август, сентябрь, октябрь] -
        #                    - (DBT (ООД) на 2019-11-01 * VariableExpenses * 0.01 + ConstantExpenses) / 12 -
        #                    - NonRecurringExpenses;
        #
        # ВАЖНО: В NonRecurringExpenses нужно включать величину накопленных, но не выплаченных процентов!
        # Таким образом, следующие две строчки введены для того, чтобы включить все нужные проценты
        # [YLD за июль, август, сентябрь, октябрь] внутрь расчетного периода первого денежного потока.

        # Обратить внимание! Такая схема работает только в том случае, если месяц выкупа пула равен месяцу LLD.
        # В том случае, если месяц LLD будет позже, чем месяц выкупа, то схема не досчитает процентных поступлений.
        # Такую ситуацию нужно будет продумать в будущем.

        if nearest_cpn_mon == params.FirstCouponMonth:
            YLD[cut] += YLD[:cut].sum()

        DT0, BOP, EOP, YLD, AMT, ADV, SCR = DT0[cut:], BOP[cut:], EOP[cut:], YLD[cut:], AMT[cut:], ADV[cut:], SCR[cut:]

    # 6) --- create an auxiliary dataframe of scenario- & date- multi-indexed pool's cashflows  ---
    DT0 = np.vstack([np.full(BOP.shape[1], params.EvaluationDate) - cpnday, DT0])
    YLD = np.vstack([np.full(BOP.shape[1], np.nan), YLD])
    ADV = np.vstack([np.full(BOP.shape[1], np.nan), ADV])
    EOP = np.vstack([np.full(BOP.shape[1], CurrentAmount), EOP])

    aux = pd.DataFrame({CFN.SCN: SCR.T.ravel(),
                        CFN.DAT: DT0.astype('datetime64[D]').T.ravel() + cpnday,
                        'PERC': YLD.T.ravel(),
                        'ADV': ADV.T.ravel(),
                        'NOM': EOP.T.ravel()}).set_index([CFN.SCN, CFN.DAT])

    # 7) --- create a table that consists of accumulated percentages, accumulated depreciation, ---
    #    --- accumulated advance amortization and bond nominals at dates of coupon payments     ---
    res = aux[['PERC','ADV']].groupby(level=[0,1]).sum().join(aux.NOM.groupby(level=[0,1]).tail(1))
    res['DEP'] = res.NOM.groupby(level=0).diff(periods=-1).shift(1).fillna(0)

    # 8) --- check for clean-up condition  ---
    clean_up_value = params.CleanUpCall * 0.01 * params.StartIssueAmount
    not_clean_up = (res.NOM > clean_up_value).astype(int).replace(0, np.nan).groupby(level=0).fillna(1, limit=1).fillna(0).astype(bool)
    res[['NOM','DEP']] = res[['NOM','DEP']].mask(~not_clean_up, np.nan)

    # 8a) --- create clean-up as a separate column...
    res['CLN'] = 0.0

    # 9) --- make the last bond amortization equal to the last nominal (clean-up is made)  ---
    def clean_up(scenario):
        dep = scenario.DEP.fillna(scenario.NOM.min(), limit=1)
        # 9a) ...equal to depreciation ---
        scenario.CLN = dep - scenario.DEP.fillna(0.0, limit=1)
        scenario.DEP = dep
        return scenario
    res = res.groupby(level=0).apply(clean_up)

    # 10) --- convert everything irrelevant to nans ---
    res[['PERC', 'ADV']] = res[['PERC', 'ADV']].mask(~res.DEP.notnull(), np.nan)

    # 11) --- prepare advance service payments  ---
    res['SERV'] = (res.NOM * params.VariableExpenses * 0.01 + params.ConstantExpenses) * int(params.CouponPeriod) / 12
    res.loc[res.groupby(level=0).SERV.head(1).index, 'SERV'] = np.nan

    # 12) --- if the next coupon payment is the first one, then add non-recurring expenses  ---
    if res.index.get_level_values(level=1)[1] == params.FirstCouponDate:
        res.loc[res.groupby(level=0).SERV.head(2).index, 'SERV'] += params.NonRecurringExpenses

    # 13) --- fill nans with zeros (very important here since float-nan=nan) ---
    res = res.fillna(0)

    # 14) --- define floating coupon as the diff between percentage flows and service payments  ---
    res['CPN'] = res.PERC - res.SERV
    res['CF'] = res.DEP + res.CPN

    # 15) --- add time-variables and pay special attention to the length of the current coupon period ---
    res = res.reset_index()
    res['Period'] = (res[CFN.DAT].values - params.EvaluationDate) / np.timedelta64(365, 'D')
    res['Delta'] = res[CFN.DAT].groupby(res[CFN.SCN]).diff() / np.timedelta64(365, 'D')

    if res[CFN.DAT][1] == params.FirstCouponDate:
        res.loc[res[res[CFN.DAT]==res[CFN.DAT][1]].index, 'Delta'] = (res[CFN.DAT][1] - params.IssuedDate) / np.timedelta64(365, 'D')
    else:
        previous_cpn_date = nearest_cpn_mon - params.CouponPeriod + cpnday
        res.loc[res[res[CFN.DAT]==res[CFN.DAT][1]].index, 'Delta'] = (res[CFN.DAT][1] - previous_cpn_date) / np.timedelta64(365, 'D')

    # 16) --- construct resulting dataframe ---
    del res['PERC'], res['SERV']
    res = res[[CFN.SCN, CFN.DAT, 'Period', 'Delta', 'NOM', 'CF', 'CPN', 'DEP', 'ADV', 'CLN']]
    res.columns = [CFN.SCN, CFN.DAT, 'Period', 'Delta', 'Debt', 'CF', 'Coupon', 'Depreciation', 'Advance', 'Cleanup']

    # 17) --- add macro-variables
    if sc is not None:
        macro = sc[[CFN.ZCY, CFN.SCR, CFN.SPT, CFN.MS6, CFN.MIR, CFN.HPI, CFN.HPR]].reset_index().copy(deep=True)

        # --- before interpolating take into account that macro variables all start from the evaluation date, not the first day of the month:
        shiftdelta = (params.EvaluationDate - np.datetime64(macro[CFN.DAT].min(), 'D'))
        macro[CFN.DAT] = macro[CFN.DAT].values.astype('datetime64[D]') + shiftdelta

        def interpolate(x):
            coupon_dates = pd.to_numeric(x[CFN.DAT]).values
            macro_scenario = macro[macro[CFN.SCN] == x[CFN.SCN].values[0]]
            macro_dates = pd.to_numeric(macro_scenario[CFN.DAT]).values

            y = x.copy(deep=True)
            for variable in [CFN.ZCY, CFN.SCR, CFN.SPT, CFN.MS6, CFN.MIR, CFN.HPI, CFN.HPR]:
                y[variable] = np.interp(coupon_dates, macro_dates, macro_scenario[variable].values)

            return y

        res = res.groupby(CFN.SCN).apply(interpolate)

        res[CFN.HPI] *= 100.0
        res[CFN.HPR] = (res[CFN.HPR] - 1.0) * 100.0

    # 18) --- some final stuff
    res.set_index([CFN.SCN, CFN.DAT], inplace=True)
    res.sort_index(inplace=True)

    res[CFN.CPR] = (res['Advance'] / (res['Depreciation'] + res['Debt'])) / res['Delta'] * 100
    res[CFN.INT] = (res['Coupon'] / (res['Depreciation'] + res['Debt'])) / res['Delta'] * 100

    res.fillna(0.0, inplace=True)

    return res

def stats(cfc, cfm, cfp, par, draw_hist=True):

    clear_price = 'clear_price'
    dirty_price = 'dirty_price'
    spread_oas = 'spread_oas'
    cpr_name = 'cpr_name'
    irr_name = 'irr_name'
    durmac = 'durmac'
    durmod = 'durmod'
    eff_dur = 'eff_dur'
    convex = 'convex'

    Index0 = [spread_oas, dirty_price, clear_price, irr_name, durmac, durmod, eff_dur, convex, cpr_name]

    # --- get clear price ---
    if par.EvaluationDate <= par.IssuedDate:
        ACI = 0.0
    else:
        # find next coupon date
        evoday = par.EvaluationDate - par.EvaluationMonth
        cpnday = par.FirstCouponDate - par.FirstCouponMonth

        cpnmon = par.FirstCouponMonth
        curmon = par.EvaluationMonth
        while cpnmon < curmon:
            cpnmon += par.CouponPeriod

        if cpnmon == curmon:
            if cpnday <= evoday:
                cpnmon += par.CouponPeriod

        t0 = curmon + evoday
        t1 = cpnmon + cpnday
        curr_period = float((t1 - t0) / np.timedelta64(365, 'D'))
        next_period = cfc.loc[pd.IndexSlice[:, t1], 'Delta'].values
        next_coupon = cfc.loc[pd.IndexSlice[:, t1], 'Coupon'].values

        ACI = 100.0 * (1.0 - curr_period / next_period) * next_coupon / (par.CurrentNominal * par.StartIssueAmount / par.Nominal)

    res = pd.DataFrame(None, index=Index0, columns=['-100bp', 'Average', '+100bp'])

    if par.CalculationType == 'OAS':
        # fix  OAS
        used_spread = par.CalculationValue
    else:
        # find OAS
        used_spread = minimize(lambda x: ((100.0 * mbs_npv(cfc, spread=x, usemkt=False)/(par.CurrentNominal * par.StartIssueAmount / par.Nominal)).mean() - par.CalculationValue) ** 2, 0.0).x[0]

    # find price
    fv = mbs_npv(cfc, spread=used_spread, usemkt=False)

    # # sequential version
    # irr_dist = mbs_rate(cfc, fv.mean(), len(fv))

    prm = 100.0 * fv / ((par.CurrentNominal / par.Nominal) * par.StartIssueAmount)

    res.loc[dirty_price, 'Average'] = prm.mean()
    res.loc[spread_oas, 'Average'] = used_spread
    res.loc[clear_price, 'Average'] = (prm - ACI).mean()

    res.loc[[irr_name, durmac, durmod], 'Average'] = mbs_simple_stats(cfc, (par.CurrentNominal * par.StartIssueAmount / par.Nominal) * prm.mean() / 100.0, used_spread)

    res.loc[cpr_name, 'Average'] = mbs_cpr(cfc)

    if cfm is not None:
        prm1 = 100.0 * mbs_npv(cfm, spread=used_spread, usemkt=False) / (par.CurrentNominal * par.StartIssueAmount / par.Nominal)
        res.loc[dirty_price, '-100bp'] = prm1.mean()
        res.loc[clear_price, '-100bp'] = (prm1 - ACI).mean()
        res.loc[[irr_name, durmac, durmod], '-100bp'] = mbs_simple_stats(cfm, (par.CurrentNominal * par.StartIssueAmount / par.Nominal) * prm1.mean() / 100.0, used_spread)
        res.loc[cpr_name, '-100bp'] = mbs_cpr(cfm)

    if cfp is not None:
        prm2 = 100.0 * mbs_npv(cfp, spread=used_spread, usemkt=False) / (par.CurrentNominal * par.StartIssueAmount / par.Nominal)
        res.loc[dirty_price, '+100bp'] = prm2.mean()
        res.loc[clear_price, '+100bp'] = (prm2 - ACI).mean()
        res.loc[[irr_name, durmac, durmod], '+100bp'] = mbs_simple_stats(cfp, (par.CurrentNominal * par.StartIssueAmount / par.Nominal) * prm2.mean() / 100.0, used_spread)
        res.loc[cpr_name, '+100bp'] = mbs_cpr(cfp)

    if cfm is not None and cfp is not None:
        res.loc[[eff_dur, convex], 'Average'] = mbs_complex_stats(cfc, cfm, cfp, used_spread)

    if draw_hist:
        hmin = np.round(min(prm.min(), (prm - ACI).min()), 2) - 0.01
        hmax = np.round(max(prm.max(), (prm - ACI).max()), 2) + 0.01
        grid = np.linspace(hmin, hmax, 20)

        Dirty = np.bincount(np.digitize(prm.values, bins=grid), minlength=len(grid))[1:].astype(float) / len(prm)
        Clear = np.bincount(np.digitize((prm - ACI).values, bins=grid), minlength=len(grid))[1:].astype(float) / len(prm)

        hf = pd.DataFrame(
            {
                'Premium': (grid[:-1] + grid[1:])/2.0,
                'Clear': Clear,
                'Dirty': Dirty
            }
        )[['Premium', 'Clear', 'Dirty']]
    else:
        hf = None

    return res, hf

# @debugging
@profiling
def run(inputs, cashflows, scenarios, verbose=False, **kwargs):

    result = None

    # --- compute rmbsflow ---
    if inputs.Parameters.CouponType == "Fixed":
        result = mbs_cf_fixed(params=inputs.Parameters, fct_cf=cashflows, hst_cf=inputs.Datasets.Hist_CF, sc=scenarios)
    elif inputs.Parameters.CouponType == "Floating":
        result = mbs_cf_floating(params=inputs.Parameters, fct_cf=cashflows, hst_cf=inputs.Datasets.Hist_CF, sc=scenarios)
    else:
        result = mbs_cf_fixed(params=inputs.Parameters, fct_cf=cashflows, hst_cf=inputs.Datasets.Hist_CF, sc=scenarios)

    return result
