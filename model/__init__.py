# -*- coding: utf8 -*-
import os
import numpy
import pandas

import model_lld as lld
import model_cir as cir
import model_cfm as cfm
import model_mbs as mbs

import common
import specs

from impdata.names import LLD, CFN


def saferun(func):

    def wrapper(self, *args, **kwargs):
        if self.updater.exit_status:
            return
        try:
            frepr = func.__name__ + ' ' * (12 - len(func.__name__))
            print '%s - ...' % frepr,
            func(self, *args, **kwargs)
            print '\r%s - OK' % frepr
        except Exception as e:
            self.updater.send_err(
                description="%s raised: %s" % (func.__name__, str(e)),
                errorcode=getattr(common.Errors, func.__name__+'Error') if hasattr(common.Errors, func.__name__+'Error') else common.Errors.ModelBaseError
            )

    return wrapper

def progress(func):

    def wrapper(self, *args, **kwargs):
        if self.updater.exit_status:
            return None
        if self.updater.show_progress:
            if self.updater.was_ever_stopped:
                self._restart_updater()
            self.updater.start()
            func(self, *args, **kwargs)
            self.updater.stop()
        else:
            func(self, *args, **kwargs)

    return wrapper


def defaultrun(x, inputs, lld_list, seed=None, delta=0, max_years=15, **kwargs):

    scr_df = cir.run(inputs, delta, max_years, seeds=[x[1]], debug=None, verbose=None, timings=None)

    res_df = None
    for lld_df in lld_list:
        for product in lld_df[LLD.Product].unique():

            if product in specs.adjust_dict.keys():
                inputs.Adjusts = specs.adjust_dict[product]
            else:
                print 'Unsupported product "%s" (set to default)' % product
                inputs.Adjusts = specs.adjust_dict[0]

            if res_df is None:
                res_df = cfm.run(inputs, lld_df.loc[lld_df[LLD.Product] == product, :].copy(deep=True), scr_df, seed=seed, debug=None, verbose=True, timings=None)
            else:
                res_df += cfm.run(inputs, lld_df.loc[lld_df[LLD.Product] == product, :].copy(deep=True), scr_df, seed=seed, debug=None, verbose=True, timings=None)

        inputs.Adjusts = None

        mbs_df = mbs.run(inputs, res_df, scr_df, debug=None, verbose=True, timings=None)
        mbs_df.rename(index={0: x[0]}, level=0, inplace=True)
        res_df.rename(index={0: x[0]}, level=0, inplace=True)
        scr_df.rename(index={0: x[0]}, level=0, inplace=True)

        res = [mbs_df, res_df, scr_df]

        return res

def adjustsrun(x, inputs, lld_list, seed=None, delta=0, max_years=15, **kwargs):

    scr_df = cir.run(inputs, delta, max_years, seeds=[x[1]], debug=None, verbose=None, timings=None)

    if 'scr_df' in kwargs:
        scr_df.update(kwargs['scr_df'])

    res_df = None

    inputs.Adjusts = kwargs['adjustments']
    for lld_df in lld_list:
        for product in lld_df[LLD.Product].unique():
            if res_df is None:
                res_df = cfm.run(inputs, lld_df.loc[lld_df[LLD.Product] == product, :].copy(deep=True), scr_df, seed=seed, debug=None, verbose=True, timings=None)
            else:
                res_df += cfm.run(inputs, lld_df.loc[lld_df[LLD.Product] == product, :].copy(deep=True), scr_df, seed=seed, debug=None, verbose=True, timings=None)
    inputs.Adjusts = None

    mbs_df = mbs.run(inputs, res_df, scr_df, debug=None, verbose=True, timings=None)
    mbs_df.rename(index={0: x[0]}, level=0, inplace=True)
    res_df.rename(index={0: x[0]}, level=0, inplace=True)
    scr_df.rename(index={0: x[0]}, level=0, inplace=True)

    res = [mbs_df, res_df, scr_df]

    return res


class ModelManager(object):

    def __init__(self, rclient=None, loud=True, show_progress=True):

        self.updater = common.StatusUpdater(loud=loud, show_progress=show_progress)
        self.rclient = rclient

    def _restart_updater(self):

        self.updater.upd_que = None
        cur_status = self.updater.exit_status

        self.updater = common.StatusUpdater(
            calc_id=self.updater.calc_id,
            proxy_addr=self.updater.proxy_addr,
            owner_addr=self.updater.owner_addr,
            ms=self.updater.updateInterval,
            loud=self.updater.loud,
            show_progress=self.updater.show_progress
        )
        self.updater.exit_status = cur_status

    @saferun
    def LoadParams(self, fname, sqldb=None, mods=None, proxy=None, homedir=None, encoding='utf8'):

        self.inputs = common.CParams(fname)

        # set updater to notify service if needed
        if proxy is not None:
            self.updater.calc_id = self.inputs.CalculationID
            self.updater.proxy_addr = proxy
            self.updater.owner_addr = (self.inputs.ServiceHost, self.inputs.ServicePort)

        # load additional data
        self.inputs.GetDataFromDB(sqldb, homedir=homedir, encoding=encoding)

        # modify input arguments
        if mods is not None:
            for mod in mods:
                self.inputs.Parameters.set_par(*mod.split('='))

        # fix random number sequences
        numpy.random.seed(self.inputs.Parameters.RandomNumberSeed)
        self.seeds_lld = 123456
        self.seeds_cir = numpy.random.randint(1, 1000000)
        self.seeds_cfm = numpy.random.randint(1, 1000000)

    @saferun
    def LoadLLD(self, data):

        self.lld_df, self.stats = lld.run(df=data, inputs=self.inputs, newseed=self.seeds_lld)

    @saferun
    def Initialize(self):

        date = numpy.datetime64(self.inputs.Parameters.EvaluationDate, 'M')
        year = numpy.timedelta64(12, 'M')

        # clear HPIndex, Regions & Regions to prevent heavy network traffic
        self.inputs.Datasets.HPIndex = None
        self.inputs.Datasets.Regions = None
        self.inputs.Datasets.History = self.inputs.Datasets.History.loc[date - year:date, :].copy(deep=True)

        # clear Adjusts - they can be read at each node without network transmission
        self.inputs.Adjusts = None

        # ====== MACRO SIMULATIONS & CASHFLOW MODEL ======
        # generate macro seeds sequence
        numpy.random.seed(self.seeds_cir)
        self.xmap = numpy.zeros((self.inputs.Parameters.NumberOfMonteCarloScenarios, 2), dtype=numpy.uint64)
        self.xmap[:, 0] = numpy.arange(self.inputs.Parameters.NumberOfMonteCarloScenarios)
        self.xmap[:, 1] = numpy.random.randint(1, high=1001, size=self.inputs.Parameters.NumberOfMonteCarloScenarios)
        for i in range(1, self.inputs.Parameters.NumberOfMonteCarloScenarios):
            self.xmap[i, 1] += self.xmap[i - 1][1]

        # set total count of part to be calculated
        self.updater.total = len(self.xmap) * (3 if self.inputs.Parameters.CalculationOfEffectiveIndicators else 1)

        # --- divide big lld into several parts ---
        if (self.inputs.Parameters.NumberOfSyntheticLoans is not None) and (self.inputs.Parameters.NumberOfSyntheticLoans > 0):
            self.lld_list = [self.lld_df]
            self.part_num = 1
        else:
            lld_max = 100
            L = range(len(self.lld_df))
            self.lld_list = []
            while len(L) > 0:
                L0 = []
                for i in range(lld_max):
                    if len(L) > 0:
                        L0.append(L.pop(0))
                    else:
                        break
                self.lld_list.append(self.lld_df.iloc[numpy.array(L0), :].copy(deep=True))
            self.part_num = len(self.lld_list)

    @progress
    def Run(self, target=defaultrun, delta=100, **kwargs):

        # --- compute base scenario and remember central scenario ---
        self.lst_c = [res for res in common.runpar(target, self.xmap, self.inputs, self.lld_list, seed=self.seeds_cfm, max_years=self.inputs.Parameters.ModelingHorizon, rclient=self.rclient, upd_queue=self.updater.upd_que, **kwargs)]
        if self.inputs.Parameters.CalculationOfEffectiveIndicators:
            self.lst_m = [res for res in common.runpar(target, self.xmap, self.inputs, self.lld_list, seed=self.seeds_cfm, delta=-delta, max_years=self.inputs.Parameters.ModelingHorizon, rclient=self.rclient, upd_queue=self.updater.upd_que, **kwargs)]
            self.lst_p = [res for res in common.runpar(target, self.xmap, self.inputs, self.lld_list, seed=self.seeds_cfm, delta=+delta, max_years=self.inputs.Parameters.ModelingHorizon, rclient=self.rclient, upd_queue=self.updater.upd_que, **kwargs)]
        else:
            self.lst_m = []
            self.lst_p = []

    @saferun
    def Pricing(self):

        if self.inputs.Parameters.CalculationOfEffectiveIndicators:
            self.mbs_res, self.price_hist = mbs.stats(self.lst_c[0], self.lst_m[0], self.lst_p[0], self.inputs.Parameters)
        else:
            self.mbs_res, self.price_hist = mbs.stats(self.lst_c[0], None, None, self.inputs.Parameters)

    @saferun
    def Dumping(self, dumpdir=None, names=None, encoding='utf8'):

        if dumpdir is None:
            return

        if not os.path.isdir(dumpdir):
            os.makedirs(dumpdir)

        if names is None:
            names = ['mbsflows', 'cashflow', 'macroscr']
        names += ['df%d'%(i+1) for i in range(len(names), len(self.lst_c)+1)]

        # save modeled data
        for dat, scr in zip([self.lst_c, self.lst_m, self.lst_p], ['central', 'm100bp', 'p100bp']):
            for res, nam in zip(dat, names):
                if isinstance(res, pandas.core.frame.DataFrame):
                    res.to_csv(os.path.join(dumpdir, '%s_%s.csv' % (nam, scr)), sep='\t', encoding=encoding)
                elif isinstance(res, (str, unicode)):
                    with open(os.path.join(dumpdir, '%s_%s.txt' % (nam, scr)), 'w') as f:
                        f.write(res.encode(encoding))
                else:
                    pass

        # save clustered lld
        self.lld_df.to_csv(os.path.join(dumpdir, 'lld_dataframe.csv'), sep='\t', index=False, encoding=encoding)
        self.stats.to_csv(os.path.join(dumpdir, 'stt_dataframe.csv'), sep='\t', index=False, encoding=encoding)

        # save pricing table
        self.mbs_res.to_csv(os.path.join(dumpdir, 'pricing_result.csv'), sep='\t', encoding=encoding)
