# -*- coding: utf8 -*-
import os
import time
import argparse
import numpy as np
import pandas as pd

import model
import reader
import specs

def modelrun(x, inputs, lld_df, seed=None, delta=0, max_years=15):

    # --- run model for macro scenarios ---
    scr_df = model.cir.run(inputs, delta, max_years, seeds=[x[1]])

    # --- for each product run cashflows model ---
    res_df = None
    for product in lld_df[model.LLD.Product].unique():

        # find product specification
        if product in specs.adjust_dict.keys():
            inputs.Adjusts = specs.adjust_dict[product]
        else:
            print time.strftime('%Y-%m-%d %H:%M:%S'), 'Model does not support product type = "%s" (product was set to default type).' % product
            inputs.Adjusts = specs.adjust_dict[0]

        # run model and accumulate results
        if res_df is None:
            res_df = model.cfm.run(inputs, lld_df.loc[lld_df[model.LLD.Product] == product, :].copy(deep=True), scr_df, seed=seed)
        else:
            res_df += model.cfm.run(inputs, lld_df.loc[lld_df[model.LLD.Product] == product, :].copy(deep=True), scr_df, seed=seed)

    inputs.Adjusts = None

    # --- run mbs tranching model ---
    mbs_df = model.mbs.run(inputs, res_df, scr_df)

    # --- add info about calculated scenario ---
    scr_df.rename(index={0: x[0]}, level=0, inplace=True)
    res_df.rename(index={0: x[0]}, level=0, inplace=True)
    mbs_df.rename(index={0: x[0]}, level=0, inplace=True)

    # --- return results for current macro scenario ---
    return mbs_df, res_df, scr_df,

def main(args):

    if args.params is None or not os.path.isfile(args.params):
        print "File '%s' do not exist!" % args.params
        exit(0)

    # ====== MODEL INPUTS ======
    try:
        inputs = model.CParams(args.params)
    except Exception as e:
        raise e

    if args.mods is not None:
        for mod in args.mods:
            inputs.Parameters.set_par(*mod.split('='))

    # --- fix random number sequences ---
    np.random.seed(inputs.Parameters.RandomNumberSeed)
    seeds_lld = 123456
    seeds_cir = np.random.randint(1, 1000000)
    seeds_cfm = np.random.randint(1, 1000000)

    # ====== DATA VALIDATION ======
    try:
        print time.strftime('%Y-%m-%d %H:%M:%S'), 'Checking input LLD ...'
        if (inputs.InputFileName is not None) and (os.path.isfile(inputs.InputFileName)):
            # get paths
            workdir, name = os.path.split(inputs.InputFileName)
            body, ext = name.split('.')
            new_fname0 = os.path.join(workdir, 'temp_' + body + '.' + ext)
            new_fname1 = os.path.join(workdir, 'data_' + body + '.' + ext)
            new_fname2 = os.path.join(workdir, 'stat_' + body + '.' + ext)
            # check input lld
            reader.main(reader.parser.parse_args(['-i', inputs.InputFileName, '-o', new_fname0]))
            # make clusterization
            lld_df, stats = model.lld.run(df=pd.read_csv(new_fname0, sep='\t', encoding=args.encoding, parse_dates=True), inputs=inputs, newseed=seeds_lld)
            # save clusterization results
            lld_df.to_csv(new_fname1, sep='\t', encoding=args.encoding)
            stats.to_csv(new_fname2, sep='\t', encoding=args.encoding)
        else:
            raise Exception('file "%s" not found' % inputs.InputFileName)
    except Exception as e:
        print "Error while reading LLD:", str(e), '(please check file for ".csv" structure or missed rows/columns)'
        exit(0)

    # clear HPIndex, Regions & Regions to prevent heavy network traffic
    inputs.Datasets.HPIndex = None
    inputs.Datasets.Regions = None
    inputs.Datasets.History = inputs.Datasets.History.loc[np.datetime64(inputs.Parameters.EvaluationDate, 'M') - np.timedelta64(12, 'M'):np.datetime64(inputs.Parameters.EvaluationDate, 'M'), :].copy(deep=True)

    # clear Adjusts - they can be read at each node without network transmission
    inputs.Adjusts = None

    # ====== MACRO SIMULATIONS & CASHFLOW MODEL ======
    # generate macro seeds sequence
    np.random.seed(seeds_cir)
    xmap = np.zeros((inputs.Parameters.NumberOfMonteCarloScenarios, 2), dtype=np.uint64)
    xmap[:, 0] = np.arange(inputs.Parameters.NumberOfMonteCarloScenarios)
    xmap[:, 1] = np.random.randint(1, high = 1001, size = inputs.Parameters.NumberOfMonteCarloScenarios)
    for i in range(1, inputs.Parameters.NumberOfMonteCarloScenarios):
        xmap[i, 1] += xmap[i - 1][1]

    # status updater shows progress
    sUpdt = model.StatusUpdater(len(xmap) * (3 if inputs.Parameters.CalculationOfEffectiveIndicators else 1), loud=True)
    sUpdt.start()

    mbs_m = None
    mbs_p = None
    if inputs.Parameters.CalculationOfEffectiveIndicators:
        # --- compute +/-100bp scenarios if needed ---
        mbs_m, res_m, scr_m = model.runpar(modelrun, xmap, inputs, lld_df, seed=seeds_cfm, delta=-100, max_years=inputs.Parameters.ModelingHorizon, upd_queue=sUpdt.upd_que)
        mbs_p, res_p, scr_p = model.runpar(modelrun, xmap, inputs, lld_df, seed=seeds_cfm, delta=+100, max_years=inputs.Parameters.ModelingHorizon, upd_queue=sUpdt.upd_que)
    # --- compute base scenario and remember central scenario ---
    mbs_c, res_c, scr_c = model.runpar(modelrun, xmap, inputs, lld_df, seed=seeds_cfm, max_years=inputs.Parameters.ModelingHorizon, upd_queue=sUpdt.upd_que)

    sUpdt.stop()

    # ====== MBS MODEL ======
    try:
        print time.strftime('%Y-%m-%d %H:%M:%S'), 'Calculating price ...',
        mbs_res, price_hist = model.mbs.stats(mbs_c, mbs_m, mbs_p, inputs.Parameters)
        print '\r', time.strftime('%Y-%m-%d %H:%M:%S'), 'Calculating price - OK'
    except Exception as e:
        print "Error while pricing cashflows:", str(e), '(please check for correct mbs inputs parameters)'

    # --- dump main dataframes ---
    try:
        print time.strftime('%Y-%m-%d %H:%M:%S'), 'Saving results ...',
        if not os.path.isdir(workdir):
            os.makedirs(workdir)
        if args.dumpall is None:
            if inputs.Parameters.CalculationOfEffectiveIndicators:
                # save -100 bp
                scr_m = scr_m.mean(axis=0, level=1)
                res_m = res_m.mean(axis=0, level=1)
                mbs_m = mbs_m.mean(axis=0, level=1)
                # save +100 bp
                scr_p = scr_p.mean(axis=0, level=1)
                res_p = res_p.mean(axis=0, level=1)
                mbs_p = mbs_p.mean(axis=0, level=1)
            # save base
            scr_c = scr_c.mean(axis=0, level=1)
            res_c = res_c.mean(axis=0, level=1)
            mbs_c = mbs_c.mean(axis=0, level=1)

        if inputs.Parameters.CalculationOfEffectiveIndicators:
            # save -100 bp
            scr_m.to_csv(os.path.join(workdir, 'macroscr_m100bp.csv'), sep='\t', encoding=args.encoding)
            res_m.to_csv(os.path.join(workdir, 'cashflow_m100bp.csv'), sep='\t', encoding=args.encoding)
            mbs_m.to_csv(os.path.join(workdir, 'mbsflows_m100bp.csv'), sep='\t', encoding=args.encoding)
            # save +100 bp
            scr_p.to_csv(os.path.join(workdir, 'macroscr_p100bp.csv'), sep='\t', encoding=args.encoding)
            res_p.to_csv(os.path.join(workdir, 'cashflow_p100bp.csv'), sep='\t', encoding=args.encoding)
            mbs_p.to_csv(os.path.join(workdir, 'mbsflows_p100bp.csv'), sep='\t', encoding=args.encoding)
        # save base
        scr_c.to_csv(os.path.join(workdir, 'macroscr_central.csv'), sep='\t', encoding=args.encoding)
        res_c.to_csv(os.path.join(workdir, 'cashflow_central.csv'), sep='\t', encoding=args.encoding)
        mbs_c.to_csv(os.path.join(workdir, 'mbsflows_central.csv'), sep='\t', encoding=args.encoding)

        # save other
        mbs_res.to_csv(os.path.join(workdir, 'pricing_result.csv'), sep='\t', encoding=args.encoding)

        print '\r', time.strftime('%Y-%m-%d %H:%M:%S'), 'Results saved to "%s"' % workdir
    except Exception as e:
        print "Error while saving results:", str(e)


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", action="store", type=str, help="path tojson-file with model params")
parser.add_argument("-m", "--mods", action="append", type=str, help="parameters modifier list")
parser.add_argument("--dumpall", action="count", help="if set - save all scenarios")
parser.add_argument("-e", "--encoding", action="store", type=str, default='utf8')

if __name__ == "__main__":

    main(parser.parse_args())

    # # === EXAMPLE OF USAGE FROM ANOTHER SCRIPT ===
    # argslist = [
    #     '-p', r'C:\CFM_Public\examples\RU000A0ZYL89\run_params.json',
    #     '-m', 'NumberOfMonteCarloScenarios=1',
    #     '-m', 'RandomNumberSeed=123456',
    #     '--dumpall'
    # ]
    # main(parser.parse_args(argslist))
    # # --- to use script from IDE ---
