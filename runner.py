# -*- coding: utf8 -*-
import io
import os
import argparse

import pandas

import model
import reader
import specs

from model.common import Errors


def modelrun(x, inputs, lld_list, seed=None, delta=0, max_years=15, debug=None, profiling=None, verbose=None, dumpdir=None):

    # check if profiling mode is on
    timings = None if profiling is None else pandas.Series(0.0, index=[], name='Timings')

    # check if debug mode is on
    debug_buf = None if debug is None else io.StringIO()

    scr_df = model.cir.run(inputs, delta, max_years, seeds=[x[1]], debug=debug, verbose=verbose, timings=timings)

    res_df = None

    # n = len(lld_list)
    # k = 1
    for lld_df in lld_list:
        # print "%d / %d" %(k, n)
        # k += 1
        for product in lld_df[model.LLD.Product].unique():

            if product in specs.adjust_dict.keys():
                inputs.Adjusts = specs.adjust_dict[product]
            else:
                print 'Unsupported product "%s" (set to default)' % product
                inputs.Adjusts = specs.adjust_dict[0]

            if res_df is None:
                res_df = model.cfm.run(inputs, lld_df.loc[lld_df[model.LLD.Product] == product, :].copy(deep=True), scr_df, seed=seed, debug=debug_buf, verbose=verbose, timings=timings)
            else:
                res_df += model.cfm.run(inputs, lld_df.loc[lld_df[model.LLD.Product] == product, :].copy(deep=True), scr_df, seed=seed, debug=debug_buf, verbose=verbose, timings=timings)

    inputs.Adjusts = None

    mbs_df = model.mbs.run(inputs, res_df, scr_df, debug=debug, verbose=verbose, timings=timings)
    mbs_df.rename(index={0: x[0]}, level=0, inplace=True)

    res = [mbs_df]

    if dumpdir is not None:
        res_df.rename(index={0: x[0]}, level=0, inplace=True)
        res.append(res_df)
        scr_df.rename(index={0: x[0]}, level=0, inplace=True)
        res.append(scr_df)

    if debug is not None:
        res.append(debug_buf.getvalue())
        debug_buf.close()

    if profiling is not None:
        res.append(timings)

    return res


def main(args):

    # ====== ERRORS CHECK, CONNECTION ======
    if args.params is None or not os.path.isfile(args.params):
        print "File '%s' does not exist" % args.params
        return Errors.AccessError1

    # create model object
    m = model.ModelManager(rclient=None, loud=True)

    # read input params
    m.LoadParams(fname=args.params, mods=args.mods, encoding=args.encoding)

    # check input lld
    workdir, name = os.path.split(m.inputs.InputFileName)
    body, ext = name.split('.')
    temp_lld_name = os.path.join(workdir, 'temp_%s.%s' % (body, ext))
    reader.main(reader.parser.parse_args(['-i', m.inputs.InputFileName, '-o', temp_lld_name]))

    # load lld data
    m.LoadLLD(data=pandas.read_csv(temp_lld_name, sep='\t', encoding=args.encoding, parse_dates=True))

    # initiate calculation queue
    m.Initialize()

    # run model with different args
    m.Run(target=modelrun, debug=args.debug, profiling=args.profiling, verbose=args.verbose, dumpdir=args.dumpdir)

    # make mbs pricing great again
    m.Pricing()

    # dump some results if needed
    m.Dumping(dumpdir=args.dumpdir, encoding=args.encoding)

    # get profiling results
    if args.profiling is not None:
        tf = pandas.DataFrame(m.lst_c[-1])
        print tf
        print 'Total: %0.4f' % tf['Timings'].sum()

    # get debugging log results
    if args.debug is not None:
        if args.profiling is None:
            debug_log = m.lst_c[-1]
        else:
            debug_log = m.lst_c[-2]
        with open(args.debug, 'w') as f:
            f.write(debug_log.encode(args.encoding))
        # print debug_log

    return m.updater.exit_status


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", action="store", type=str, help="path tojson-file with model params")
parser.add_argument("-m", "--mods", action="append", type=str, help="parameters modifier list")
parser.add_argument("--dumpdir", action="store", type=str, help="location to save results")
parser.add_argument("--debug", action="store", type=str, help="filename to save debug outputs")
parser.add_argument("--profiling", action="count", help="turn on profiling mode")
parser.add_argument("--verbose", action="count", help="turn on verbose mode")
parser.add_argument("-e", "--encoding", action="store", type=str, default='utf8')


if __name__ == "__main__":

    exit(main(parser.parse_args()))

    # === EXAMPLE OF RUNNER USAGE ===
    # argslist = [
    #     '-p', r"C:\Users\andrei.shevchuk\Desktop\tester_CIR\run_RU000A0ZYJT2_2019-08-08.json",
    #     '-o', r"C:\Users\andrei.shevchuk\Desktop\tester_CIR\res_cir_revised_RU000A0ZYJT2_2019-08-08.json",
    #     '-m','NumberOfMonteCarloScenarios=10000',
    #     #'--dumpdir', r'C:\Temp\sometest',
    #     '--sqldb', 'webcalculator',
    #     '-m','RandomNumberSeed=123'
    #
    # ]
    # main(parser.parse_args(argslist))
    # --- to use script from IDE ---
