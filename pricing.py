# -*- coding: utf8 -*-
import os
import numpy as np
import pandas as pd
import json
import argparse

import impdata


def main(args):

    if args.spread is None:
        if args.price is None:
            print 'One of [Spread] or [Price] should be specified.'
            return
        else:
            # compute with fixed price
            CalculationType = "Price"
            CalculationValue = args.price
    else:
        # compute with fixed spread
        CalculationType = "OAS"
        CalculationValue = args.spread

    # get current zcyc params
    g_params = impdata.kbd_func(args.curdate, spread=0.0)

    # get mbs params
    b_params = impdata.mbs_func(args.isin, args.curdate)

    if args.curdate is not None:
        ActualLLDDate = "%04d-%02d-%02d" % (pd.to_datetime(args.curdate).year, pd.to_datetime(args.curdate).month, 1)
    else:
        ActualLLDDate = None

    # if ActualLLDDate is None or b_params["IssuedDate"] is None:
    #     print 'Data is not available for "%s" at "%s" (err code 1)' % (args.isin, args.curdate)
    #     return
    #
    # if np.datetime64(args.curdate, 'D') < np.datetime64(ActualLLDDate, 'D'):
    #     print 'Data is not available for "%s" at "%s" (err code 2)' % (args.isin, args.curdate)
    #     return
    #
    # if np.datetime64(args.curdate, 'D') < np.datetime64(b_params["IssuedDate"], 'D') - np.timedelta64(30, 'D'):
    #     print 'Data is not available for "%s" at "%s" (err code 3)' % (args.isin, args.curdate)
    #     return

    for_json = {
        # description
        "ISIN": b_params['ISIN'],

        # input lld
        "InputFileName": args.inputs,

        # zcyc data
        "Coefficients": g_params,

        # params
        "Parameters": {
            "ID": 0,

            # common params
            "CalculationType": CalculationType,
            "CalculationValue": CalculationValue,
            "EvaluationDate": args.curdate,
            "ActualLLDDate": ActualLLDDate,

            # zcyc common params
            "UseStandartZCYC": True,
            "ZCYCDate": args.curdate,
            "ZCYCValues": None,

            # mbs constant params
            "Nominal": b_params["Nominal"],
            "CouponType": b_params["CouponType"],
            "CouponPercentage": b_params["CouponPercentage"],
            "CouponPeriod": b_params["CouponPeriod"],
            "IssuedDate": b_params["IssuedDate"],
            "FirstCouponDate": b_params["FirstCouponDate"],
            "StartIssueAmount": b_params["StartIssueAmount"],
            "CleanUpCall": b_params["CleanUpCall"],
            "NonRecurringExpenses": b_params["NonRecurringExpenses"],
            "ConstantExpenses": b_params["ConstantExpenses"],
            "VariableExpenses": b_params["VariableExpenses"],

            # mbs time-varying params
            "CurrentNominal": b_params["CurrentNominal"],
            "AccruedPercentage": b_params["AccruedPercentage"],

            # calculation
            "NumberOfMonteCarloScenarios": args.scenarios,
            "ModelingHorizon": args.horizon,
            "NumberOfEventSimulations": args.events,
            "NumberOfSyntheticLoans": args.clusters,
            "RandomNumberSeed": args.seed,
            "CalculationOfEffectiveIndicators": args.noconv is None,

            # model params
            "ModelsOfDefaults": 1.0,
            "EarlyPaymentModels": 1.0,
            "ModelsOfAcceleratedDepreciation": 1.0,
            "BetModels": 1.0,
            "MortgageRatesModels": 1.0,
            "RealEstateIndexModels": 1.0
        }
    }

    objrepr = json.dumps(for_json, indent=4, ensure_ascii=False, default=str).encode(args.encoding)
    if args.params is not None:
        with open(args.params, 'w') as f:
            f.write(objrepr)
    else:
        print objrepr


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", action="store", type=str, help="result path to json-file with model params")
parser.add_argument("-i", "--inputs", action="store", type=str, help="path lld datafile")

parser.add_argument("--isin", action="store", type=str, help="Bond ISIN")
parser.add_argument("--curdate", action="store", type=str, help="Current Date")

parser.add_argument("--spread", action="store", type=float, help="Bond ID")
parser.add_argument("--price", action="store", type=float, help="Bond ID")
parser.add_argument("--scenarios", action="store", type=int, default=5000, help="NumberOfMonteCarloScenarios")
parser.add_argument("--horizon", action="store", type=int, default=12, help="ModelingHorizon")
parser.add_argument("--events", action="store", type=int, default=1000, help="NumberOfEventSimulations")
parser.add_argument("--clusters", action="store", default=10, type=int, help="NumberOfSyntheticLoans")
parser.add_argument("--seed", action="store", type=int, help="RandomNumberSeed")
parser.add_argument("--noconv", action="count", help="CalculationOfEffectiveIndicators == 0")

parser.add_argument("-e", "--encoding", action="store", type=str, default='utf8')


if __name__ == "__main__":

    main(parser.parse_args())

    # # === EXAMPLE OF USAGE FROM ANOTHER SCRIPT ===
    # argslist = [
    #     '--isin', 'RU000A0ZYL89',
    #     '--params', r'C:\CFM_Public\examples\RU000A0ZYL89\run_params.json',
    #     '--inputs', r'C:\CFM_Public\examples\RU000A0ZYL89\LLD 2019-04-01.csv',
    #     '--spread', '120',
    #     '--curdate', '2019-04-15',
    #     '--scenarios', '1',
    #     '--events', '1000',
    #     '--clusters', '10',
    #     '--seed', '123456',
    # ]
    # main(parser.parse_args(argslist))
    # # --- to use script from IDE ---

