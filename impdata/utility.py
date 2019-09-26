# -*- coding: utf8 -*-
import numpy as np
import pandas as pd

from gparams import get_gparams
from mbs import get_bparams


def temp_json_dict(b_params, g_params, inp_fname, curdate, CalculationType, CalculationValue, ActualLLDDate,
                   scenarios=2500, horizon=15, events=1000, clusters=10, seed=None, noconv=None):

    for_json = {

        # description
        "ISIN": b_params['ISIN'],

        # input lld
        "InputFileName": inp_fname,

        # zcyc data
        "Coefficients": g_params,

        # params
        "Parameters": {
            "ID": 0,

            # common params
            "CalculationType": CalculationType,
            "CalculationValue": CalculationValue,
            "EvaluationDate": curdate,
            "ActualLLDDate": ActualLLDDate,

            # zcyc common params
            "UseStandartZCYC": True,
            "ZCYCDate": curdate,
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
            "NumberOfMonteCarloScenarios": scenarios,
            "ModelingHorizon": horizon,
            "NumberOfEventSimulations": events,
            "NumberOfSyntheticLoans": clusters,
            "RandomNumberSeed": seed,
            "CalculationOfEffectiveIndicators": noconv is None,

            # model params
            "ModelsOfDefaults": 1.0,
            "EarlyPaymentModels": 1.0,
            "ModelsOfAcceleratedDepreciation": 1.0,
            "BetModels": 1.0,
            "MortgageRatesModels": 1.0,
            "RealEstateIndexModels": 1.0
        }
    }

    return for_json


def gen_utility_json(args):

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
    g_params = get_gparams(args.curdate, spread=0.0)

    # get mbs params
    b_params = get_bparams(args.isin, args.curdate)

    if args.curdate is not None:
        ActualLLDDate = "%04d-%02d-%02d" % (pd.to_datetime(args.curdate).year, pd.to_datetime(args.curdate).month, 1)
    else:
        ActualLLDDate = None

    return temp_json_dict(b_params, g_params, args.inputs, args.curdate, CalculationType, CalculationValue, ActualLLDDate,
                          args.scenarios, args.horizon, args.events, args.clusters, args.seed, args.noconv)


def gen_std_json(curdate, bondisin=None, price=100.0, spread=None):

    if spread is None:
        if price is None:
            print 'One of [Spread] or [Price] should be specified.'
            return
        else:
            # compute with fixed price
            CalculationType = "Price"
            CalculationValue = price
    else:
        # compute with fixed spread
        CalculationType = "OAS"
        CalculationValue = spread

    # get current zcyc params
    g_params = get_gparams(curdate, spread=0.0)

    # get mbs params
    b_params = get_bparams(bondisin, curdate)

    if curdate is not None:
        ActualLLDDate = "%04d-%02d-%02d" % (pd.to_datetime(curdate).year, pd.to_datetime(curdate).month, 1)
    else:
        ActualLLDDate = None

    return temp_json_dict(b_params, g_params, None, curdate, CalculationType, CalculationValue, ActualLLDDate)

