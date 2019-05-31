# -*- coding: utf8 -*-
import numpy
import pandas

data = pandas.DataFrame([
    [u'RU000A0ZYJT2', 1000, 'Fixed', 11.5, 3, '2017-12-07', '2018-04-28', 48197806000, 5, 56100000, 16480000, 1, u'ВТБ-1'],
    [u'RU000A0ZYL89', 1000, 'Floating', 0, 3, '2017-12-20', '2018-04-28', 7557011000, 5, 60300307.42, 2690800, 1.162, u'Райф-1'],
],
	columns=[
        u'ISIN',
        u'Nominal',
        u'CouponType',
        u'CouponPercentage',
        u'CouponPeriod',
        u'IssuedDate',
        u'FirstCouponDate',
        u'StartIssueAmount',
        u'CleanUpCall',
        u'NonRecurringExpenses',
        u'ConstantExpenses',
        u'VariableExpenses',
        u'MBS'
    ]
)

noms = pandas.DataFrame([
    [u'RU000A0ZYJT2', '2018-04-28', 801.83, 0],
    [u'RU000A0ZYJT2', '2018-07-28', 704.63, 0],
    [u'RU000A0ZYJT2', '2018-10-28', 611.01, 0],
    [u'RU000A0ZYJT2', '2019-01-28', 529.94, 0],
    [u'RU000A0ZYL89', '2018-04-28', 813.19, 0],
    [u'RU000A0ZYL89', '2018-07-28', 682.78, 0],
    [u'RU000A0ZYL89', '2018-10-28', 583.65, 0],
    [u'RU000A0ZYL89', '2019-01-28', 501, 0],
],
    columns=[u'ISIN', u'MeasureDate', u'CurrentNominal', u'AccruedPercentage']
)
noms[u'MeasureDate1'] = pandas.to_datetime(noms[u'MeasureDate'])
noms.sort_values(by=[u'ISIN', u'MeasureDate1'], inplace=True)


def get_bparams(isin, curdate):

    row1 = data.loc[data[u'ISIN']==isin, :]

    if len(row1) > 0:
        res = {
            "Nominal": row1.values[0, 1],
            "CouponType": row1.values[0, 2],
            "CouponPercentage": row1.values[0, 3],
            "CouponPeriod": row1.values[0, 4],
            "IssuedDate": "%s" % numpy.datetime64(row1.values[0, 5], 'D'),
            "FirstCouponDate": "%s" % numpy.datetime64(row1.values[0, 6], 'D'),
            "StartIssueAmount": row1.values[0, 7],
            "CleanUpCall": row1.values[0, 8],
            "NonRecurringExpenses": row1.values[0, 9],
            "ConstantExpenses": row1.values[0, 10],
            "VariableExpenses": row1.values[0, 11],
            "MBS": row1.values[0, 12],
            "ISIN": row1.values[0, 0]
        }
    else:
        res = {
            "Nominal": 1000.0,
            "CouponType": "Floating",
            "CouponPercentage": 0,
            "CouponPeriod": 3,
            "IssuedDate": None,
            "FirstCouponDate": None,
            "StartIssueAmount": None,
            "CleanUpCall": 5.0,
            "NonRecurringExpenses": None,
            "ConstantExpenses": None,
            "VariableExpenses": None,
            "MBS": "Unknown",
            "ISIN": "Unknown"
        }

    row2 = noms.loc[(noms[u'ISIN'] == isin) & (pandas.to_datetime(noms[u'MeasureDate']) <= curdate), :]
    if len(row2) > 0:
        res["CurrentNominal"] = row2.values[-1, 2]
        res["AccruedPercentage"] = row2.values[-1, 3]
    else:
        res["CurrentNominal"] = 1000.0
        res["AccruedPercentage"] = 0.0

    return res

