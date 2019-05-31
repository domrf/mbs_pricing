# -*- coding: utf8 -*-
import os
import argparse
import json
import numpy as np
import pandas as pd

from model import LLD
import specs


def str_conv(x):
    if x != '':
        return x
    else:
        return np.nan

def float_conv(x):
    try:
        return np.float(x)
    except:
        return np.nan

def int_conv(x):
    try:
        return np.int(np.float(x))
    except:
        return np.nan

def date_conv(x):
    try:
        return np.datetime64(x, 'D')
    except:
        return np.nan

def check_field(sf, bounds=None, fillna=None, unique=False, msglist=None):

    # if bounds - tuple - (min, max) interval
    # if bounds - list  - [posval1, posval2, posval3...]

    lower = None
    upper = None
    values = None

    if type(bounds) == tuple:
        lower = bounds[0]
        upper = bounds[1]
    elif type(bounds) == list:
        values = bounds

    temp = sf.copy(deep=True)
    nums = np.array(range(1, len(sf) + 1))

    # --- check NA values ---
    L0 = pd.isnull(temp)
    if any(L0):
        if fillna is None:
            median = temp[~L0].quantile(.5)
            if pd.isnull(median):
                median = 0.0
                warn_list.append({'column': '', 'first': 0, 'count': 0, 'error': u'В файле отсутсвуют данные "%s"' % sf.name})
        else:
            median = fillna
        temp[L0] = median
        if msglist is not None:
            msglist.append({'column':sf.name,'first':nums[L0].min(),'count': int(L0.sum()),'warning': u'неверный формат или пропущеное значение заменено на "%s"'%median})

    # --- check bounds ---
    if lower is not None:
        L0 = temp < lower
        if any(L0):
            median = temp[~L0].quantile(.5)
            temp[L0] = median if fillna is None else fillna
            if msglist is not None:
                msglist.append({'column':sf.name,'first':nums[L0].min(),'count': int(L0.sum()),'warning': u'значение меньше допустимого "%s" заменено на "%s"'%(lower, median)})

    if upper is not None:
        L0 = temp > upper
        if any(L0):
            median = temp[~L0].quantile(.5)
            temp[L0] = median if fillna is None else fillna
            if msglist is not None:
                msglist.append({'column':sf.name,'first':nums[L0].min(),'count': int(L0.sum()),'warning':u'значение больше допустимого "%s" заменено на "%s"'%(upper, median)})


    # --- check list values ---
    if values is not None:
        curr_values = sf.unique()
        for value in curr_values:
            if value not in values and not np.isnan(value):
                L0 = temp == value
                if any(L0):
                    median = temp[~L0].quantile(.5)
                    temp[L0] = median if fillna is None else fillna
                    if msglist is not None:
                        msglist.append({'column':sf.name,'first':nums[L0].min(),'count': int(L0.sum()),'warning':u'недопустимое значение справочника "%s" заменено на "%s"'%(value, median)})

    if unique:
        L0 = len(sf) - len(sf.unique().tolist())
        if L0 > 0:
            temp = np.array(range(len(sf))).astype('str')
            if msglist is not None:
                msglist.append({'column':sf.name,'first': 1,'count':L0,'warning':u'не уникальные номера договоров заменены на уникальные'})

    return temp

def main(args):

    if args.input is None or not os.path.isfile(args.input):
        print "LLD file '%s' do not exist!" % args.input
        exit(0)

    warn_list = []
    useconv = {
        LLD.Mortgage_No: str_conv,
        LLD.Issue_Date: date_conv,
        LLD.Maturity_Date: date_conv,
        LLD.Current_Debt: float_conv,
        LLD.Interest_Percent: float_conv,
        LLD.LTV: float_conv,
        LLD.Region: int_conv,
        LLD.Original_Debt: float_conv,
        LLD.Income: float_conv,
        LLD.Education: int_conv,
        LLD.Product: int_conv
    }

    # --- try reading file with different encodings ---
    try:
        df = pd.read_csv(args.input, sep='\t', nrows=0, encoding=args.encoding)
    except UnicodeDecodeError:
        df = None
        warn_list.append({'column':'','first':0,'count':0,'error':u'Файл с данными должен быть в кодировке "%s"'%args.encoding})
    except Exception as e:
        print str(e)
        df = None
        warn_list.append({'column':'','first':0,'count':0,'error':u'Формат Файла с данными: таблица в текстовом виде с разделителем колонок - табуляция, разедлителем десятичных - точка, даты в формате "ДДДД-ММ-ДД"'})
    else:
        if not all([col in df.columns for col in useconv]):
            for col in useconv:
                if not (col in df.columns):
                    warn_list.append({'column':'','first':0,'count':0,'error':u'В файле отсутсвуют данные "%s" (или кодировка не "%s")'%(col,args.encoding)})

    # --- if encoding is ok than read fuul dataframe ---
    if df is not None:

        df = pd.read_csv(args.input, sep='\t', converters=useconv, encoding=args.encoding)

        INTEREST_MAX = 30.0
        LTV_MAX = 90.0
        PTI_MAX = 45.0
        NUMS = np.array(range(1, len(df) + 1))
        Current_Date = None if args.datadate is None else np.datetime64(args.datadate, 'D')

        # --- check for unique mortgage_no ---
        if LLD.Mortgage_No in df.columns:
            df[LLD.Mortgage_No] = check_field(df[LLD.Mortgage_No], unique=True, msglist=warn_list)

        # --- check for correct issue dates ---
        if LLD.Issue_Date in df.columns:
            df[LLD.Issue_Date] = check_field(df[LLD.Issue_Date], msglist=warn_list)

            if Current_Date is not None:
                # logical error in issue dates
                L0 = df[LLD.Issue_Date] > Current_Date
                if any(L0):
                    df.loc[L0, LLD.Issue_Date] = Current_Date - np.timedelta64(30, 'D')
                    warn_list.append({'column': LLD.Issue_Date,'first': NUMS[L0].min(),'count':int(L0.sum()),'warning': u'дата выдачи кредита позднее даты актуальности данных "%s"'%Current_Date})

        # --- check for correct maturity dates ---
        if LLD.Maturity_Date in df.columns:
            df[LLD.Maturity_Date] = check_field(df[LLD.Maturity_Date], msglist=warn_list)

            if Current_Date is not None:
                # logical error in maturity dates
                L0 = df[LLD.Maturity_Date] < Current_Date
                if any(L0):
                    df.loc[L0, LLD.Maturity_Date] = Current_Date + np.timedelta64(30, 'D')
                    warn_list.append({'column':LLD.Maturity_Date,'first':NUMS[L0].min(),'count':int(L0.sum()),'warning':u'дата погашения кредита меньше даты актуальности данных "%s"'%Current_Date})

            if LLD.Issue_Date in df.columns:
                # logical error in issue dates
                L0 = df[LLD.Issue_Date] > df[LLD.Maturity_Date]
                if any(L0):
                    df.loc[L0, LLD.Issue_Date] = df.loc[L0, LLD.Issue_Date] + np.timedelta64(5*12*30, 'D')
                    warn_list.append({'column': LLD.Issue_Date,'first': NUMS[L0].min(),'count':int(L0.sum()),'warning': u'дата выдачи кредита позднее даты погашения "%s"'%Current_Date})

        # --- check current debt ---
        if LLD.Current_Debt in df.columns:
            df[LLD.Current_Debt] = check_field(df[LLD.Current_Debt], bounds=(0.0, None), msglist=warn_list)

        # --- check mortgage rates ---
        if LLD.Interest_Percent in df.columns:
            df[LLD.Interest_Percent] = check_field(df[LLD.Interest_Percent], bounds=(0.01, INTEREST_MAX), msglist=warn_list)

            # logical warning - possibly percents should be multiplied by 100
            L0 = df[LLD.Interest_Percent] < (INTEREST_MAX * 0.01)
            if any(L0):
                warn_list.append({'column':LLD.Interest_Percent,'first': NUMS[L0].min(),'count':int(L0.sum()),'warning': u'значение процентной ставки меньше 1%'})

        # --- check loan/value ---
        if LLD.LTV in df.columns:
            df[LLD.LTV] = check_field(df[LLD.LTV], bounds=(0.01, LTV_MAX), msglist=warn_list)

            # logical warning - possibly percents should be multiplied by 100
            L0 = df[LLD.LTV] < (LTV_MAX * 0.01)
            if (float(L0.sum()) / len(df)) > 0.20:
                warn_list.append({'column':LLD.LTV,'first': NUMS[L0].min(),'count':int(L0.sum()),'warning': u'значение кредит/залог меньше 1% более чем у пятой части пула'})

        # --- check regions ---
        if LLD.Region in df.columns:
            df[LLD.Region] = check_field(df[LLD.Region], bounds=range(1, 101), msglist=warn_list)

        # --- check original debt ---
        if LLD.Original_Debt in df.columns:
            df[LLD.Original_Debt] = check_field(df[LLD.Original_Debt], bounds=(0.01, None), msglist=warn_list)

            if LLD.Current_Debt in df.columns:
                # logical error - to low original debt
                L0 = df[LLD.Original_Debt] < df[LLD.Current_Debt]
                if any(L0):
                    df.loc[L0, LLD.Original_Debt] = df.loc[L0, LLD.Current_Debt]
                    warn_list.append({'column':LLD.Original_Debt,'first':NUMS[L0].min(),'count':int(L0.sum()),'warning':u'первоначальный ОД по кредиту заменен на текущий ОД'})

        # --- check mortgage incomes ---
        if LLD.Income in df.columns:
            df[LLD.Income] = check_field(df[LLD.Income], bounds=(0.0, None), msglist=warn_list)

            if (LLD.Original_Debt in df.columns) and (LLD.Interest_Percent in df.columns):
                # logical warning - to high pti
                L0 = PTI_MAX < (df[LLD.Original_Debt] * df[LLD.Interest_Percent] / (12.0 * df[LLD.Income]))
                if any(L0):
                    warn_list.append({'column':LLD.Income,'first':NUMS[L0].min(),'count':int(L0.sum()),'warning':u'показатель PTI по кредиту возможно превышает 45%'})

        # --- check education ---
        if LLD.Education in df.columns:
            df[LLD.Education] = check_field(df[LLD.Education], bounds=[0, 1, 2, 3, 4, 5, 6, 7], msglist=warn_list)

        # --- check product groups ---
        if LLD.Product in df.columns:
            df[LLD.Product] = check_field(df[LLD.Product], bounds=specs.adjust_dict.keys(), fillna=0, msglist=warn_list)


    # --- print out about warnings ---
    crit_error = u'Ошибка: %s.'
    warn_error = u'Предупреждение: в колонке "%s" в строке №%d %s (всего %d).'

    report = {'Errors': [],'Warnings':[]}

    for warn in warn_list:
        if 'error' in warn:
            warn['repr'] = crit_error % warn['error']
            report['Errors'].append(warn)
        else:
            warn['repr'] = warn_error % (warn['column'], warn['first'], warn['warning'], warn['count'])
            report['Warnings'].append(warn)

    objrepr = json.dumps(report, indent=4, ensure_ascii=False).encode(args.encoding).replace('NaN', 'null')

    # report about data-warnings
    L0 = len(report['Warnings'])
    for i in range(L0):
        print '\t', i, report['Warnings'][i]['repr']

    # report about data-erorrs
    L0 = len(report['Errors'])
    for i in range(L0):
        print '\t', i, report['Errors'][i]['repr']

    if args.result is not None:
        with open(args.result, 'w') as f:
            f.write(objrepr)

    if df is not None:
        df.to_csv(args.output, sep='\t', encoding=args.encoding, index=False)
    else:
        open(args.output, 'w').close()


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", action="store", type=str)
parser.add_argument("-o", "--output", action="store", type=str)
parser.add_argument("-r", "--result", action="store", type=str)
parser.add_argument("-d", "--datadate", action="store", type=str)
parser.add_argument("-e", "--encoding", action="store", type=str, default='utf8')


if __name__ == "__main__":

    main(parser.parse_args())

    # # === EXAMPLE OF USAGE ===
    # # --- to use script from IDE ---
    # argslist = [
    #     '-i', r'C:\CFM_Public\examples\RU000A0ZYJT2\LLD 2019-04-01.csv',
    #     '-o', r'C:\CFM_Public\examples\RU000A0ZYJT2\temp_LLD 2019-04-01.csv',
    # ]
    # main(parser.parse_args(argslist))
    # # --- to use script from IDE ---
