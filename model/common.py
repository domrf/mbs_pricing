# -*- coding: utf8 -*-
import re
import time
import json

import threading
import Queue

import numpy as np
import pandas as pd

import impdata

from specs import cir_default

ALF = "abcdefghijklmnopqrstuvwxyz"
PTT = re.compile("(?:[\[][^\[\]]+[\]])")


class Errors:

    NoErrors = 0
    CannotConnectToCluster = 1
    AccessError1 = 2
    AccessError2 = 3
    AccessError3 = 4
    AccessError4 = 5
    LoadParamsError = 6
    LoadLLDError = 7
    InitializeError = 8
    PricingError = 9
    DumpingError = 10
    SavingError = 11
    ModelBaseError = 12


class CModelParams(object):

    def __init__(self, params):

        # with open(fname, 'r') as f:
        #     temp = json.loads(f.read())

        def convert_ZCYCValues(values_list):
            N = len(values_list)
            a = np.zeros((N, 2))
            for i in range(N):
                a[i, 0] = values_list[i]['X']
                a[i, 1] = values_list[i]['Y'] / 100.0
            a = a[a[:, 0].argsort()]
            return {'xp': a[:, 0], 'fp': a[:, 1]}

        # --- basic params ---
        self.EvaluationDate = np.datetime64(params['EvaluationDate'], 'D')
        self.EvaluationMonth = np.datetime64(params['EvaluationDate'], 'M')
        self.ActualLLDDate = np.datetime64(params['ActualLLDDate'], 'D')
        self.ActualLLDMonth = np.datetime64(params['ActualLLDDate'], 'M')
        self.ModelingHorizon = int(params['ModelingHorizon'])
        self.NumberOfMonteCarloScenarios = params['NumberOfMonteCarloScenarios']
        self.NumberOfEventSimulations = params['NumberOfEventSimulations']
        self.NumberOfSyntheticLoans = params['NumberOfSyntheticLoans']
        try:
            self.RandomNumberSeed = int(params['RandomNumberSeed'])
        except:
            self.RandomNumberSeed = None
        self.CalculationOfEffectiveIndicators = params['CalculationOfEffectiveIndicators']

        # --- macro model params ---
        self.UseStandartZCYC = params['UseStandartZCYC']
        self.ZCYCDate = np.datetime64(params['ZCYCDate'], 'D')
        self.ZCYCMonth = np.datetime64(params['ZCYCDate'], 'M')
        self.ZCYCValues = None if params['ZCYCValues'] is None else convert_ZCYCValues(params['ZCYCValues'])
        self.BetModels = params['BetModels']
        self.MortgageRatesModels = params['MortgageRatesModels']
        self.RealEstateIndexModels = params['RealEstateIndexModels']
        self.ForwardMonths = int(round(params['ForwardMonths'])) if 'ForwardMonths' in params else 0

        # --- cashflow model params ---
        self.ModelsOfDefaults = params['ModelsOfDefaults']
        self.EarlyPaymentModels = params['EarlyPaymentModels']
        self.ModelsOfAcceleratedDepreciation = params['ModelsOfAcceleratedDepreciation']
        # self.ReplacePrepayments = params['ReplacePrepayments']
        self.ReplacePrepayments = None

        # --- mbs specific params ---
        self.CalculationType = params['CalculationType']
        self.CalculationValue = params['CalculationValue']
        self.Nominal = params['Nominal']
        self.CurrentNominal = 1000.0 if params['CurrentNominal'] is None else params['CurrentNominal']
        self.CouponType = params['CouponType']
        self.AccruedPercentage = params['AccruedPercentage']
        self.CouponPercentage = params['CouponPercentage']
        self.CouponPeriod = np.timedelta64(params['CouponPeriod'], 'M')
        self.IssuedDate = np.datetime64(params['IssuedDate'], 'D')
        self.IssuedMonth = np.datetime64(params['IssuedDate'], 'M')
        self.FirstCouponDate = np.datetime64(params['FirstCouponDate'], 'D')
        self.FirstCouponMonth = np.datetime64(params['FirstCouponDate'], 'M')
        self.StartIssueAmount = params['StartIssueAmount']
        self.CleanUpCall = params['CleanUpCall']
        self.NonRecurringExpenses = params['NonRecurringExpenses']
        self.ConstantExpenses = params['ConstantExpenses']
        self.VariableExpenses = params['VariableExpenses']

        # --- mortgage vacations ---
        self.mv_length = 6
        self.mv_pow1 = 1.0
        self.mv_pow2 = 1.0

    def set_par(self, p_name, p_value):

        if p_name == 'EvaluationDate':
            self.EvaluationDate = np.datetime64(p_value, 'D')
            self.EvaluationMonth = np.datetime64(p_value, 'M')
        elif p_name == 'ActualLLDDate':
            self.ActualLLDDate = np.datetime64(p_value, 'D')
            self.ActualLLDMonth = np.datetime64(p_value, 'M')
        elif p_name == 'ZCYCDate':
            self.ZCYCDate = np.datetime64(p_value, 'D')
            self.ZCYCMonth = np.datetime64(p_value, 'M')
        elif p_name == 'IssuedDate':
            self.IssuedDate = np.datetime64(p_value, 'D')
            self.IssuedMonth = np.datetime64(p_value, 'M')
        elif p_name == 'FirstCouponDate':
            self.FirstCouponDate = np.datetime64(p_value, 'D')
            self.FirstCouponMonth = np.datetime64(p_value, 'M')
        elif p_name == 'CouponPeriod':
            self.CouponPeriod = np.timedelta64(p_value, 'M')
        elif p_name in ['CalculationPurpose', 'CouponType']:
            setattr(self, p_name, p_value)
        elif p_name in ['NumberOfMonteCarloScenarios', 'NumberOfEventSimulations', 'NumberOfSyntheticLoans', 'ModelingHorizon']:
            setattr(self, p_name, int(p_value))
        elif p_name == 'RandomNumberSeed':
            try:
                setattr(self, p_name, int(p_value))
            except:
                setattr(self, p_name, None)
        else:
            setattr(self, p_name, float(p_value))

    def to_json_dict(self):

        res = {
            # --- basic params ---
            'EvaluationDate': str(self.EvaluationDate),
            # 'EvaluationMonth': str(self.EvaluationMonth),
            'ActualLLDDate': str(self.ActualLLDDate),
            # 'ActualLLDMonth': str(self.ActualLLDMonth),
            'ModelingHorizon': self.ModelingHorizon,
            'NumberOfMonteCarloScenarios': self.NumberOfMonteCarloScenarios,
            'NumberOfEventSimulations': self.NumberOfEventSimulations,
            'NumberOfSyntheticLoans': self.NumberOfSyntheticLoans,
            'RandomNumberSeed': self.RandomNumberSeed,
            'CalculationOfEffectiveIndicators': self.CalculationOfEffectiveIndicators,

            # --- macro model params ---
            'UseStandartZCYC': self.UseStandartZCYC,
            'ZCYCDate': str(self.ZCYCDate),
            # 'ZCYCMonth': str(self.ZCYCMonth),
            'ZCYCValues': None,
            'BetModels': self.BetModels,
            'MortgageRatesModels': self.MortgageRatesModels,
            'RealEstateIndexModels': self.RealEstateIndexModels,
            'ForwardMonths': self.ForwardMonths,

            # --- cashflow model params ---
            'ModelsOfDefaults': self.ModelsOfDefaults,
            'EarlyPaymentModels': self.EarlyPaymentModels,
            'ModelsOfAcceleratedDepreciation': self.ModelsOfAcceleratedDepreciation,
            # 'ReplacePrepayments': self.ModelsOfAcceleratedDepreciation,

            # --- mbs specific params ---
            'CalculationType': self.CalculationType,
            'CalculationValue': self.CalculationValue,
            'Nominal': self.Nominal,
            'CurrentNominal': self.CurrentNominal,
            'CouponType': self.CouponType,
            'AccruedPercentage': self.AccruedPercentage,
            'CouponPercentage': self.CouponPercentage,
            'CouponPeriod': int(self.CouponPeriod),
            'IssuedDate': str(self.IssuedDate),
            # 'IssuedMonth': str(self.IssuedMonth),
            'FirstCouponDate': str(self.FirstCouponDate),
            # 'FirstCouponMonth': str(self.FirstCouponMonth),
            'StartIssueAmount': self.StartIssueAmount,
            'CleanUpCall': self.CleanUpCall,
            'NonRecurringExpenses': self.NonRecurringExpenses,
            'ConstantExpenses': self.ConstantExpenses,
            'VariableExpenses': self.VariableExpenses,
        }

        return res


class CZCYCParams(object):

    def __init__(self, coeffs):

        self.ID = coeffs['ID']
        self.Date = np.datetime64(coeffs['Date'], 'D')
        self.Month = np.datetime64(coeffs['Date'], 'M')
        self.B0 = coeffs['B0']
        self.B1 = coeffs['B1']
        self.B2 = coeffs['B2']
        self.TAU = coeffs['TAU']
        self.G1 = coeffs['G1']
        self.G2 = coeffs['G2']
        self.G3 = coeffs['G3']
        self.G4 = coeffs['G4']
        self.G5 = coeffs['G5']
        self.G6 = coeffs['G6']
        self.G7 = coeffs['G7']
        self.G8 = coeffs['G8']
        self.G9 = coeffs['G9']

    def _asdict(self):

        return {
            "B0": self.B0,
            "B1": self.B1,
            "B2": self.B2,
            "TAU": self.TAU,
            "G1": self.G1,
            "G2": self.G2,
            "G3": self.G3,
            "G4": self.G4,
            "G5": self.G5,
            "G6": self.G6,
            "G7": self.G7,
            "G8": self.G8,
            "G9": self.G9,
        }


class CModelDatasets(object):

    def __init__(self, isin):

        # 1. history data
        self.History = impdata.hst_data

        # 2. regions data
        self.Regions = impdata.rgn_data

        # 3. house price index
        self.HPIndex = impdata.hpi_data

        # 4. prepare history cashflow for mbs
        self.Hist_CF = impdata.cpn_func(isin)


class CMacroModel(object):

    def __init__(self, params):

        self.cir_a  = params['cir_a']
        self.cir_s  = params['cir_s']
        self.cir_ax = params['cir_ax']
        self.cir_sx = params['cir_sx']
        self.cir_tx = params['cir_tx']
        self.ms6_s  = params['ms6_s']
        self.hpr_s  = params['hpr_s']


class CParams(object):

    def __init__(self, inputjson):

        # <inputjson> is a json-dictonary or filename to read
        if isinstance(inputjson, dict):
            temp = inputjson
        else:
            with open(inputjson, 'r') as f:
                temp = json.loads(f.read())

        self.ISIN = temp['ISIN']
        self.InputFileName = temp['InputFileName']

        self.Parameters = CModelParams(temp['Parameters'])
        if temp['Coefficients'] is None:
            self.Coefficients = None
            self.Parameters.UseStandartZCYC = False
        else:
            self.Coefficients = CZCYCParams(temp['Coefficients'])

        self.Adjusts = None
        self.Datasets = None
        self.Macromodel = None

    def GetDataFromDB(self, server=None, homedir=None, encoding='utf8'):

        self.Datasets = CModelDatasets(self.ISIN)
        self.Macromodel = CMacroModel(cir_default)


class LocalTask(object):

    def __init__(self, res):

        self.result = res

    def __del__(self):

        self.result = None

    def ready(self):

        return True

    def get(self):

        return self.result


class LocalView(object):

    def apply_async(self, func, *args, **kwargs):

        return LocalTask(func(*args, **kwargs))


class LocalClient(object):

    max_procs = 1
    max_errors = 1

    def purge_everything(self):
        pass

    def close(self):
        pass

    def abort(self, *args, **kwargs):
        pass

    def load_balanced_view(self):
        return LocalView()

    def purge_results(self, *args, **kwargs):
        pass


class LocalUpdater(object):

    def put(self, *args, **kwargs):
        pass


class StatusUpdater(threading.Thread):

    def __init__(self, ms=5.0, loud=False, show_progress=True):

        super(StatusUpdater, self).__init__()

        self.upd_que = Queue.Queue()
        self.updateInterval = ms
        self.total = 1
        self.loud = loud
        self.exit_status = Errors.NoErrors
        self.show_progress = show_progress
        self.was_ever_stopped = False

        # --- self-kill flag ---
        self.alive = threading.Event()
        self.alive.set()

    def run(self):

        cnt = 0
        t0 = time.time()
        ts = t0
        t1 = t0

        while self.alive.isSet():

            try:
                item = self.upd_que.get()
                cnt += 1
                t0 = time.time()

                # --- None-message in queue is a signal to finalize calculation ---
                if item is None:
                    self.alive.clear()
                    self.send_est('%02d:%02d:%02d' % to_hrs(t0 - ts), 100.0 * float(cnt-1) / self.total, eol=cnt==self.total+1)
                    break
                # --- Any other type of message is an expetion code ---
                else:
                    # --- give some output and notify about progress ---
                    if t0 - t1 > self.updateInterval:
                        t1 = t0
                        secest = (self.total - cnt - 1) * (t0 - ts) / float(cnt)
                        Estimated = '%02d:%02d:%02d' % to_hrs(secest)
                        Percentage = 100.0 * float(cnt - 1) / self.total
                        self.send_est(Estimated, Percentage)
                    else:
                        pass

            # --- if no results, wait for some time ---
            except Queue.Empty:
                time.sleep(0.05)

            # --- if any errors, stop calculation and exit ---
            except Exception as e:
                self.send_err(description=repr(e))
                break

    def send_err(self, description=None, errorcode=Errors.ModelBaseError):

        self.exit_status = errorcode

        if self.loud:
            print 'Raised: %s' % description

    def send_est(self, est, per, eol=False):

        if self.loud:
            if eol:
                print '\r', 'Tot %s%s - OK' % (est, ' '*(12-12))
            else:
                print '\r', 'Est %s%s - %4.1f%%' % (est, ' '*(12-12), per),

    def stop(self, timeout=None):
        self.was_ever_stopped = True
        self.upd_que.put(None)
        threading.Thread.join(self, timeout)
        return self.exit_status

    def abort(self, timeout=None):
        self.alive.clear()
        threading.Thread.join(self, timeout)


def to_hrs(sec):
    hrs = int(sec / 3600.0)
    min = int((sec - hrs * 3600) / 60.0)
    return hrs, min, sec - min * 60 - hrs * 3600


def xparse(evalstr, ptt=PTT, alfabet=ALF):

    lst = PTT.findall(evalstr)
    kwargs = {}
    if len(lst) > 0:
        k = 0
        for el in set(lst):
            evalstr = evalstr.replace(el, alfabet[k])
            kwargs[alfabet[k]] = el[1:-1]
            k += 1
    else:
        kwargs[alfabet[0]] = evalstr
        evalstr = None

    return evalstr, kwargs


def accumulate(res_que=None, res_ptr=[], multipart=False, loud=False):

    if loud: print 'debug: accumulate proc'

    while res_que is not None:

        # --- check queue for item ---
        try:
            item = res_que.get()

            # --- None-message in queue is a signal to finalize calculation ---
            if item is None:
                if loud: print 'debug: accumulating process started'
                for i in range(len(res_ptr)):
                    if len(res_ptr[i]) > 0:
                        if isinstance(res_ptr[i][0], pd.core.frame.DataFrame):
                            # --- concatenate and group pandas dataframes ---
                            # sum isn't needed when loan cluster number is low enough and there is no slicing
                            res_ptr[i] = pd.concat(res_ptr[i])
                            # to enable when LLD slicing is on
                            if multipart:
                                res_ptr[i] = res_ptr[i].reset_index().groupby(res_ptr[i].index.names).sum()
                        elif isinstance(res_ptr[i][0], (str, unicode)):
                            res_ptr[i] = '\n'.join(res_ptr[i])
                        else:
                            # sum list of values
                            res_ptr[i] = sum(res_ptr[i])
                    else:
                        res_ptr[i] = None
                if loud: print 'debug: accumulating process ended'
                break
            # --- List-message in queue is a list of dataframes to combine ---
            elif isinstance(item, tuple) or isinstance(item, list):
                # --- create result dataframes if needed ---
                while len(res_ptr) < len(item):
                    res_ptr.append(None)
                for i in range(len(item)):
                    if res_ptr[i] is None:
                        res_ptr[i] = []
                    res_ptr[i].append(item[i])
            # --- Any other type of message is an exception code ---
            else:
                for i in range(len(res_ptr)):
                    res_ptr[i] = None
                raise Exception("Queue aborted (code %s)" % str(item))

        # --- if no results, wait for some time ---
        except Queue.Empty:
            if loud: print 'debug: waiting for queue (50ms)'
            time.sleep(0.05)

        # --- if any errors, stop calculation and exit ---
        except Exception as e:
            print 'Raised: %s' % str(e)
            break

    if loud: print 'debug: processed %d/%d items' % (len(res_ptr[0]), len(res_ptr), )


def worker(wid, func, x, *args, **kwargs):

    try:

        # --- run cashflow model ---
        answer = wid
        result = func(x, *args, **kwargs)

    except MemoryError as e:

        answer = -1
        result = repr(e)

    except Exception as e:

        answer = -2
        result = repr(e)

    return "localhost", answer, result


def runpar(func, xmap, *args, **kwargs):

    try:
        upd_queue = kwargs.pop('upd_queue')
    except:
        upd_queue = LocalUpdater()

    try:
        multipart = kwargs.pop('multipart')
    except:
        multipart = False

    try:
        loud = kwargs.pop('loud')
    except:
        loud = False

    try:
        rclient = kwargs.pop('rclient')
    except:
        rclient = None

    if rclient is None:
        if loud:
            print 'debug: non-parallel mode'
        rclient = LocalClient()
    dview = rclient.load_balanced_view()

    result = []
    msq_queue = Queue.Queue()
    parser = threading.Thread(target=accumulate, kwargs={'res_que': msq_queue, 'res_ptr': result, 'multipart': multipart, 'loud': loud})
    parser.daemon = True
    parser.start()

    if loud: print 'debug: processing {} jobs'.format(len(xmap))

    # --- running jobs ---
    queue_status = None
    jobs_queue = range(len(xmap))
    pending_jobs = {}
    while len(jobs_queue) > 0 or len(pending_jobs) > 0:
        # add new tasks
        while len(jobs_queue) > 0 and len(pending_jobs) < rclient.max_procs:
            job_id = jobs_queue.pop(0)  # i = [1 ... len(tsk_queue)]
            pending_jobs[job_id] = dview.apply_async(worker, job_id, func, xmap[job_id], *args, **kwargs)

        # check current running tasks
        ready_tasks = []
        for job_id in pending_jobs.keys():
            if pending_jobs[job_id].ready():
                ready_tasks.append(pending_jobs.pop(job_id))
                try:
                    host, answer, data = ready_tasks[-1].get()
                    if answer >= 0:
                        # --- save results or put job to queue ---
                        msq_queue.put(data)
                        upd_queue.put(True)
                    else:
                        # --- stop queue if any errors ---
                        raise Exception('%s: "%s"' % (host, str(data)))
                except Exception as e:
                    print "Job %d raised: %s" % (job_id, str(e))
                    queue_status = -1

        # # purge AsyncResult object
        # if len(ready_tasks) > 0:
        #     rclient.purge_results(jobs=ready_tasks)

        if not parser.isAlive():
            queue_status = -1

        # abort queue
        if queue_status == -1:
            for i in range(len(result)):
                result[i] = None
            result = None
            break

        # wait for some time
        time.sleep(0.05)

    if loud: print 'debug: finishing status={}'.format(queue_status)

    # --- send stop message for worker ---
    msq_queue.put(queue_status)
    parser.join()
    if loud: print 'debug: threads joined'

    # del pending_jobs
    # del dview

    if loud: print 'debug: memory cleared'

    return result


def profiling(func):

    def wrapper(*args, **kwargs):

        try:
            timings = kwargs['timings']
        except KeyError:
            timings = None

        if timings is None:
            return func(*args, **kwargs)
        else:
            # measure function timings
            t0 = time.clock()
            result = func(*args, **kwargs)
            t1 = time.clock()

            # define self full name
            if len(args) > 0 and args[0].__class__.__module__ == func.__module__:
                fname = func.__module__+':'+args[0].__class__.__name__+'.'+func.__name__
            else:
                fname = func.__module__+':'+func.__name__

            # save measures
            try:
                timings[fname] += t1 - t0
            except KeyError:
                timings[fname] = t1 - t0

            return result

    return wrapper


def debugging(func):

    def wrapper(*args, **kwargs):

        try:
            buf = kwargs['debug']
        except KeyError:
            buf = None

        if buf is None:
            return func(*args, **kwargs)
        else:
            # define self full name
            if len(args) > 0 and args[0].__class__.__module__ == func.__module__:
                fname = func.__module__+':'+args[0].__class__.__name__+'.'+func.__name__
            else:
                fname = func.__module__+':'+func.__name__

            buf.write(u'=== function call begin ===\n')
            buf.write(u'%s(*args, **kwargs)\n' % fname)
            if len(args) > 0:
                buf.write(u'--- args ---\n')
                for i in range(len(args)):
                    buf.write(u'args[%d] = %s\n' % (i, repr(args[i]).decode('utf8')))
            if len(kwargs) > 0:
                buf.write(u'--- kwargs ---\n')
                for key in kwargs.keys():
                    buf.write(u'kwargs[\'%s\'] = %s\n' % (key, repr(kwargs[key]).decode('utf8')))

            # get function output
            try:
                result = func(*args, **kwargs)
                buf.write(u'--- returns ---\n')
                buf.write(repr(result).decode('utf8'))
            except Exception as e:
                result = None
                buf.write(u'--- exception ---\n')
                buf.write(repr(e).decode('utf8'))
            finally:
                buf.write(u'\n')
            buf.write(u'=== function call end ===\n')

            return result

    return wrapper


# def debugging(func):
#
#     def wrapper(*args, **kwargs):
#
#         try:
#             buf = kwargs['debug']
#         except KeyError:
#             buf = None
#
#         if buf is None:
#             return func(*args, **kwargs)
#         else:
#             # define self full name
#             if len(args) > 0 and args[0].__class__.__module__ == func.__module__:
#                 fname = func.__module__+':'+args[0].__class__.__name__+'.'+func.__name__
#             else:
#                 fname = func.__module__+':'+func.__name__
#
#             print '=== function call begin ==='
#             print 'Call: %s(*args, **kwargs):' % fname
#             if len(args) > 0:
#                 print '--- args ---'
#                 for i in range(len(args)):
#                     print 'args[%d] = %s' % (i, repr(args[i]))
#             if len(kwargs) > 0:
#                 print '--- kwargs ---'
#                 for key in kwargs.keys():
#                     print 'kwargs[%s] = %s' % (key, repr(kwargs[key]))
#
#             # get function output
#             try:
#                 result = func(*args, **kwargs)
#                 print '--- sucsessful result ---'
#                 print repr(result)
#             except Exception as e:
#                 result = None
#                 print '--- exception raised ---'
#                 print repr(e)
#             print '=== function call end ==='
#
#             return result
#
#     return wrapper
