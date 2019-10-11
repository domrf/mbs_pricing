# -*- coding: utf8 -*-
import os
import Queue
import threading
import numpy as np
import pandas as pd

import model
import reader

import impdata
import wx

EVT_RESULT_BLACK = wx.NewId()
EVT_RESULT_GREY = wx.NewId()
EVT_RESULT_GREEN = wx.NewId()
EVT_RESULT_MIN = wx.NewId()


class ResultContainer(object):

    def __init__(self):

        self.result = None

        # --- self-kill flag ---
        self.alive = threading.Event()
        self.alive.set()

    def __call__(self, result):

        self.result = result
        self.alive.clear()

    def get_result(self):

        while self.alive.isSet():
            time.sleep(0.001)

        return self.result


class ResultEvent(wx.PyEvent):

    def __init__(self, evt_id, data):
        """Init Result Event."""
        wx.PyEvent.__init__(self)

        self.SetEventType(evt_id)
        self.data = data


class ResultCallback(ResultContainer):

    def __init__(self, wxObject=None, wxEventId=None):

        super(ResultCallback, self).__init__()

        # --- wx events ---
        self.wxObject = wxObject
        self.wxEventId = wxEventId

    def __call__(self, result):

        if self.wxObject is not None:
            wx.PostEvent(self.wxObject, ResultEvent(self.wxEventId, result))

        super(ResultCallback, self).__call__(result)


class Adjustments(object):

    def __init__(self, module_ptr):

        self.PartChange = module_ptr.PartChange

        self.Default = module_ptr.Default
        self.Prepayment = module_ptr.Prepayment
        self.Partial = module_ptr.Partial
        self.PartSize = module_ptr.PartSize

        # self.debug_name = module_ptr.debug_name

        self.const_adjusts = {}
        self.time_adjusts = {}
        self.comb_adjusts = {}

        self.base = module_ptr.base.copy(deep=True)

        for key in module_ptr.const_adjusts.keys():
            self.const_adjusts[key] = module_ptr.const_adjusts[key].copy(deep=True)

        for key in module_ptr.time_adjusts.keys():
            self.time_adjusts[key] = module_ptr.time_adjusts[key].copy(deep=True)

        for key in module_ptr.comb_adjusts.keys():
            self.comb_adjusts[key] = module_ptr.comb_adjusts[key].copy(deep=True)


class EstimationMessage(object):
    """ A command to the client thread.
        Each command type has its associated data:
    """
    START, RUN, EXEC, MIN = range(4)

    def __init__(self, type, data=((), {}), callback=None):
        self.type = type
        self.data = data
        self.callback = (lambda *args, **kwargs: None) if callback is None else callback


class EstimationInterface(threading.Thread):

    def __init__(self, cmd_q=None):

        super(EstimationInterface, self).__init__()

        # --- command and reply queues ---
        self.cmd_q = Queue.Queue() if cmd_q is None else cmd_q

        # create model object
        self.model_obj = model.ModelManager(rclient=None, loud=True, show_progress=False)

        # --- self-kill flag ---
        self.alive = threading.Event()
        self.alive.set()

        # --- interface handlers ---
        self.handlers = {
            EstimationMessage.START: self._prepare,
            EstimationMessage.RUN: self._runcalc,
            EstimationMessage.EXEC: self._runfunc,
        }

    def run(self):

        while self.alive.isSet():
            try:
                # Queue.get with timeout to allow checking self.alive
                cmd = self.cmd_q.get(True, 0.1)
                cmd.callback(self.handlers[cmd.type](*cmd.data[0], **cmd.data[1]))
            except Queue.Empty as e:
                # --- if no commands then run timers ---
                continue

    def join(self, timeout=None):

        self.alive.clear()
        threading.Thread.join(self, timeout)

    def _prepare(self, for_json):

        # read input params
        self.model_obj.LoadParams(fname=for_json, mods=['CalculationOfEffectiveIndicators=0'])

        # load lld data
        self.model_obj.LoadLLD(data=pd.read_csv(for_json['InputFileName'], sep='\t', encoding='utf8', parse_dates=True))

        # initiate calculation queue
        self.model_obj.Initialize()

    def _runcalc(self, **kwargs):

        # recompute model with new adjustments
        self.model_obj.Run(target=model.adjustsrun, **kwargs)

        mbs_c = self.model_obj.lst_c[0]
        res_c = self.model_obj.lst_c[1]

        # clean_up_date_index = np.argmax(mbs_c['CF'][1:].values == 0) - 1
        # clean_up_date = mbs_c['CF'][1:].index[clean_up_date_index][1]

        res_c[model.CFN.CPR] = 1 - (1 - ((res_c[model.CFN.ADV] - res_c[model.CFN.PDL]) / res_c[model.CFN.DBT])) ** 12
        res_c[model.CFN.CDR] = 1 - (1 - (res_c[model.CFN.PDL] / res_c[model.CFN.DBT])) ** 12

        res_c[model.CFN.NUM] *= 10**-3
        res_c[[model.CFN.DBT, model.CFN.AMT, model.CFN.ADV, model.CFN.YLD, model.CFN.PLN, model.CFN.PRT, model.CFN.FLL, model.CFN.PDL]] *= 10**-9

        return res_c

    def _runfunc(self, func, *args, **kwargs):

        return func(*args, **kwargs)


class Estimation(object):

    def __init__(self):
        self._est = EstimationInterface()
        self._already_started = False

    def start_thread(self):
        if not self._already_started:
            self._est.start()
            self._already_started = True

    def prepare(self, *args, **kwargs):
        callback = kwargs.pop('callback') if 'callback' in kwargs else ResultContainer()
        self.start_thread()
        self._est.cmd_q.put(EstimationMessage(type=EstimationMessage.START, data=(args, kwargs), callback=callback))
        return callback

    def runcalc(self, *args, **kwargs):
        callback = kwargs.pop('callback') if 'callback' in kwargs else ResultContainer()
        self._est.cmd_q.put(EstimationMessage(type=EstimationMessage.RUN, data=(args, kwargs), callback=callback))
        return callback

    def runfunc(self, *args, **kwargs):
        callback = kwargs.pop('callback') if 'callback' in kwargs else ResultContainer()
        self._est.cmd_q.put(EstimationMessage(type=EstimationMessage.EXEC, data=(args, kwargs), callback=callback))
        return callback

    # def runtest(self, *args, **kwargs):
    #     callback = kwargs.pop('callback') if 'callback' in kwargs else ResultContainer()
    #     self.start_thread()
    #     self._est.cmd_q.put(EstimationMessage(type=EstimationMessage.EXEC, data=(args, kwargs), callback=callback))
    #     return callback


class MinimizationInterface(threading.Thread):

    def __init__(self, parent, cmd_q=None):

        super(MinimizationInterface, self).__init__()

        # --- self-kill flag ---
        self.alive = threading.Event()
        self.alive.set()

        # --- command and reply queues ---
        self.cmd_q = Queue.Queue() if cmd_q is None else cmd_q

        self.parent = parent

        # --- interface handlers ---
        self.handlers = {
            # EstimationMessage.MIN: self._runmin,
        }

    def run(self):

        while self.alive.isSet():
            try:
                # Queue.get with timeout to allow checking self.alive
                cmd = self.cmd_q.get(True, 0.1)
                cmd.callback(self.handlers[cmd.type](*cmd.data[0], **cmd.data[1]))
            except Queue.Empty as e:
                # --- if no commands then run timers ---
                continue

    def join(self, timeout=None):

        self.alive.clear()
        threading.Thread.join(self, timeout)

    # def _runmin(self, x0, column):
    #
    #     def fmin(x, srs, col):
    #
    #         # устанавливаем новые значения x
    #         # ...
    #
    #         # делаем запрос на пересчет modified
    #         cbk_list = srs.update_modified()
    #
    #         # делаем блокирующий опрос результатов и подсчитываем rmse
    #         total_rmse = 0.0
    #         for id in cbk_dict:
    #             res = cbk_dict[id].get_result()
    #             total_rmse += srs.only_rmse(srs.sql_results[id], res, col)
    #
    #         return total_rmse
    #
    #     # запускаем процедуру минимизации, параметром выступает ссылка на объект-родитель EstimationSeries
    #     r = minimize(fmin, x0, args=(self.parent, column), method='Nelder-Mead')
    #
    #     # возвращаем результат минимизации ввиде callback-объекта
    #     return r.x

class EstimationSeries(object):

    def __init__(self, main_frame):

        self.main_frame = main_frame

        self.series = []
        self.models = []
        self.panels = []

        self.sql_results = []
        self.base_results = []
        self.modified_results = []

        self.prepared_and_drawn = []
        self.many_scenarios = []

        self.total_rmse = 0.0

        self._min = MinimizationInterface(self)
        self._min.start()

    def runmin(self, *args, **kwargs):
        #
        # ''' Эта процедура запускает процесс минимизации в дочернем процессе, результаты помещаются в callback.
        #     Пример использования:
        #         self.runmin(x0, column=model.CFN.CPR, callback=ResultCallback(some_panel, EVT_RESULT_MIN))
        # '''
        # callback = kwargs.pop('callback') if 'callback' in kwargs else ResultContainer()
        # self._min.cmd_q.put(EstimationMessage(type=EstimationMessage.MIN, data=(args, kwargs), callback=callback))
        #
        # return callback
        pass

    def add_blank_model(self):

        # model object and connected panel
        self.series.append(None)
        self.models.append(None)
        self.panels.append(None)

        # results to draw
        self.sql_results.append(None)
        self.base_results.append(None)
        self.modified_results.append(None)

        # some parameters
        self.prepared_and_drawn.append(False)  # deprecated
        self.many_scenarios.append(False)

    def refill_model(self, curdate, bondisin, many_scenarios, panel):

        # model object and connected panel
        self.series[panel.id] = (curdate, bondisin)
        if self.models[panel.id] is None:
            self.models[panel.id] = Estimation()
        self.panels[panel.id] = panel

        # results to draw
        self.sql_results[panel.id] = None
        self.base_results[panel.id] = None
        self.modified_results[panel.id] = None

        # some parameters
        self.prepared_and_drawn[panel.id] = False  # deprecated
        self.many_scenarios[panel.id] = many_scenarios

    def _prepare_model(self, id):

        if self.models[id] is None:
            return

        kwargs = {'for_json': impdata.gen_std_json(*self.series[id])}
        kwargs['for_json']['InputFileName'] = r'impdata\lld_database\LLD_%s_%s.csv' % (kwargs['for_json']['ISIN'], kwargs['for_json']['Parameters']['ActualLLDDate'])
        kwargs['for_json']['Parameters']['NumberOfMonteCarloScenarios'] = 1 if not self.many_scenarios[id] else 50
        kwargs['for_json']['Parameters']['NumberOfEventSimulations'] = 2000

        # по нажатию кнопки загружаем модель (на этом этапе оповещение о загрузке не требуется, callback=None)
        self.models[id].prepare(**kwargs)

    def _update_hist(self, id):

        curdate, bondisin = self.series[id]

        kwargs = {'func': impdata.sql_pool_flows, 'isin': bondisin}

        if self.panels[id] is not None:
            kwargs['callback'] = ResultCallback(self.panels[id], EVT_RESULT_BLACK)

        # Делаем запрос за историей в sql и просим оповестить панель self.panels[id] событием EVT_RESULT_BLACK
        self.models[id].runfunc(**kwargs)

    def _update_base(self, id):

        if self.models[id] is None:
            return

        kwargs = {'adjustments': self.main_frame.current_base_adjust_module}

        if self.panels[id] is not None:
            kwargs['callback'] = ResultCallback(self.panels[id], EVT_RESULT_GREY)

        if not self.many_scenarios[id]:
            kwargs['scr_df'] = self.prepare_scr(self.main_frame.macro_with_hist)

        # делаем запрос на серый график и просим оповестить панель self.panels[id] событием EVT_RESULT_GREY
        self.models[id].runcalc(**kwargs)

    def _update_modified(self, id):

        ''' Функция используется при минимизации. На каждом шаге итерации в минимизации осуществляется
            запрос на перерасчет всех моделей с одним сценарием (с последующей отрисовкой графиков).
            Данная функция отвечает за этот запрос. '''

        if self.models[id] is None:
            return

        kwargs = {'adjustments': self.main_frame.current_modified_adjust_module}

        if self.panels[id] is not None:
            kwargs['callback'] = ResultCallback(self.panels[id], EVT_RESULT_GREEN)

        if not self.many_scenarios[id]:
            kwargs['scr_df'] = self.prepare_scr(self.main_frame.macro_with_hist)

        # делаем запрос на серый график и просим оповестить панель self.panels[id] событием EVT_RESULT_GREY
        return self.models[id].runcalc(**kwargs)

    def prepare_model(self, id=None):

        if id is None:
            for id in range(len(self.models)):
                self._prepare_model(id)
        else:
            self._prepare_model(id)

    def update_hist(self, id=None):

        if id is None:
            for id in range(len(self.models)):
                self._update_hist(id)
        else:
            self._update_hist(id)

    def update_base(self, id=None):

        if id is None:
            for id in range(len(self.models)):
                self._update_base(id)
        else:
            self._update_base(id)

    def update_modified(self, id=None):

        cbk_list = {}
        mid_list = range(len(self.models)) if id is None else [id]
        for id in mid_list:
            cbk = self._update_modified(id)
            # TODO в список результатов собираются только случаи 1 сценария
            if not self.many_scenarios[id]:
                cbk_list[id] = cbk

        return cbk_list

    def only_rmse(self, hst, res, column):

        idx = sorted(list(set(hst.index.values) & set(res.index.values)))
        if len(idx) > 0:
            return (np.nansum((hst.loc[idx, column].values - res.loc[idx, column].values) ** 2) / len(idx))**0.5
        else:
            return 0.0

    # deprecated
    def rmse(self, id, column):

        if self.models[id] is None or self.many_scenarios[id]:
            return 0.0

        if self.sql_results[id] is None or self.modified_results[id] is None:
            return 0.0

        return self.only_rmse(self.sql_results[id], self.modified_results[id], column)

    # deprecated
    def calculate_total_rmse(self, column):

        total_rmse = 0.0

        for id in range(len(self.models)):
            total_rmse += self.rmse(id, column)

        return total_rmse

    def prepare_scr(self, df):
        df0 = df.copy(deep=True)
        df0.reset_index(inplace=True)
        df0.set_index([model.CFN.SCN, model.CFN.DAT], inplace=True)

        return df0
