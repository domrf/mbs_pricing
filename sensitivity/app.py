# -*- coding: utf8 -*-
import wx
import wx.adv
import wx.aui
import wx.lib.scrolledpanel as scrolled

import pandas as pd
import numpy as np

import datetime as dt
from dateutil.relativedelta import relativedelta

import specs
import model
import func_and_var_plots
import macro_plot

from impdata import sql_macrohist, gen_std_json
from impdata.names import CFN

from estimation import EstimationSeries, Adjustments

nb_style = wx.aui.AUI_NB_DEFAULT_STYLE & ~ (wx.aui.AUI_NB_TAB_MOVE | wx.aui.AUI_NB_TAB_SPLIT |
                                            wx.aui.AUI_NB_CLOSE_ON_ALL_TABS | wx.aui.AUI_NB_CLOSE_ON_ACTIVE_TAB)

macro_cols = [CFN.ZCY, CFN.SNG, CFN.SCR, CFN.SPT, CFN.MS6,
              CFN.MIR, CFN.SP6, CFN.HPI, CFN.HPR]

macro_names = ['ZCY', 'SNG', 'SCR', 'SPT', 'MS6', 'MIR', 'SP6', 'HPI', 'HPR']


class ScrolledPanelForVariables(scrolled.ScrolledPanel):

    def __init__(self, parent, main_frame):

        scrolled.ScrolledPanel.__init__(self, parent, -1)

        self.main_frame = main_frame

        self.vbox_left = wx.BoxSizer(wx.VERTICAL)
        self.vbox_right = wx.BoxSizer(wx.VERTICAL)
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.empty_panel_for_new = EmptyPanelForNewVariable(self, self.main_frame)
        self.vbox_left.Add(self.empty_panel_for_new)

        self.hbox.Add(self.vbox_left)
        self.hbox.Add(self.vbox_right)
        self.SetSizer(self.hbox)
        self.SetAutoLayout(1)
        self.SetupScrolling(scroll_y=True)

        self.number_of_variables=0
        self.variables_id_list = []


class EmptyPanelForNewVariable(wx.Panel):

    def __init__(self, parent, main_frame):

        wx.Panel.__init__(self, parent, size=(594, 389.344537815126))

        self.picture = wx.Image(r'sensitivity\pics\NewVariable.png', wx.BITMAP_TYPE_ANY)
        self.picture = self.picture.Scale(594, 389.344537815126, wx.IMAGE_QUALITY_HIGH)
        self.picture = self.picture.ConvertToBitmap()
        self.button_new_variable = wx.BitmapButton(self, -1, self.picture, pos=(0, 0), size=(594, 389.344537815126))
        self.button_new_variable.Bind(wx.EVT_BUTTON, self.add_panel_with_new_variable)

        self.main_frame = main_frame
        self.is_empty = True

    def add_panel_with_new_variable(self, event):

        scrolled_panel = self.main_frame.scrolled_panel_for_variables
        num_of_vars = scrolled_panel.number_of_variables
        scrolled_panel.__dict__['var'+str(num_of_vars)] = PanelForVariable(scrolled_panel, self.main_frame, id=num_of_vars)

        # self.main_frame.estimation_series.series.append(None)
        # self.main_frame.estimation_series.models.append(None)
        # self.main_frame.estimation_series.panels.append(None)
        # self.main_frame.estimation_series.sql_results.append(None)
        # self.main_frame.estimation_series.base_results.append(None)
        # self.main_frame.estimation_series.modified_results.append(None)
        # self.main_frame.estimation_series.prepared_and_drawn.append(None)
        # self.main_frame.estimation_series.many_scenarios.append(None)

        if not bool(num_of_vars % 2):
            scrolled_panel.vbox_left.Hide(len(scrolled_panel.vbox_left.Children)-1)
            scrolled_panel.vbox_left.Remove(len(scrolled_panel.vbox_left.Children)-1)
            scrolled_panel.vbox_left.Add(scrolled_panel.__dict__['var'+str(num_of_vars)])
            scrolled_panel.vbox_left.Layout()
            scrolled_panel.vbox_right.Add(EmptyPanelForNewVariable(scrolled_panel, self.main_frame))
            scrolled_panel.vbox_right.Layout()
            scrolled_panel.number_of_variables += 1
        else:
            scrolled_panel.vbox_right.Hide(len(scrolled_panel.vbox_right.Children) - 1)
            scrolled_panel.vbox_right.Remove(len(scrolled_panel.vbox_right.Children)-1)
            scrolled_panel.vbox_right.Add(scrolled_panel.__dict__['var'+str(num_of_vars)])
            scrolled_panel.vbox_right.Layout()
            scrolled_panel.vbox_left.Add(EmptyPanelForNewVariable(scrolled_panel, self.main_frame))
            scrolled_panel.vbox_left.Layout()
            scrolled_panel.number_of_variables += 1
        scrolled_panel.hbox.Layout()
        scrolled_panel.SetAutoLayout(1)
        scrolled_panel.SetupScrolling(scroll_x=False, scroll_y=True, scrollToTop=False, scrollIntoView=False)

        base_adjust_module = self.main_frame.BigNoteBookForFunctions.GetCurrentPage().base_adjust_module

        for panel in scrolled_panel.Children:
            if not panel.is_empty:
                panel.variable_plot.current_base_adjust_module = base_adjust_module




class PanelWithNotebook(wx.Panel):

    ''' Панель-болванка, на которую прикрепляются блокноты (notebooks) с function_plot'ами.
        Технический элемент, используется при формировании структуры спецификаций внутри объекта MainFrame. '''

    def __init__(self, parent, main_frame, first_level_name = None, second_level_name = None, third_level_name = None,  fourth_level_name = None, create_function=False):

        wx.Panel.__init__(self, parent)
        self.SetBackgroundColour(wx.WHITE)

        self.main_frame = main_frame

        self.first_level_name = first_level_name
        self.second_level_name = second_level_name
        self.third_level_name = third_level_name
        self.fourth_level_name = fourth_level_name

        self.base_adjust_module = None
        self.modified_adjust_module = None

        # Если на шаге алгоритма по созданию структуры спецификаций нет запроса на создание графика-объекта FunctionPlot,
        # то панель использоваться как панель-болванка для вложения в нее следующего блокнота.
        if create_function is False:
            self.sizer = wx.BoxSizer(wx.HORIZONTAL)
            self.Notebook = wx.aui.AuiNotebook(self, style=nb_style)
            self.sizer.Add(self.Notebook, 1, flag=wx.EXPAND)
            self.SetSizer(self.sizer)

        # В ином случае, если на очередном шаге алгоритм привел нас во вкладку с конкретной переменной (типа
        # u'Сценарная ставка с лагом 6 мес.'), то нужно создавать для этой переменной график-объект FunctionPlot:
        else:
            y_add = 0
            if self.third_level_name == 'base': y_add = 28

            # Инициализация графика-объекта с виджетами для визуального представления спецификаций моделей калькулятора.
            self.function_plot = func_and_var_plots.FunctionPlot(self.main_frame, self, figsize=(5, 3.5), position=(80, 3 + y_add),
                                                                 first_level_name=self.first_level_name, second_level_name=self.second_level_name,
                                                                 third_level_name=self.third_level_name, fourth_level_name=self.fourth_level_name)
# </editor-fold>

class PanelForVariable(wx.Panel):

    ''' Панель-основа для визульаного представления оценок (объект VariablePlot). '''

    def __init__(self, parent, main_frame, id):

        wx.Panel.__init__(self, parent, size=(594, 389.344537815126), style=wx.SIMPLE_BORDER)
        self.SetBackgroundColour(wx.WHITE)

        self.main_frame = main_frame
        self.id = id

        # Флаг is_empty используется для того, чтобы в последующих алгоритмах отличать
        # PanelForVariable от EmptyPanelForNewVariable.
        self.is_empty = False

        # Инициализация графика-объекта с виджетами для визуального представления результатов оценки.
        # Все виджеты подключаются к панели через график-объект (у каждого такого графика-объета своя self.panel).
        self.variable_plot = func_and_var_plots.VariablePlot(self.main_frame, self, figsize=(5.84, 3.32), position=(2, 55))


class PanelForMacroPlot(wx.Panel):

    ''' Панель, которая становится основой для закладок в MainFrame.Notebook_for_MacroPlot.
        К этим панелям прикрепляются графики с макро (self.macro_plot). Таким образом,
        одна панель создается на одну макро-переменную. '''

    def __init__(self, parent, main_frame, macro_series, last_hist_date, col):

        wx.Panel.__init__(self, parent)
        self.SetBackgroundColour(wx.WHITE)

        self.main_frame = main_frame
        self.last_hist_date = last_hist_date

        # Инициализация графика-объекта c виджетами для визуализации и
        # управления одной переменной из макро-блока. Т.о. у каждой панели
        # свой график-объект, их количество равно количеству переменных макро-блока.
        self.macro_plot = macro_plot.MacroPlot(self.main_frame, self, figsize=(6.7, 3.4), position=(55, 45),
                                               macro_series=macro_series, last_hist_date=self.last_hist_date, column=col)


# class UniversalPanel(wx.Panel):
#
#     ''' Универсальная панель, которая имеет несколько видов (type):
#         -> '''
#
#     def __init__(self, parent, main_frame, type, **kwargs):
#
#         wx.Panel.__init__(self, parent)
#         self.SetBackgroundColour(wx.WHITE)
#
#         self.main_frame = main_frame
#



class MainFrame(wx.Frame):

    '''
         Основной класс приложения.
         Основные attributes, отвечающие за работу всего приложения:
         -> self.estimation_series                --->>> объект класса EstimationSeries, в котором хранятся и из которого управляются все оценки-объкты;
         -> self.BigNoteBookForFunctions          --->>> большой блокнот, отражающий стрктуру спецификаций, к котормоу привязаны все function_plot'ы
                                                         (графики-объекты с виджетами для визуального представления спецификаций моделей калькулятора);
         -> self.Notebook_for_MacroPlot           --->>> блокнот, каждая вкладка которого отвечает за визуальную презентацию одной макро-переменной
                                                         (посредством графиков-объектов MacroPlot);
         -> self.scrolled_panel_for_variables     --->>> панель с полосой прокрутки (scrollbar), хранилище всех VariablePlot'ов;
         -> self.current_base_adjust_module       --->>> ссылка на модуль с базовой спецификацией, по которой пересчитываются оценки;
         -> self.current_modified_adjust_module   --->>> ссылка на модуль с модифицированной спецификацией, по которой пересчитываются оценки;
         -> self.macro_with_hist                  --->>> ссылка на результирующую таблицу макро-блока, по которой считаются оценки
                                                         (самих таблиц две: self.macro_one_sim_with_hist_copy и self.macro_mean_with_hist_copy,
                                                         в зависимости от выбранного режима макро);
    '''

    def __init__(self):

        super(MainFrame, self).__init__(parent=None, title=u'Калькулятор ИЦБ: Анализ чувствительности')
        self.Maximize(True)

        self.list_of_boxes_for_RMSE = []

        self.mainpanel = wx.Panel(self, wx.ID_ANY)
        self.scrolled_panel_for_variables = ScrolledPanelForVariables(self.mainpanel, self)

        self.main_horizontal_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.vertical_sizer_for_notebooks = wx.BoxSizer(wx.VERTICAL)

        # <editor-fold desc="Notebook for Macro">
        self.macro_many_sim = generate_macro(date=dt.datetime.today().date().strftime('%Y-%m-%d'), seed=123, numb_of_sim=50)
        self.macro_mean_with_hist, self.last_hist_date = add_hist(self.macro_many_sim, many=True)

        self.macro_one_sim = generate_macro(date=dt.datetime.today().date().strftime('%Y-%m-%d'), seed=123, numb_of_sim=1)
        self.macro_one_sim_with_hist, self.last_hist_date = add_hist(self.macro_one_sim)

        self.macro_mean_with_hist.set_index(CFN.DAT, inplace=True)
        self.macro_one_sim_with_hist.set_index(CFN.DAT, inplace=True)

        self.macro_mean_with_hist_copy = self.macro_mean_with_hist.copy()
        self.macro_one_sim_with_hist_copy = self.macro_one_sim_with_hist.copy()

        # create a reference only
        self.macro_with_hist = self.macro_one_sim_with_hist

        self.default_left_date = pd.to_datetime(self.macro_with_hist.index[0]).date() - relativedelta(months=12)
        self.default_right_date = pd.to_datetime(self.macro_with_hist.index[-1]).date() + relativedelta(months=12)

        self.history_left_date = pd.to_datetime(self.macro_with_hist.index[0]).date() - relativedelta(months=1)
        self.history_right_date = pd.to_datetime(self.last_hist_date).date() + relativedelta(months=36)

        self.Notebook_for_MacroPlot = wx.aui.AuiNotebook(self.mainpanel, style=nb_style)
        for col, name in zip(macro_cols, macro_names):
            self.Notebook_for_MacroPlot.AddPage(PanelForMacroPlot(self.Notebook_for_MacroPlot, self, self.macro_with_hist[col], self.last_hist_date, col), name)
        # </editor-fold>

        # <editor-fold desc="Notebook for functions">
        self.BigNoteBookForFunctions = wx.aui.AuiNotebook(self.mainpanel, style=nb_style)
        self.BigNoteBookForFunctions.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CHANGED, self.when_first_level_tab_changed)

        self.first_level_tabs_as_panels = []
        self.create_tabs_for_specs()
        # </editor-fold>

        self.vertical_sizer_for_notebooks.Add(self.BigNoteBookForFunctions, 10, flag=wx.EXPAND)
        self.vertical_sizer_for_notebooks.Add(self.Notebook_for_MacroPlot, 9, flag=wx.EXPAND)

        self.main_horizontal_sizer.Add(self.scrolled_panel_for_variables, 13, flag=wx.EXPAND)
        self.main_horizontal_sizer.Add(self.vertical_sizer_for_notebooks, 8, flag=wx.EXPAND)
        self.mainpanel.SetSizer(self.main_horizontal_sizer)

        self.current_base_adjust_module = self.BigNoteBookForFunctions.GetCurrentPage().base_adjust_module
        self.current_modified_adjust_module = self.BigNoteBookForFunctions.GetCurrentPage().modified_adjust_module

        self.estimation_series = EstimationSeries(self)
        self.synchronize_mode_on = False
        self.you_may_continue = False

    def when_first_level_tab_changed(self, event):

        ''' MainFrame должен понимать, по какой спецификации пользователь хочет
            запускать оценки. Выбор спецификаций осуществляется путем переключения
            закладок на верхнем уровне структуры спецификаций. '''

        self.current_base_adjust_module = self.BigNoteBookForFunctions.GetCurrentPage().base_adjust_module
        self.current_modified_adjust_module = self.BigNoteBookForFunctions.GetCurrentPage().modified_adjust_module

    def update_RMSE_value(self, value):

        for box in self.list_of_boxes_for_RMSE:
            box.SetLabelText(str(round(value, 5)))



    def create_tabs_for_specs(self):

        ''' Функция отвечает за корректную систематизицию вкладок для function_plot'ов.
            Главный принцип -- структура должна полностью соответствовать текущему
            содержимому модуля specs, который задается через specs.__init__.py. '''

        # Создадим лист, элементы которого - листы из 4 строк (strings), соответствующие всем
        # возможным комбинациям вкладок (то есть всем существующим комбинациям
        # спецификация -> модель -> группа переменных* -> переменная).
        # К примеру: [['prime_m', 'Partial', 'comb_adjusts', u'Сценарная ставка с лагом 6 мес.'], [...], ... ]
        # *Базовые функции здесь пока игнорируются.
        list_of_specs = []
        for key in specs.adjust_dict.keys():
            module = specs.adjust_dict[key]
            for dict_name in ['const_adjusts', 'time_adjusts', 'comb_adjusts']:
                dict_with_frames = getattr(module, dict_name)
                for df_name in dict_with_frames.keys():
                    df = dict_with_frames[df_name]
                    for column in df.columns[1:]:
                        small_list = ["%d: %s" % (key, module.__name__), column, dict_name, df_name]
                        list_of_specs.append(small_list)

        df = pd.DataFrame(list_of_specs, columns=['level_1', 'level_2', 'level_3', 'level_4'])

        first_level_tab_names = df['level_1'].unique()
        second_level_tab_names = []
        third_level_tab_names = []
        fourth_level_tab_names = []

        for first_level_tab_name in first_level_tab_names:
            second_level_names_for_this_first_tab = df[df['level_1'] == first_level_tab_name]['level_2'].unique()
            second_level_tab_names.append(second_level_names_for_this_first_tab)
            third_level_tab_names_for_second_tabs_in_this_first_tab = []
            fourth_level_tab_names_for_second_tabs_in_this_first_tab = []
            for second_level_name in second_level_names_for_this_first_tab:
                third_level_names_for_this_second_tab = \
                    df[(df['level_1'] == first_level_tab_name) & (df['level_2'] == second_level_name)]['level_3'].unique()
                third_level_tab_names_for_second_tabs_in_this_first_tab.append(list(third_level_names_for_this_second_tab))
                fourth_level_names_for_third_tabs_in_this_second_tab = []
                for third_level_name in third_level_names_for_this_second_tab:
                    fourth_level_names_for_this_third_level_tab = df[
                        (df['level_1'] == first_level_tab_name) & (df['level_2'] == second_level_name) & (df['level_3'] == third_level_name)]['level_4'].unique()
                    fourth_level_names_for_third_tabs_in_this_second_tab.append(
                        list(fourth_level_names_for_this_third_level_tab))
                fourth_level_tab_names_for_second_tabs_in_this_first_tab.append(
                    fourth_level_names_for_third_tabs_in_this_second_tab)
            third_level_tab_names.append(third_level_tab_names_for_second_tabs_in_this_first_tab)
            fourth_level_tab_names.append(fourth_level_tab_names_for_second_tabs_in_this_first_tab)

        for tab_name in first_level_tab_names:
            self.BigNoteBookForFunctions.AddPage(PanelWithNotebook(self.BigNoteBookForFunctions, self, tab_name), tab_name)

        for child in self.BigNoteBookForFunctions.Children:
            if isinstance(child, PanelWithNotebook):
                self.first_level_tabs_as_panels.append(child)

        for tab in self.first_level_tabs_as_panels:
            module_name = tab.first_level_name.partition('specs.')[2]
            tab.base_adjust_module = Adjustments(getattr(specs, module_name))
            tab.modified_adjust_module = Adjustments(getattr(specs, module_name))

        for first_level_tab, second_level_names, third_level_names, fourth_level_names in zip(self.first_level_tabs_as_panels, second_level_tab_names, third_level_tab_names, fourth_level_tab_names):
            for tab_name in second_level_names:
                first_level_tab_name = first_level_tab.first_level_name
                first_level_tab.Notebook.AddPage(PanelWithNotebook(first_level_tab, self, first_level_tab_name, tab_name), tab_name)
                first_level_tab.sizer.Layout()

            second_level_tabs_as_panels = []
            for child in first_level_tab.Notebook.Children:
                if isinstance(child, PanelWithNotebook):
                    second_level_tabs_as_panels.append(child)

            for second_level_tab, third_level_tab_names, fourth_level_tab_names in zip(second_level_tabs_as_panels, third_level_names, fourth_level_names):
                second_level_tab_name = second_level_tab.second_level_name
                second_level_tab.Notebook.AddPage(PanelWithNotebook(second_level_tab, self, first_level_tab_name, second_level_tab_name, 'base', create_function=True), 'base')
                for tab_name in third_level_tab_names:
                    second_level_tab.Notebook.AddPage(PanelWithNotebook(second_level_tab, self, first_level_tab_name, second_level_tab_name, tab_name), tab_name)
                    second_level_tab.sizer.Layout()

                third_level_tabs_as_panels = []
                for child in second_level_tab.Notebook.Children:
                    if isinstance(child, PanelWithNotebook):
                        third_level_tabs_as_panels.append(child)

                for third_level_tab, fourth_level_names in zip(third_level_tabs_as_panels[1:], fourth_level_tab_names):
                    third_level_tab_name = third_level_tab.third_level_name
                    for tab_name in fourth_level_names:
                        third_level_tab.Notebook.AddPage(PanelWithNotebook(third_level_tab, self, first_level_tab_name, second_level_tab_name, third_level_tab_name, tab_name, create_function=True), tab_name)


def generate_macro(date, seed, numb_of_sim):

    ''' Не более, чем разовый запуск блока model_cir. '''

    for_json = gen_std_json(curdate=date, bondisin='RU000A0ZYJT2')
    for_json['Parameters']['NumberOfMonteCarloScenarios'] = numb_of_sim
    for_json['Parameters']['NumberOfEventSimulations'] = 2000
    # for_json['Parameters']['RandomNumberSeed'] = seed

    inputs = model.common.CParams(for_json)
    inputs.GetDataFromDB()

    np.random.seed(inputs.Parameters.RandomNumberSeed)
    xmap = np.zeros((inputs.Parameters.NumberOfMonteCarloScenarios, 2), dtype=np.uint64)
    xmap[:, 0] = np.arange(inputs.Parameters.NumberOfMonteCarloScenarios)
    xmap[:, 1] = np.random.randint(1, high=1001, size=inputs.Parameters.NumberOfMonteCarloScenarios)
    for i in range(1, inputs.Parameters.NumberOfMonteCarloScenarios):
        xmap[i, 1] += xmap[i - 1][1]

    scr_df = model.cir.run(inputs, delta=0, max_years=15, seeds=list(xmap[:, 1]), debug=None, verbose=None, timings=None)

    return scr_df


def add_hist(scr_df, many=False):

    macrohist_df = sql_macrohist()
    macrohist_df.columns = [CFN.DAT, CFN.SPT, CFN.MS6, CFN.HPR, CFN.MIR]
    macrohist_df[CFN.SPT] = 100 * (np.exp(macrohist_df[CFN.SPT].values / 100) - 1)

    df = scr_df.copy(deep=True)
    df.reset_index(inplace=True)
    df = df.groupby(by=CFN.SCN).apply(lambda x: convert_macro(x, macrohist_df))

    df[CFN.SCN] = df[CFN.SCN].fillna(method='backfill').astype('int')
    df.reset_index(drop=True, inplace=True)
    df.set_index([CFN.SCN, CFN.DAT], inplace=True)

    last_hist_date = macrohist_df[CFN.DAT].values[-1]
    df.reset_index(inplace=True)

    if many:
        df = df.groupby(CFN.DAT).mean().reset_index()
        df[CFN.SCN] = int(0)

    return df, last_hist_date


def convert_macro(df, hist_df):

    hist_df[CFN.SCN] = df[CFN.SCN].values[0]

    df = pd.concat([hist_df, df[df.iloc[:, 1] > hist_df[CFN.DAT].values[-1]]])

    df[CFN.HPI] = df[CFN.HPR].values.cumprod()
    df[CFN.SCR] = df[CFN.SPT].values
    df[CFN.SP6] = df[CFN.SPT].shift(6)

    # !!! Отсюда ведется управление длины отображающейся истории !!!
    # !!! По умолчанию start = 124 будет означать, что исторический ряд будет дальше везде начинаться с 01.01.2016 !!!
    start = 124
    df = df[start:]

    df[CFN.SNG] = np.tile(np.arange(1,13), len(df)/12 + 1)[:len(df)]
    df.fillna(6.5, inplace=True)

    return df


def main():

    app = wx.App()
    frame = MainFrame()
    frame.Show()
    app.MainLoop()


