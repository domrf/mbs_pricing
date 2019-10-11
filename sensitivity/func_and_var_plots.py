# -*- coding: utf8 -*-
import wx
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import model
from interactive import MovingCursorInsidePlot
from impdata import get_avail_bonds
import types
import copy


from estimation import EVT_RESULT_BLACK, EVT_RESULT_GREY, EVT_RESULT_GREEN


bonds = get_avail_bonds()
bonds.set_index('Name', inplace=True)

months = [u'Январь', u'Февраль', u'Март', u'Апрель',
          u'Май', u'Июнь', u'Июль', u'Август',
          u'Сентябрь', u'Октябрь', u'Ноябрь', u'Декабрь']


res_columns = ['NUM', 'DBT', 'YLD', 'AMT',
               'ADV', 'PLN', 'PRT', 'FLL',
               'PDL', 'CPR', 'CDR']

columns_dict = {'NUM': model.CFN.NUM,
                'DBT': model.CFN.DBT,
                'YLD': model.CFN.YLD,
                'AMT': model.CFN.AMT,
                'ADV': model.CFN.ADV,
                'PLN': model.CFN.PLN,
                'PRT': model.CFN.PRT,
                'FLL': model.CFN.FLL,
                'PDL': model.CFN.PDL,
                'CPR': model.CFN.CPR,
                'CDR': model.CFN.CDR}




class FunctionPlot(object):

    def __init__(self, main_frame, panel, figsize, position, first_level_name, second_level_name, third_level_name, fourth_level_name):

        self.main_frame = main_frame

        self.figsize = figsize
        self.x_pos, self.y_pos = position
        self.panel = panel

        self.first_level_name = first_level_name
        self.second_level_name = second_level_name
        self.third_level_name = third_level_name
        self.fourth_level_name = fourth_level_name

        self.figure = mpl.figure.Figure(figsize=self.figsize)
        self.canvas = FigureCanvas(panel, -1, self.figure)
        self.canvas.SetPosition(position)
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.075, top=0.95)

        self.ax = self.figure.add_subplot(111)
        self.ax.grid(linestyle='--', lw=0.4, alpha=0.5, zorder=0)
        self.ax.axvline(0, c='black', lw=0.4, alpha=0.6)
        self.ax.axhline(0, c='black', lw=0.4, alpha=0.6)
        self.ax.xaxis.set_tick_params(labelsize=8)
        self.ax.yaxis.set_tick_params(labelsize=8)

        for tab in self.main_frame.first_level_tabs_as_panels:
            if tab.first_level_name == self.first_level_name:
                self.base_adjust_module = tab.base_adjust_module
                self.modified_adjust_module = tab.modified_adjust_module

        if self.third_level_name is not 'base':

            self.base_adjust_dict = getattr(self.base_adjust_module, self.third_level_name)
            self.base_adjust_frame = self.base_adjust_dict[self.fourth_level_name]

            self.modified_adjust_dict = getattr(self.modified_adjust_module, self.third_level_name)
            self.modified_adjust_frame = self.modified_adjust_dict[self.fourth_level_name]

        elif self.third_level_name is 'base':

            self.base_adjust_frame = self.base_adjust_module.base
            self.modified_adjust_frame = self.modified_adjust_module.base

        self.first_column = self.base_adjust_frame.columns[0]
        self.adjust_column = self.second_level_name

        self.base_x = self.base_adjust_frame[self.first_column]
        self.base_y = self.base_adjust_frame[self.adjust_column]

        self.modified_x = self.modified_adjust_frame[self.first_column]
        self.modified_y = self.modified_adjust_frame[self.adjust_column]

        self.factor_base_plot, = self.ax.plot(self.base_x, self.base_y, c='gray', alpha=0.4, lw=2, zorder=0, ls='-')
        self.new_factor_plot, = self.ax.plot(self.modified_x, self.modified_y, c='g', alpha=0.8, lw=1.5, zorder=1)
        self.ax.set_xlim([self.base_x.min(), self.base_x.max()])
        self.ax.set_ylim([self.base_y.min(), self.base_y.max()])

        self.panel.box_y_top = wx.SpinCtrlDouble(panel, pos=(self.x_pos - 60, self.y_pos + 10), size=(55, 20), min=-1000, max=1000, inc=0.05, initial=round(self.base_y.max(),2))
        self.panel.box_y_bottom = wx.SpinCtrlDouble(panel, pos=(self.x_pos - 60, self.y_pos + 315), size=(55, 20), min=-1000, max=1000, inc=0.05, initial=round(self.base_y.min(),2))
        self.panel.box_x_left = wx.SpinCtrlDouble(panel, pos=(self.x_pos + 20, self.y_pos + 355), size=(55, 20), min=-1000, max=1000, inc=0.05, initial=round(self.base_x.min(),2))
        self.panel.box_x_right = wx.SpinCtrlDouble(panel, pos=(self.x_pos + 470, self.y_pos + 355), size=(55, 20), min=-1000, max=1000, inc=0.05, initial=round(self.base_x.max(),2))

        self.panel.box_y_top.Bind(wx.EVT_SPINCTRLDOUBLE, self.onEnter_y_top)
        self.panel.box_y_bottom.Bind(wx.EVT_SPINCTRLDOUBLE, self.onEnter_y_bottom)
        self.panel.box_x_left.Bind(wx.EVT_SPINCTRLDOUBLE, self.onEnter_x_left)
        self.panel.box_x_right.Bind(wx.EVT_SPINCTRLDOUBLE, self.onEnter_x_right)

        self.panel.box_for_function = wx.TextCtrl(self.panel, pos=(self.x_pos + 81, self.y_pos + 355), size=(260, 55), style=wx.TE_MULTILINE)

        self.panel.button_for_function = wx.Button(panel, label=u'Обновить', pos=(self.x_pos + 351, self.y_pos + 356), size=(95,26))
        self.panel.button_for_function.SetBackgroundColour('White')
        self.panel.button_for_function.SetForegroundColour(wx.Colour('GREY'))
        self.panel.button_for_function.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        self.panel.button_for_function.Bind(wx.EVT_BUTTON, self.base_function_button)

        self.panel.new_button = wx.Button(panel, label=u'Обновить', pos=(self.x_pos + 351, self.y_pos + 383), size=(95,26))
        self.panel.new_button.SetBackgroundColour('White')
        self.panel.new_button.SetForegroundColour(wx.Colour('FOREST GREEN'))
        self.panel.new_button.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        self.panel.new_button.Bind(wx.EVT_BUTTON, self.new_button_function)

        self.num_of_params = 0
        self.parameters_names = ['a:', 'b:', 'c:', 'd:', 'e:', 'f:', 'g:', 'h:']
        self.arguments_part_inside_function = []
        self.function_created = False
        self.f = None

    def base_function_button(self, event):

        ''' Кнопка "Базовый" одновременно управляет всеми моделями с одним сценарием:
            переоценка базовой версии + запрос на перерисовку серого
            графика каждой такой модели (управление моделями с большим количеством
            сценариев происходит непосредственно на самих графиках/панелях с
            такими моделями). '''

        self.main_frame.estimation_series.update_base()

    def create_function(self):

        ''' Парсинг функции из окна ввода и определение ее как метода для function_plot. '''

        function_string = '''def f(self, args):'''
        for parameter in self.arguments_part_inside_function:
            function_string += parameter
        function_string += '''\n\tx = copy.copy(self.base_x)'''
        function_string += '''\n\ty = copy.copy(self.base_y)'''
        for i in range(self.panel.box_for_function.GetNumberOfLines()):
            function_string += '\n\t' + self.panel.box_for_function.GetLineText(i)
        function_string += '''\n\treturn x, y'''

        exec function_string
        self.f = types.MethodType(locals()['f'], self)


    def new_button_function(self, event):

        ''' Кнопка "Новый" одновременно управляет всеми моделями с одним сценарием:
            переоценка модифицированной версии + запрос на перерисовку зеленого
            графика каждой такой модели (управление моделями с большим количеством
            сценариев происходит непосредственно на самих графиках/панелях с
            такими моделями). '''

        self.create_function()

        self.modified_x[:], self.modified_y[:] = self.f([])

        self.new_factor_plot.set_xdata(self.modified_x)
        self.new_factor_plot.set_ydata(self.modified_y)
        self.canvas.draw_idle()

        self.main_frame.estimation_series.update_modified()

    # def minimize_button_function_on(self, event):
    #
    #     ''' Нажатие кнопки "Minimize" для запуска алгоритма минимизации. '''
    #
    #     self.minimization_timer.Start(350)
    #     self.panel.minimize_button.SetLabel('Stop')
    #     self.panel.minimize_button.Bind(wx.EVT_BUTTON, self.minimize_button_function_stop)

    # def minimize_button_function_stop(self, event):
    #
    #     ''' Нажатие кнопки "Stop" для отсановки алгоритма минимизации. '''
    #
    #     self.panel.minimize_button.SetLabel('Minimize')
    #     self.panel.minimize_button.Bind(wx.EVT_BUTTON, self.minimize_button_function_on)
    #
    #     self.minimization_timer.Stop()
    #     self.nelder_mead_obj.refresh()

    # def minimization_process(self, event):
    #
    #     # ----- ШАГ 1: ЗАПУСК АЛГОРИТМА -----
    #     # Первым делом процесс минимизации должен понять, с какой параметрической
    #     # функцией он будет работать:
    #     if not self.nelder_mead_obj.function_created:
    #         self.create_function()
    #         self.nelder_mead_obj.function_created = True
    #         return
    #
    #     # ----- ШАГ 2: ИТЕРАЦИИ АЛГОРИТМА -----
    #     # Работаем только с теми оценками-объектами, которые:
    #     #     1. имеют 1 сценарий;
    #     #     2. уже подготовлены и отрисованы.
    #     list_1 = self.main_frame.estimation_series.many_scenarios
    #     list_2 = self.main_frame.estimation_series.prepared_and_drawn
    #
    #     if False not in [y for x, y in zip(list_1, list_2) if not x]:
    #         if not self.nelder_mead_obj.finished:
    #
    #             # То есть как только закончилась отрисовка всех оценок с одним сценарием
    #             # (оценки с большим количеством сценариев игнорируются),
    #             # запускаем следующую итерацию алгоритма:
    #             self.nelder_mead_obj.iteration()
    #             return
    #
    #     # ----- ШАГ 3: ЗАВЕРШЕНИЕ АЛГОРИТМА -----
    #     if self.nelder_mead_obj.finished:
    #         self.panel.minimize_button.SetLabel('Minimize')
    #         self.panel.minimize_button.Bind(wx.EVT_BUTTON, self.minimize_button_function_on)
    #         self.minimization_timer.Stop()
    #         self.nelder_mead_obj.refresh()
    #         return


    def queries(self):

        ''' Функция используется при минимизации. На каждом шаге итерации в минимизации осуществляется
            запрос на перерасчет всех моделей с одним сценарием (с последующей отрисовкой графиков).
            Данная функция отвечает за этот запрос. '''

        self.main_frame.estimation_series.update_modified()

    # def add_param_button_function(self, event):
    #
    #     if self.num_of_params == 0:
    #         self.minimize_but_ypos += 33
    #     else:
    #         self.minimize_but_ypos += 30
    #
    #     self.panel.__dict__['param_text_' + str(self.num_of_params)] = wx.StaticText(self.panel, label=self.parameters_names[self.num_of_params],
    #                                                                          pos=(self.x_pos + 536, self.minimize_but_ypos + 2))
    #     self.panel.__dict__['param_control_' + str(self.num_of_params)] = wx.SpinCtrlDouble(self.panel, pos=(self.x_pos + 554, self.minimize_but_ypos),
    #                                                                                       size=(50, 20), inc=0.1, initial=1.0)
    #
    #     self.nelder_mead_obj.x_start = np.append(self.nelder_mead_obj.x_start, [self.panel.__dict__['param_control_'+str(self.num_of_params)].GetValue()])
    #     self.panel.__dict__['param_control_' + str(self.num_of_params)].Bind(wx.EVT_SPINCTRLDOUBLE, self.__getattribute__('change_x_start' + str(self.num_of_params)))
    #
    #     arg_inside = '''\n\t''' + self.parameters_names[self.num_of_params][0] + ''' = args[''' + str(self.num_of_params) +  ''']'''
    #     self.arguments_part_inside_function.append(arg_inside)
    #
    #     self.num_of_params += 1
    #
    #     if self.num_of_params == 8:
    #         self.panel.add_param_button.Disable()
    #
    #     if self.num_of_params == 1:
    #         self.panel.minimize_button.Enable()
    #         self.panel.remove_param_button.Enable()

    # def remove_param_button_function(self, event):
    #
    #     self.minimize_but_ypos -= 30
    #
    #     self.arguments_part_inside_function.pop()
    #     self.nelder_mead_obj.x_start = self.nelder_mead_obj.x_start[:-1]
    #     self.num_of_params -= 1
    #
    #     self.panel.__dict__['param_text_' + str(self.num_of_params)].Destroy()
    #     self.panel.__dict__['param_control_' + str(self.num_of_params)].Destroy()
    #
    #     if self.num_of_params == 7:
    #         self.panel.add_param_button.Enable()
    #
    #     if self.num_of_params == 0:
    #         self.panel.minimize_button.Disable()
    #         self.panel.remove_param_button.Disable()

    # def change_x_start0(self, event):
    #     self.nelder_mead_obj.x_start[0] = self.panel.__dict__['param_control_0'].GetValue()
    #     print self.nelder_mead_obj.x_start
    #
    # def change_x_start1(self, event):
    #     self.nelder_mead_obj.x_start[1] = self.panel.__dict__['param_control_1'].GetValue()
    #     print self.nelder_mead_obj.x_start
    #
    # def change_x_start2(self, event):
    #     self.nelder_mead_obj.x_start[2] = self.panel.__dict__['param_control_2'].GetValue()
    #     print self.nelder_mead_obj.x_start
    #
    # def change_x_start3(self, event):
    #     self.nelder_mead_obj.x_start[3] = self.panel.__dict__['param_control_3'].GetValue()
    #     print self.nelder_mead_obj.x_start
    #
    # def change_x_start4(self, event):
    #     self.nelder_mead_obj.x_start[4] = self.panel.__dict__['param_control_4'].GetValue()
    #     print self.nelder_mead_obj.x_start
    #
    # def change_x_start5(self, event):
    #     self.nelder_mead_obj.x_start[5] = self.panel.__dict__['param_control_5'].GetValue()
    #     print self.nelder_mead_obj.x_start
    #
    # def change_x_start6(self, event):
    #     self.nelder_mead_obj.x_start[6] = self.panel.__dict__['param_control_6'].GetValue()
    #     print self.nelder_mead_obj.x_start
    #
    # def change_x_start7(self, event):
    #     self.nelder_mead_obj.x_start[7] = self.panel.__dict__['param_control_7'].GetValue()
    #     print self.nelder_mead_obj.x_start

    def onEnter_y_top(self, event):

        raw_value = self.panel.box_y_top.GetValue()
        self.y_bottom = self.ax.get_ylim()[0]
        try:
            if self.y_bottom < float(raw_value):
               self.ax.set_ylim([self.y_bottom, float(raw_value)])
               self.canvas.draw_idle()
        except:
            pass

    def onEnter_y_bottom(self, event):

        raw_value = self.panel.box_y_bottom.GetValue()
        self.y_top = self.ax.get_ylim()[1]
        try:
            if float(raw_value) < self.y_top:
                self.ax.set_ylim([float(raw_value), self.y_top])
                self.canvas.draw_idle()
        except:
            pass

    def onEnter_x_left(self, event):

        raw_value = self.panel.box_x_left.GetValue()
        self.x_right = self.ax.get_xlim()[1]
        try:
            if float(raw_value) < self.x_right:
                self.ax.set_xlim([float(raw_value), self.x_right])
                self.canvas.draw_idle()
        except:
            pass

    def onEnter_x_right(self, event):

        raw_value = self.panel.box_x_right.GetValue()
        self.x_left = self.ax.get_xlim()[0]
        try:
            if self.x_left < float(raw_value):
                self.ax.set_xlim([self.x_left, float(raw_value)])
                self.canvas.draw_idle()
        except:
            pass


class VariablePlot(object):

    def __init__(self, main_frame, panel, figsize, position):

        self.main_frame = main_frame
        self.figsize = figsize
        self.x_pos, self.y_pos = position
        self.panel = panel
        self.panel.variables = res_columns

        # <editor-fold desc="Initiate plot">
        self.figure = mpl.figure.Figure(figsize=self.figsize)
        self.canvas = FigureCanvas(panel, -1, self.figure)
        self.canvas.SetPosition(position)
        self.figure.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.98)

        self.ax = self.figure.add_subplot(111)
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        # </editor-fold>

        # <editor-fold desc="Buttons">
        # Кнопка запуска модели:
        self.panel.prepare_button = wx.Button(self.panel, label=u'Загрузить', pos=(self.x_pos+102, self.y_pos - 46), size=(62,22))
        self.panel.prepare_button.SetBackgroundColour(wx.WHITE)
        self.panel.prepare_button.SetFont(wx.Font(8, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        self.panel.prepare_button.Bind(wx.EVT_BUTTON, self.prepare_button)

        # Кнопка сохранения графика:
        self.panel.save_button = wx.Button(self.panel, label=u'Сохранить', pos=(self.x_pos+512, self.y_pos-46), size=(62,22))
        self.panel.save_button.SetBackgroundColour(wx.WHITE)
        self.panel.save_button.SetFont(wx.Font(8, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        self.panel.save_button.Bind(wx.EVT_BUTTON, self.save_current_graph)
        self.panel.save_button.Disable()

        # Кнопка для управления базовыми линиями в случае 50 сценариев:
        self.panel.grey_button = wx.Button(self.panel, label='', pos=(self.x_pos + 208, self.y_pos - 46), size=(21,22))
        self.panel.grey_button.SetBackgroundColour(wx.Colour('LIGHT GREY'))
        self.panel.grey_button.Bind(wx.EVT_BUTTON, self.grey_button_function)
        self.panel.grey_button.Disable()
        self.panel.grey_button.Hide()

        # Кнопка для управления модифицированными линиями в случае 50 сценариев:
        self.panel.green_button = wx.Button(self.panel, label='', pos=(self.x_pos + 228, self.y_pos - 46), size=(21,22))
        self.panel.green_button.SetBackgroundColour(wx.Colour('FOREST GREEN'))
        self.panel.green_button.Bind(wx.EVT_BUTTON, self.green_button_function)
        self.panel.green_button.Disable()
        self.panel.green_button.Hide()

        # Кнопка для обновления модифициронной оценки для случая 50 графиков (для экокномии времени перерисовка
        # модифицированной модели для случая 50 сценариев совершается по отдельному запросу):
        self.panel.refresh_picture = wx.Image(r'sensitivity\pics\refresh.png', wx.BITMAP_TYPE_ANY)
        self.panel.refresh_picture = self.panel.refresh_picture.Scale(19, 19, wx.IMAGE_QUALITY_HIGH)
        self.panel.refresh_picture = self.panel.refresh_picture.ConvertToBitmap()

        self.panel.refresh_button = wx.BitmapButton(self.panel, -1, self.panel.refresh_picture, pos=(self.x_pos + 254, self.y_pos - 45), size=(19, 19))
        self.panel.refresh_button.Bind(wx.EVT_LEFT_DOWN, self.refresh_many_scenarios)
        self.panel.refresh_button.Disable()
        self.panel.refresh_button.Hide()
        # </editor-fold>

        # <editor-fold desc="Widgets: ИЦБ, месяц и год оценки, переменная для вывода на график:">
        # Раскрывающиеся листы для выбора параметров: ИЦБ, месяц и год оценки, переменная для вывода на график:
        self.panel.combo_bonds = wx.ComboBox(self.panel, value=bonds.index.tolist()[0], pos=(self.x_pos + 103, self.y_pos - 22), size=(60, 24), choices=bonds.index.tolist())
        self.panel.combo_bonds.Bind(wx.EVT_COMBOBOX, self.set_range)

        self.panel.combo_month = wx.ComboBox(self.panel, pos=(self.x_pos + 168, self.y_pos - 22), size=(80, 30))
        self.panel.combo_month.Bind(wx.EVT_COMBOBOX_DROPDOWN, self.control_month)

        self.panel.combo_year = wx.ComboBox(self.panel, pos=(self.x_pos + 253, self.y_pos - 22), size=(50, 30))
        self.panel.combo_year.Bind(wx.EVT_COMBOBOX, self.control_month)

        self.panel.combo_variable = wx.ComboBox(self.panel, value=self.panel.variables[9], pos=(self.x_pos + 308, self.y_pos - 22), size=(65, 30), choices=self.panel.variables)
        self.panel.combo_variable.Bind(wx.EVT_COMBOBOX, self.update_combo_variable)

        for control in [self.panel.combo_bonds, self.panel.combo_month, self.panel.combo_year, self.panel.combo_variable]:
            control.SetEditable(False)
            control.SetBackgroundColour(wx.WHITE)
            control.SetFont(wx.Font(8, wx.DEFAULT, wx.NORMAL, wx.NORMAL))
            control.SetInsertionPoint(control.GetLastPosition())

        # Технический момент: сделать так, чтобы при добавлении новой панели там выдавалась та же переменная, что и на других:
        for panel in self.main_frame.scrolled_panel_for_variables.Children:
            if not panel.is_empty and self.main_frame.scrolled_panel_for_variables.number_of_variables > 0:
                    self.panel.combo_variable.SetValue(panel.variable_plot.panel.combo_variable.GetValue())
                    break

        # Выставить дефолтные месяц и год для текущей ИЦБ:
        self.panel.bond_chosen = self.panel.combo_bonds.GetValue()
        self.panel.issued_date = bonds.loc[self.panel.bond_chosen, 'IssuedDate']
        self.panel.year_issue, self.panel.month_issue = int(self.panel.issued_date[:4]), int(self.panel.issued_date[5:7])
        first_date = dt.datetime(self.panel.year_issue, self.panel.month_issue, 15).date() + relativedelta(months=1)

        self.panel.months = months
        self.panel.combo_month.SetLabelText(self.panel.months[first_date.month-1])

        self.panel.today = dt.datetime.today().date()
        self.panel.combo_year.SetItems(map(lambda x: str(x), range(first_date.year, self.panel.today.year + 1)))
        self.panel.combo_year.SetValue(str(first_date.year))
        # </editor-fold>

        # <editor-fold desc="Widgetets: управление осью Y, количество сценариев">
        self.panel.text_y_top = wx.SpinCtrlDouble(self.panel, pos=(self.x_pos+35, self.y_pos-44), size=(60,20), min=-10000, max=10000, inc=0.01, initial=1)
        self.panel.text_y_bottom = wx.SpinCtrlDouble(self.panel, pos=(self.x_pos+35, self.y_pos-20), size=(60,20), min=-10000, max=10000, inc=0.01, initial=0)
        self.panel.text_y_top.Bind(wx.EVT_TEXT, self.onEnter_y_top)
        self.panel.text_y_bottom.Bind(wx.EVT_TEXT, self.onEnter_y_bottom)

        self.panel.text_scenarios = wx.ComboBox(self.panel, value='1', pos=(self.x_pos + 168, self.y_pos - 45), size=(40, 20), choices=['1', '50'])
        self.panel.text_scenarios.Bind(wx.EVT_COMBOBOX, self.change_scen_num)
        # </editor-fold>

        # <editor-fold desc="Widgets: окна для вывода значений по наведению курсора">
        self.panel.text_history_on = wx.TextCtrl(self.panel, pos=(self.x_pos+383, self.y_pos-22), size=(60, 22), style=wx.TE_CENTRE)
        self.panel.text_base_on = wx.TextCtrl(self.panel, pos=(self.x_pos+448, self.y_pos-22), size=(60, 22), style=wx.TE_CENTRE)
        self.panel.text_modified_on = wx.TextCtrl(self.panel, pos=(self.x_pos+513, self.y_pos-22), size=(60, 22), style=wx.TE_CENTRE)

        for control, color in zip([self.panel.text_history_on, self.panel.text_base_on, self.panel.text_modified_on],
                                  [wx.Colour('DARK SLATE GREY'), wx.Colour('GREY'), wx.Colour('FOREST GREEN')]):
            control.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))
            control.SetForegroundColour(color)
            control.SetEditable(False)
            control.SetBackgroundColour(wx.WHITE)

        self.base_update_text_controls = [self.panel.text_history_on, self.panel.text_base_on]
        self.modified_update_text_controls = [self.panel.text_history_on, self.panel.text_modified_on]
        # </editor-fold>

        # <editor-fold desc="Cвязь функций перерисовки с откликов о результатах">
        # Привязываем три евента к одной функции перерисовки
        self.panel.Connect(-1, -1, EVT_RESULT_BLACK, self.OnCalculationReady_ANY)
        self.panel.Connect(-1, -1, EVT_RESULT_GREY, self.OnCalculationReady_ANY)
        self.panel.Connect(-1, -1, EVT_RESULT_GREEN, self.OnCalculationReady_ANY)
        # </editor-fold>

        # <editor-fold desc="Вспомогательные болванки">
        self.base_cid, self.modified_cid, self.base_cursor, self.modified_cursor = None, None, None, None
        self.sql_variable_table, self.base_variable_table, self.modified_variable_table = None, None, None
        self.isin, self.evaluation_date, self.estimation_object = None, None, None

        # График считается активным, если на нем была запущена заготовка модели и уже отрисован
        # фон согласно графику макро (см. функцию plot_backgorund).
        # Данный флаг активно используется при синхронизированном управлении осями.
        self.plot_is_active = False

        # Этот технический флаг нужен для запуска интерактивного курсора. Используется при инициализации объекта
        # MovingCursorInsidePlot как указаение один раз нарисовать ax.ly (двигающуюся серую линию) и больше не
        # перерисовывать ее при обновлении объекта MovingCursorInsidePlot.
        self.first_time_drawn = True

        # К какой колонке нужно обращться в результирующих таблицах согласно виджету self.panel.combo_variable:
        self.column = columns_dict[self.panel.combo_variable.GetValue()]

        # Два флага для того, чтобы программа понимала, в каком режиме она находится: 1 сценарий или 50 сценариев.
        # Однако понимать она может с двух позиций: с точки зрения реально запущенной модели и с точки зрения выбранного
        # количества сценариев на виджете выбора количества сценариев (то есть с точки зрения self.panel.text_scenarios).
        self.scenarios_num = 1        # --- в каком режиме мы находимся с точки зрения self.panel.text_scenarios
        self.many_scenarios = False   # --- в каком режиме мы находимся с точки зрения реально запущенной модели

        # Флаги, используемые grey_button_function и green_button_function для скрытия/выведения
        # отрисованных линий базовой и модифицированной оценки соответственно (для случая 50 сценариев):
        self.modified_visible_1, self.modified_visible_2, self.modified_visible_3 = True, False, False
        self.base_visible_1, self.base_visible_2, self.base_visible_3 = True, False, False
        # </editor-fold>

    def grey_button_function(self, event):

        ''' Функция-кнопка, регулирующая вывод линий на график для оценки с большим количеством сценариев:
            Три состояния, нахождение в которых определяется флагами:
                -> self.base_visible_1 -- отображаются только среднее и квантили по баз. специф.;
                -> self.base_visible_2 -- отображаются среднее, квантили, и все 50 сценариев по баз. специф.;
                -> self.base_visible_3 -- ничего не отображается.
            По умолчанию стоит состояние 1. '''

        if self.base_visible_1:
            for i in range(50):
                self.__dict__['base_line_' + str(i)].set_visible(True)
            self.base_visible_1, self.base_visible_2 = False, True

        elif self.base_visible_2:
            for i in range(50):
                self.__dict__['base_line_' + str(i)].set_visible(False)
            self.base_line_mean.set_visible(False)
            self.base_line_q_up.set_visible(False)
            self.base_line_q_down.set_visible(False)
            self.base_visible_2, self.base_visible_3 = False, True

        elif self.base_visible_3:
            self.base_line_mean.set_visible(True)
            self.base_line_q_up.set_visible(True)
            self.base_line_q_down.set_visible(True)
            self.base_visible_3, self.base_visible_1 = False, True

        self.canvas.draw()
            
    def green_button_function(self, event):

        ''' Функция-кнопка, регулирующая вывод линий на график для оценки с большим количеством сценариев:
            Три состояния, нахождение в которых определяется флагами:
                -> self.modified_visible_1 -- отображаются только среднее и квантили по модиф. специф.;
                -> self.modified_visible_2 -- отображаются среднее, квантили, и все 50 сценариев по модиф. специф.;
                -> self.modified_visible_3 -- ничего не отображается.
            По умолчанию стоит состояние 1. '''

        if self.modified_visible_1:
            for i in range(50):
                self.__dict__['modified_line_' + str(i)].set_visible(True)
            self.modified_visible_1, self.modified_visible_2 = False, True

        elif self.modified_visible_2:
            for i in range(50):
                self.__dict__['modified_line_' + str(i)].set_visible(False)
            self.modified_line_mean.set_visible(False)
            self.modified_line_q_up.set_visible(False)
            self.modified_line_q_down.set_visible(False)
            self.modified_visible_2, self.modified_visible_3 = False, True

        elif self.modified_visible_3:
            self.modified_line_mean.set_visible(True)
            self.modified_line_q_up.set_visible(True)
            self.modified_line_q_down.set_visible(True)
            self.modified_visible_3, self.modified_visible_1 = False, True

        self.canvas.draw()

    def refresh_many_scenarios(self, event):

        ''' Кнопка-фукнция управления моделями с большим количеством сценариев.
            Расположена сверху каждой панели с моделью с большим количеством сценариев
            (белая стрелка вверх на зеленом фоне). Обновляет модифицированную версию
            модели с последующим запросом на отрисовку зеленых линий. '''

        # Функция должна работать только с уже подготовленной моделью с большим количеством сценариев
        # (за этот режим отвечает флаг self.many_scenarios):
        if self.many_scenarios:
            self.panel.green_button.Disable()
            self.panel.refresh_button.Disable()

            for i in range(50):
                self.__dict__['modified_line_' + str(i)].set_visible(False)
            self.modified_line_mean.set_visible(False)
            self.modified_line_q_up.set_visible(False)
            self.modified_line_q_down.set_visible(False)

            self.main_frame.estimation_series.update_modified(self.panel.id)
            # self.main_frame.estimation_series.update_modified(self.panel.id, many_scenarios=True, callback=ResultCallback(self.panel, EVT_RESULT_GREEN))
            self.modified_visible_1, self.modified_visible_2, self.modified_visible_3 = True, False, False

    def change_scen_num(self, event):

        '''  '''

        self.scenarios_num = int(self.panel.text_scenarios.GetValue())

        if self.scenarios_num == 1:
            self.panel.grey_button.Hide()
            self.panel.green_button.Hide()
            self.panel.refresh_button.Hide()

        elif self.scenarios_num > 1:
            self.panel.grey_button.Show()
            self.panel.green_button.Show()
            self.panel.refresh_button.Show()

    def set_range(self, event):

        self.panel.bond_chosen = self.panel.combo_bonds.GetValue()
        self.panel.issued_date = bonds.loc[self.panel.bond_chosen, 'IssuedDate']
        self.panel.year_issue = int(self.panel.issued_date[:4])
        self.panel.month_issue = int(self.panel.issued_date[5:7])
        self.panel.today = dt.datetime.today().date()

        self.panel.first_date = dt.datetime(self.panel.year_issue, self.panel.month_issue, 15).date() + relativedelta(months=1)
        self.panel.combo_month.SetLabelText(self.panel.months[self.panel.first_date.month-1])
        self.panel.combo_year.SetItems(map(lambda x: str(x), range(self.panel.first_date.year, self.panel.today.year + 1)))
        self.panel.combo_year.SetValue(str(self.panel.first_date.year))

    def control_month(self, event):

        self.panel.bond_chosen = self.panel.combo_bonds.GetValue()
        self.panel.issued_date = bonds.loc[self.panel.bond_chosen, 'IssuedDate']
        self.panel.year_issue = self.panel.issued_date[:4]
        self.panel.month_issue = int(self.panel.issued_date[5:7])
        self.panel.today = dt.datetime.today().date()

        self.panel.current_year = self.panel.combo_year.GetValue()
        if self.panel.current_year == self.panel.year_issue:
            self.panel.combo_month.Clear()
            self.panel.combo_month.SetItems(self.panel.months[self.panel.month_issue:])
            self.panel.combo_month.SetValue(self.panel.months[self.panel.month_issue:][0])
        elif self.panel.current_year == str(self.panel.today.year):
            self.panel.combo_month.Clear()
            self.panel.combo_month.SetItems(self.panel.months[:self.panel.today.month])
            self.panel.combo_month.SetValue(self.panel.months[:self.panel.today.month][0])
        else:
            self.panel.combo_month.Clear()
            self.panel.combo_month.SetItems(self.panel.months)
            self.panel.combo_month.SetValue(self.panel.months[0])

    def prepare_button(self, event):

        ''' Кнопка-функция "Prepare", запускающая подготовку модели в зависиомсти от выбранного режима
            (один или много сценариев) и делающая запрос на первую оценку с отрисовкой истории, базовой и
            модифицированной версий. '''

        self.panel.grey_button.Disable()
        self.panel.green_button.Disable()
        self.panel.refresh_button.Disable()

        # Здесь происходит установка флага, указывающего на то, в каком режиме действительно
        # работает панель/оценка: с одним или большим количеством сценариев. Важно понимать, что
        # выбранное значение сценариев на самой панели неинформативно, поэтому нужен отдельный флаг:
        self.scenarios_num = int(self.panel.text_scenarios.GetValue()) # TODO deprecated
        self.many_scenarios = True if self.scenarios_num > 1 else False # TODO deprecated

        # Прежде всего отрисовываем фон как на графиках c макроэкономикой:
        self.plot_background()

        # Определяем параметры оценки: бумагу и дату:
        self.isin = bonds.loc[self.panel.combo_bonds.GetValue(), 'ISIN']
        year = int(self.panel.combo_year.GetValue())
        month = self.panel.months.index(self.panel.combo_month.GetValue()) + 1
        self.evaluation_date = dt.datetime(year, month, 15).date().replace(day=15).strftime('%Y-%m-%d')

        # Сохраняем параметры запрошенной оценки внутри объекта управления оценками (estimation_series):
        self.main_frame.estimation_series.refill_model(self.evaluation_date, self.isin, self.many_scenarios, self.panel)

        # Заказываем загрузку модели и просим оповестить об окончании с помощью события EVT_RESULT_READY:
        self.main_frame.estimation_series.prepare_model(self.panel.id)

        # Просим сделать запрос к БД за историей и просим оповестить об окончании с помощью события EVT_RESULT_BLACK:
        self.main_frame.estimation_series.update_hist(self.panel.id)

        # Сразу просим посчитать базовый вариант и просим оповестить об окончании с помощью события EVT_RESULT_GREY:
        self.main_frame.estimation_series.update_base(self.panel.id)

        # Сразу просим посчитать модифицированный вариант и просим оповестить об окончании с помощью события EVT_RESULT_GREEN:
        self.main_frame.estimation_series.update_modified(self.panel.id)

    def OnCalculationReady_ANY(self, event):

        self.column = columns_dict[self.panel.combo_variable.GetValue()]

        # Выясняем тип события, которое поступило:
        ev_typ = event.GetEventType()

        if ev_typ == EVT_RESULT_BLACK:
            # Обновляем данные для исторической (черной) кривой на графике:
            self.sql_variable_table = event.data

            # Сохраняем результат в соответствующий лист внутри объекта управления оценками (estimation_series):
            self.main_frame.estimation_series.sql_results[self.panel.id] = self.sql_variable_table

            # Отрисовка истории:
            self.plot_scenario(type='history')

        elif ev_typ == EVT_RESULT_GREY:
            # Обновляем данные для базовой/ых (серой/ых) кривой/ых на графике:
            self.base_variable_table = self.make_good_index(event.data)

            # Сохраняем результат в лист внутри объекта управления оценками (estimation_series):
            self.main_frame.estimation_series.base_results[self.panel.id] = self.base_variable_table

            # Перерисовка базового сценария:
            if not self.many_scenarios:
                self.plot_scenario(type='base')
            else:
                self.plot_many_scenarios(type='base')

        elif ev_typ == EVT_RESULT_GREEN:
            # Обновляем данные для модифицированной/ых (зеленой/ых) кривой/ых на графике:
            self.modified_variable_table = self.make_good_index(event.data)

            # Сохраняем результат в лист внутри объекта управления оценками (estimation_series):
            self.main_frame.estimation_series.modified_results[self.panel.id] = self.modified_variable_table

            # Перерисовка модифицированного сценария:
            if not self.many_scenarios:
                self.plot_scenario(type='modified')
            else:
                self.plot_many_scenarios(type='modified')

        else:
            # Ничего не делаем - такая ситуация невозможна:
            return

        self.canvas.draw()


    def plot_scenario(self, type = 'base'):

        self.column = columns_dict[self.panel.combo_variable.GetValue()]

        if type == 'history':
            if self.sql_variable_table is None:
                return
            self.sql_line.set_data([self.sql_variable_table.index.values, self.sql_variable_table[self.column].values])
        elif type == 'base':
            if self.sql_variable_table is None:
                return
            if self.base_variable_table is None:
                return
            self.set_cursor_walking(self.sql_variable_table[self.column], self.base_variable_table, self.base_update_text_controls, type='base')
            self.base_line.set_data([self.base_variable_table.index.values, self.base_variable_table[self.column].values])
        elif type == 'modified':
            if self.sql_variable_table is None:
                return
            if self.modified_variable_table is None:
                return
            self.set_cursor_walking(self.sql_variable_table[self.column], self.modified_variable_table, self.modified_update_text_controls, type='modified')
            self.modified_line.set_data([self.modified_variable_table.index.values, self.modified_variable_table[self.column].values])
            self.main_frame.estimation_series.prepared_and_drawn[self.panel.id] = True
            # self.main_frame.update_RMSE_value(self.main_frame.estimation_series.calculate_total_rmse(model.CFN.CPR))
        else:
            pass

    def plot_many_scenarios(self, type='base'):

        self.column = columns_dict[self.panel.combo_variable.GetValue()]

        if type == 'base':

            if self.sql_variable_table is None:
                return
            if self.base_variable_table is None:
                return

            for i in range(50):
                x = self.base_variable_table[self.base_variable_table[model.CFN.SCN]==i].index
                y = self.base_variable_table[self.base_variable_table[model.CFN.SCN]==i][self.column]
                self.__dict__['base_line_' + str(i)].set_data([x.values, y.values])
                self.__dict__['base_line_' + str(i)].set_visible(False)
                
            m = self.base_variable_table.groupby(model.CFN.DAT).apply(np.mean)[self.column].to_frame()
            q_up = self.base_variable_table.groupby(model.CFN.DAT).apply(lambda x: x[self.column].quantile(0.95))
            q_down = self.base_variable_table.groupby(model.CFN.DAT).apply(lambda x: x[self.column].quantile(0.05))

            self.base_line_mean.set_data([m.index.values, m.values])
            self.base_line_q_up.set_data([q_up.index.values, q_up.values])
            self.base_line_q_down.set_data([q_down.index.values, q_down.values])

            self.set_cursor_walking(self.sql_variable_table[self.column], m, self.base_update_text_controls, type='base')

            self.panel.grey_button.Enable()

        elif type == 'modified':

            if self.sql_variable_table is None:
                return
            if self.modified_variable_table is None:
                return

            for i in range(50):
                x = self.modified_variable_table[self.modified_variable_table[model.CFN.SCN] == i].index
                y = self.modified_variable_table[self.modified_variable_table[model.CFN.SCN] == i][self.column]
                self.__dict__['modified_line_' + str(i)].set_data([x.values, y.values])
                self.__dict__['modified_line_' + str(i)].set_visible(False)

            m = self.modified_variable_table.groupby(model.CFN.DAT).apply(np.mean)[self.column].to_frame()
            q_up = self.modified_variable_table.groupby(model.CFN.DAT).apply(lambda x: x[self.column].quantile(0.95))
            q_down = self.modified_variable_table.groupby(model.CFN.DAT).apply(lambda x: x[self.column].quantile(0.05))

            self.modified_line_mean.set_data([m.index.values, m.values])
            self.modified_line_q_up.set_data([q_up.index.values, q_up.values])
            self.modified_line_q_down.set_data([q_down.index.values, q_down.values])

            self.modified_line_mean.set_visible(True)
            self.modified_line_q_up.set_visible(True)
            self.modified_line_q_down.set_visible(True)

            self.set_cursor_walking(self.sql_variable_table[self.column], m, self.modified_update_text_controls, type='modified')

            self.main_frame.estimation_series.prepared_and_drawn[self.panel.id] = True

            self.panel.green_button.Enable()
            self.panel.refresh_button.Enable()

        else:
            pass


    def set_cursor_walking(self, sql_series, var_table, controls, type='base',):

        if type == 'base':

            self.base_cursor = MovingCursorInsidePlot(self.ax, sql_series, var_table[self.column], controls, draw_line=True, first_time_drawn=self.first_time_drawn)
            self.base_cid = self.canvas.mpl_connect('motion_notify_event', self.base_cursor.mouse_move)
            self.first_time_drawn = False
            if self.modified_cursor is not None:
                self.base_cursor.redraw(self.modified_cursor.date)

        elif type == 'modified':

            self.modified_cursor = MovingCursorInsidePlot(self.ax, sql_series, var_table[self.column], controls, draw_line=False)
            self.modified_cid = self.canvas.mpl_connect('motion_notify_event', self.modified_cursor.mouse_move)
            if self.base_cursor is not None:
                self.modified_cursor.redraw(self.base_cursor.date)

    def plot_background(self):

        self.ax.cla()
        self.ax.grid(linestyle='--', lw=0.4, alpha=0.5, zorder=0)
        self.ax.axes.get_xaxis().set_visible(True)
        self.ax.axes.get_yaxis().set_visible(True)

        self.sql_line, = self.ax.plot([], [], c='darkslategrey', alpha=1, lw=1.4, zorder=4)

        if self.scenarios_num == 1:
            self.base_line, = self.ax.plot([], [], c='gray', alpha=0.4, lw=1, zorder=2, ls='-')
            self.modified_line, = self.ax.plot([], [], c='g', alpha=0.9, lw=1.3, zorder=3)
        else:
            for i in range(50):
                self.__dict__['base_line_' + str(i)], = self.ax.plot([], [], lw=0.25, c='lightgray')
                self.__dict__['modified_line_' + str(i)], = self.ax.plot([], [], lw=0.25, c='forestgreen', alpha=0.4)
            self.base_line_mean, = self.ax.plot([], [], lw=1.2, c='gray')
            self.base_line_q_up, = self.ax.plot([], [], lw=0.8, c='gray', ls='--')
            self.base_line_q_down, = self.ax.plot([], [], lw=0.8, c='gray', ls='--')

            self.modified_line_mean, = self.ax.plot([], [], lw=1.2, c='forestgreen')
            self.modified_line_q_up, = self.ax.plot([], [], lw=0.8, c='forestgreen', ls='--')
            self.modified_line_q_down, = self.ax.plot([], [], lw=0.8, c='forestgreen', ls='--')

        self.y_bottom, self.y_top = 0, 0.6

        for panel in self.main_frame.scrolled_panel_for_variables.Children:
            if not panel.is_empty and panel.variable_plot.plot_is_active:
                self.y_bottom = panel.variable_plot.y_bottom
                self.y_top = panel.variable_plot.y_top
                break

        self.ax.set_ylim([self.y_bottom, self.y_top])
        self.ax.yaxis.set_tick_params(labelsize=6.5)
        self.panel.text_y_bottom.SetValue(self.y_bottom)
        self.panel.text_y_top.SetValue(self.y_top)

        self.x_left, self.x_right = self.main_frame.Notebook_for_MacroPlot.GetCurrentPage().macro_plot.ax.get_xlim()
        self.ax.set_xlim([self.x_left, self.x_right])

        self.left_macro_border = self.main_frame.macro_with_hist.index[0]
        self.right_macro_border = self.main_frame.macro_with_hist.index[-1]
        for date in [self.left_macro_border, self.right_macro_border, self.main_frame.last_hist_date]:
            self.ax.axvline(date, c='gray', linestyle='--', lw=0.65, alpha=0.7, zorder=1)
        self.ax.axvspan(self.left_macro_border.replace(month=1, year=1971), self.left_macro_border, facecolor='lightgray', alpha=0.3, zorder=0)
        self.ax.axvspan(self.right_macro_border, self.right_macro_border.replace(month=12, year=2200), facecolor='lightgray', alpha=0.3, zorder=0)
        self.ax.axvspan(self.left_macro_border, self.main_frame.last_hist_date, facecolor='khaki', alpha=0.2, zorder=0)

        check_tick_size(self)

        self.plot_is_active = True
        self.canvas.draw()
        self.panel.save_button.Enable()


    def update_combo_variable(self, event):
        
        for panel in self.main_frame.scrolled_panel_for_variables.Children:
            if not panel.is_empty:
                panel.variable_plot.panel.combo_variable.SetValue(self.panel.combo_variable.GetValue())
                panel.variable_plot.update()

    def update(self):

        if self.plot_is_active:

            self.column = columns_dict[self.panel.combo_variable.GetValue()]
            self.plot_scenario(type='history')

            if not self.many_scenarios:

                self.plot_scenario(type='base')
                self.plot_scenario(type='modified')

            else:

                self.plot_many_scenarios(type='base')
                self.plot_many_scenarios(type='modified')

            self.canvas.draw()

    def save_current_graph(self, event):

        if self.plot_is_active:

            default_name = self.panel.combo_bonds.GetValue() + ', ' + self.panel.combo_month.GetValue() + ' ' + \
                           self.panel.combo_year.GetValue() + ', ' + self.panel.combo_variable.GetValue()

            with wx.FileDialog(self.main_frame, "Save PNG file", wildcard="PNG files (*.png)|*.png",
                               defaultFile = default_name,
                               style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:

                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return  # the user changed their mind

                # save the current contents in the file
                pathname = fileDialog.GetPath()

                try:
                    try:
                        self.ax.ly.set_visible(False)
                    except:
                        pass

                    with open(pathname, 'wb') as file:
                        self.figure.savefig(file, format='png')

                    try:
                        self.ax.ly.set_visible(True)
                    except:
                        pass

                except IOError:
                    wx.LogError("Cannot save current data in file '%s'." % pathname)

    def onEnter_y_top(self, event):

        if self.plot_is_active:
            raw_value = self.panel.text_y_top.GetValue()
            for panel in self.main_frame.scrolled_panel_for_variables.Children:
                if not panel.is_empty and panel.variable_plot.plot_is_active:
                    try:
                        if panel.variable_plot.y_bottom + 0.01 <= raw_value:
                            panel.variable_plot.panel.text_y_top.SetValue(raw_value)
                            panel.variable_plot.y_top = raw_value
                            panel.variable_plot.ax.set_ylim([panel.variable_plot.y_bottom, self.y_top])
                            panel.variable_plot.canvas.draw_idle()
                        else:
                            panel.variable_plot.panel.text_y_top.SetMin(panel.variable_plot.y_bottom + 0.01)
                    except:
                        pass

    def onEnter_y_bottom(self, event):

        if self.plot_is_active:
            raw_value = self.panel.text_y_bottom.GetValue()
            for panel in self.main_frame.scrolled_panel_for_variables.Children:
                if not panel.is_empty and panel.variable_plot.plot_is_active:
                    try:
                        if raw_value <= panel.variable_plot.y_top - 0.01:
                            panel.variable_plot.panel.text_y_bottom.SetValue(raw_value)
                            panel.variable_plot.y_bottom = raw_value
                            panel.variable_plot.ax.set_ylim([self.y_bottom, panel.variable_plot.y_top])
                            panel.variable_plot.canvas.draw_idle()
                        else:
                            panel.variable_plot.panel.text_y_bottom.SetMax(panel.variable_plot.y_top - 0.01)
                    except:
                        pass

    def make_good_index(self, var_table):

        var_table.reset_index(inplace=True)
        var_table[model.CFN.DAT] = var_table[model.CFN.DAT].dt.date
        var_table.set_index(model.CFN.DAT, inplace=True)

        return var_table

def check_tick_size(plot):
    plot.number_of_ticks = len(plot.ax.get_xticks())
    if plot.number_of_ticks > 20:
        plot.ax.xaxis.set_tick_params(labelsize=4)
    elif plot.number_of_ticks > 15:
        plot.ax.xaxis.set_tick_params(labelsize=5)
    elif plot.number_of_ticks > 12:
        plot.ax.xaxis.set_tick_params(labelsize=6)
    elif plot.number_of_ticks > 9:
        plot.ax.xaxis.set_tick_params(labelsize=7)
    elif plot.number_of_ticks > 7:
        plot.ax.xaxis.set_tick_params(labelsize=8)
    else:
        plot.ax.xaxis.set_tick_params(labelsize=9)
