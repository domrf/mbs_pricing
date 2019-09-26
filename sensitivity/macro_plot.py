# -*- coding: utf8 -*-
import wx

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import datetime as dt
from dateutil.relativedelta import relativedelta
from interactive import InteractiveMacro

from copy import copy
# from auxiliary import check_tick_size

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


class MacroPlot(object):

    def __init__(self, main_frame, panel, figsize, position, macro_series, last_hist_date, column):

        self.main_frame = main_frame

        self.panel = panel
        self.figsize = figsize
        self.x_pos, self.y_pos = position
        self.macro_series = macro_series
        self.default_macro_series = macro_series.copy()
        self.last_hist_date = last_hist_date
        self.column = column

        self.figure = mpl.figure.Figure(figsize=self.figsize)
        self.canvas = FigureCanvas(panel, -1, self.figure)
        self.canvas.SetPosition(position)
        self.figure.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.97)

        self.ax = self.figure.add_subplot(111)
        self.ax.grid(linestyle='--', lw=0.4, alpha=0.5, zorder=0)
        self.ax.yaxis.set_tick_params(labelsize=8)
        self.ax.set_xlim([self.main_frame.default_left_date, self.main_frame.default_right_date])
        check_tick_size(self)

        self.macro_line_basic, = self.ax.plot(self.macro_series.index, self.macro_series, lw=1, c='gray', alpha=0.7, zorder = 1, ls='--')
        self.macro_line, = self.ax.plot(self.macro_series.index, self.macro_series, lw=1.7, c='darkslategray', zorder = 2)
        self.just_line, = self.ax.plot([], [], lw=1.3, c='red', zorder = 2)

        self.panel.text_date_on = wx.TextCtrl(self.panel, pos=(self.x_pos+34, self.y_pos-23), size=(80, 20), style=wx.TE_CENTRE)
        self.panel.text_date_on.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.NORMAL))
        self.panel.text_series_on = wx.TextCtrl(self.panel, pos=(self.x_pos+124, self.y_pos-23), size=(60, 20), style=wx.TE_CENTRE)
        self.panel.text_series_on.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.NORMAL))
        # self.panel.synchronize = wx.CheckBox(self.panel, label=u'Совмещать отметки', pos=(self.x_pos+195, self.y_pos-21))
        # self.panel.synchronize.Bind(wx.EVT_CHECKBOX, self.synchronization)

        self.y_bottom, self.y_top = self.ax.get_ylim()
        self.x_left, self.x_right = self.ax.get_xlim()
        self.x_left = dt.datetime.fromordinal(int(self.x_left)).date()
        self.x_right = dt.datetime.fromordinal(int(self.x_right)).date()

        self.cursor = InteractiveMacro(self)
        self.cid_move_cursor = self.canvas.mpl_connect('motion_notify_event', self.cursor.mouse_move_first_day)
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.cursor.on_press)
        self.cid_move = self.canvas.mpl_connect('motion_notify_event', self.cursor.on_motion)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.cursor.on_release)
        self.cid_release_outside = self.canvas.mpl_connect('axes_leave_event', self.cursor.on_release)
        self.canvas.draw_idle()

        for date in [self.macro_series.index[0], self.macro_series.index[-1], self.last_hist_date]:
            self.ax.axvline(date, c='gray', linestyle='--', lw=0.65, alpha=0.7, zorder=1)
        self.ax.axvspan(self.macro_series.index[0].replace(month=1, year=1971), self.macro_series.index[0], facecolor='lightgray', alpha=0.3, zorder=0)
        self.ax.axvspan(self.macro_series.index[-1], self.macro_series.index[-1].replace(month=12, year=2200), facecolor='lightgray', alpha=0.3, zorder=0)
        self.ax.axvspan(self.macro_series.index[0], self.last_hist_date, facecolor='khaki', alpha=0.2, zorder=0)

        self.default_y_bottom, self.default_y_top = copy(self.y_bottom), copy(self.y_top)
        self.default_x_left, self.default_x_right = copy(self.x_left), copy(self.x_right)

        self.panel.button_for_save_macro = wx.Button(panel, label=u'Сохранить', pos=(self.x_pos+374, self.y_pos-23), size=(90, 23))
        self.panel.button_for_save_macro.Bind(wx.EVT_BUTTON, self.save_macro_plot)

        self.panel.button_for_mean_macro = wx.Button(panel, label=u'Симуляция', pos=(self.x_pos+471, self.y_pos-23), size=(90, 23))
        self.panel.button_for_mean_macro.Bind(wx.EVT_BUTTON, self.set_mean_macro)

        self.panel.button_for_default_macro = wx.Button(panel, label=u'Сбросить', pos=(self.x_pos+568, self.y_pos-23), size=(90, 23))
        self.panel.button_for_default_macro.Bind(wx.EVT_BUTTON, self.set_default_macro)

        self.panel.button_for_history_lims = wx.Button(panel, label=u'Узкий горизонт', pos=(self.x_pos+414, self.y_pos+370), size=(120, 23))
        self.panel.button_for_history_lims.Bind(wx.EVT_BUTTON, self.set_history_lims)

        self.panel.button_for_default_lims = wx.Button(panel, label=u'Широкий горизонт', pos=(self.x_pos+538, self.y_pos+370), size=(120, 23))
        self.panel.button_for_default_lims.Bind(wx.EVT_BUTTON, self.set_default_lims)

        for button in [self.panel.button_for_save_macro, self.panel.button_for_mean_macro, self.panel.button_for_default_macro,
                       self.panel.button_for_default_lims, self.panel.button_for_history_lims]:
            button.SetBackgroundColour('White')
            button.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.NORMAL))

        # <editor-fold desc="Arrows">
        self.panel.arrow_right = wx.Image(r'sensitivity\pics\right.png', wx.BITMAP_TYPE_ANY)
        self.panel.arrow_right = self.panel.arrow_right.Scale(40, 21.25, wx.IMAGE_QUALITY_HIGH)
        self.panel.arrow_right = self.panel.arrow_right.ConvertToBitmap()

        self.panel.arrow_left = wx.Image(r'sensitivity\pics\left.png', wx.BITMAP_TYPE_ANY)
        self.panel.arrow_left = self.panel.arrow_left.Scale(40, 21.25, wx.IMAGE_QUALITY_HIGH)
        self.panel.arrow_left = self.panel.arrow_left.ConvertToBitmap()

        self.panel.x_right_right_arrow = wx.BitmapButton(self.panel, -1, self.panel.arrow_right, pos=(self.x_pos + 620, self.y_pos + 340), size=(40, 21.25))
        self.panel.x_right_right_arrow.Bind(wx.EVT_LEFT_DOWN, self.x_right_right_arrow_left_down)
        self.panel.x_right_right_arrow.Bind(wx.EVT_LEFT_UP, self.x_right_right_arrow_left_up)

        self.panel.x_right_left_arrow = wx.BitmapButton(self.panel, -1, self.panel.arrow_left, pos=(self.x_pos + 505, self.y_pos + 340), size=(40, 21.25))
        self.panel.x_right_left_arrow.Bind(wx.EVT_LEFT_DOWN, self.x_right_left_arrow_left_down)
        self.panel.x_right_left_arrow.Bind(wx.EVT_LEFT_UP, self.x_right_left_arrow_left_up)

        self.panel.x_left_right_arrow = wx.BitmapButton(self.panel, -1, self.panel.arrow_right, pos=(self.x_pos + 145, self.y_pos + 340), size=(40, 21.25))
        self.panel.x_left_right_arrow.Bind(wx.EVT_LEFT_DOWN, self.x_left_right_arrow_left_down)
        self.panel.x_left_right_arrow.Bind(wx.EVT_LEFT_UP, self.x_left_right_arrow_left_up)

        self.panel.x_left_left_arrow = wx.BitmapButton(self.panel, -1, self.panel.arrow_left, pos=(self.x_pos + 30, self.y_pos + 340), size=(40, 21.25))
        self.panel.x_left_left_arrow.Bind(wx.EVT_LEFT_DOWN, self.x_left_left_arrow_left_down)
        self.panel.x_left_left_arrow.Bind(wx.EVT_LEFT_UP, self.x_left_left_arrow_left_up)

        self.panel.text_x_right = wx.TextCtrl(self.panel, pos=(self.x_pos + 549, self.y_pos + 342), size=(68, 18))
        self.panel.text_x_left = wx.TextCtrl(self.panel, pos=(self.x_pos + 74, self.y_pos + 342), size=(68, 18))

        for control in [self.panel.text_x_right, self.panel.text_x_left]:
            control.SetEditable(False)
            control.SetBackgroundColour(wx.WHITE)

        self.panel.text_x_left.SetLabelText(self.x_left.strftime('%Y-%m-%d'))
        self.panel.text_x_right.SetLabelText(self.x_right.strftime('%Y-%m-%d'))

        self.panel.text_y_top = wx.SpinCtrlDouble(self.panel, pos=(self.x_pos - 50, self.y_pos + 2), size=(50, 20), min=-1000, max=1000, inc=0.1, initial=round(self.y_top, 1))
        self.panel.text_y_bottom = wx.SpinCtrlDouble(self.panel, pos=(self.x_pos - 50, self.y_pos + 303), size=(50, 20), min=-1000, max=1000, inc=0.1, initial=round(self.y_bottom, 1))
        self.panel.text_y_top.Bind(wx.EVT_TEXT, self.onEnter_y_top)
        self.panel.text_y_bottom.Bind(wx.EVT_TEXT, self.onEnter_y_bottom)
        # </editor-fold>
        
        # <editor-fold desc="Timer functions">
        self.panel.timer_x_right_right = wx.Timer(self.panel)
        self.panel.timer_x_right_left = wx.Timer(self.panel)
        self.panel.timer_x_left_right = wx.Timer(self.panel)
        self.panel.timer_x_left_left = wx.Timer(self.panel)

        self.panel.Bind(wx.EVT_TIMER, self.x_right_right_arrow_pressed, self.panel.timer_x_right_right)
        self.panel.Bind(wx.EVT_TIMER, self.x_right_left_arrow_pressed, self.panel.timer_x_right_left)
        self.panel.Bind(wx.EVT_TIMER, self.x_left_right_arrow_pressed, self.panel.timer_x_left_right)
        self.panel.Bind(wx.EVT_TIMER, self.x_left_left_arrow_pressed, self.panel.timer_x_left_left)
        # </editor-fold>

        self.current_macro_type = 'Scenario'

    def onEnter_y_top(self, event):
        raw_value = self.panel.text_y_top.GetValue()

        self.y_top = float(raw_value)

        if self.y_bottom < self.y_top:
            self.ax.set_ylim([self.y_bottom, self.y_top])
            self.canvas.draw_idle()

    def onEnter_y_bottom(self, event):
        raw_value = self.panel.text_y_bottom.GetValue()

        self.y_bottom = float(raw_value)

        if self.y_bottom < self.y_top:
            self.ax.set_ylim([self.y_bottom, self.y_top])
            self.canvas.draw_idle()

    def set_default_macro(self, event):

        for panel in self.main_frame.Notebook_for_MacroPlot.Children:
            if panel.__class__.__name__ == 'PanelForMacroPlot':
                panel.macro_plot.macro_series.update(panel.macro_plot.default_macro_series)
                panel.macro_plot.macro_line.set_data(panel.macro_plot.macro_series.index, panel.macro_plot.macro_series.values)
                panel.macro_plot.canvas.draw()

    def set_mean_macro(self, event):

        if self.current_macro_type == 'Scenario':
            self.main_frame.macro_with_hist = self.main_frame.macro_mean_with_hist

        elif self.current_macro_type == 'Mean':
            self.main_frame.macro_with_hist = self.main_frame.macro_one_sim_with_hist

        for panel in self.main_frame.Notebook_for_MacroPlot.Children:
            if panel.__class__.__name__ == 'PanelForMacroPlot':
    
                if panel.macro_plot.current_macro_type == 'Scenario':

                    panel.macro_plot.macro_series = self.main_frame.macro_with_hist[panel.macro_plot.column]
                    panel.macro_plot.default_macro_series = self.main_frame.macro_mean_with_hist_copy[panel.macro_plot.column]
        
                    panel.macro_plot.panel.button_for_mean_macro.SetLabel(u'По КБД')
                    panel.macro_plot.current_macro_type = 'Mean'
        
                elif panel.macro_plot.current_macro_type == 'Mean':

                    panel.macro_plot.macro_series = self.main_frame.macro_with_hist[panel.macro_plot.column]
                    panel.macro_plot.default_macro_series = self.main_frame.macro_one_sim_with_hist_copy[panel.macro_plot.column]
        
                    panel.macro_plot.panel.button_for_mean_macro.SetLabel(u'Симуляция')
                    panel.macro_plot.current_macro_type = 'Scenario'
        
                panel.macro_plot.macro_line_basic.set_data(panel.macro_plot.default_macro_series.index.values, panel.macro_plot.default_macro_series)
                panel.macro_plot.macro_line.set_data(panel.macro_plot.macro_series.index.values, panel.macro_plot.macro_series)

                panel.macro_plot.cursor.update()
                panel.macro_plot.canvas.draw_idle()


    def set_default_lims(self, event):

        self.set_ticks_everywhere(self.main_frame.default_left_date, self.main_frame.default_right_date)

    def set_history_lims(self, event):

        self.set_ticks_everywhere(self.main_frame.history_left_date, self.main_frame.history_right_date)

    # def synchronization(self, event):
    #
    #     self.main_frame.synchronize_mode_on = self.panel.synchronize.GetValue()
    #
    #     for panel in self.main_frame.Notebook_for_MacroPlot.Children:
    #         if panel.__class__.__name__ == 'PanelForMacroPlot':
    #             panel.synchronize.SetValue(self.main_frame.synchronize_mode_on)

    # <editor-fold desc="Timer functions">
    def set_ticks_everywhere(self, left, right):

        for panel in self.main_frame.scrolled_panel_for_variables.Children:
            if not panel.is_empty and panel.variable_plot.plot_is_active:
                panel.variable_plot.ax.set_xlim([left, right])
                check_tick_size(panel.variable_plot)
                panel.variable_plot.canvas.draw()

        for panel in self.main_frame.Notebook_for_MacroPlot.Children:
            if panel.__class__.__name__ == 'PanelForMacroPlot':
                panel.macro_plot.ax.set_xlim([left, right])
                check_tick_size(panel.macro_plot)
                panel.macro_plot.canvas.draw_idle()

                panel.macro_plot.panel.text_x_left.SetLabelText(left.strftime('%Y-%m-%d'))
                panel.macro_plot.panel.text_x_right.SetLabelText(right.strftime('%Y-%m-%d'))

                panel.macro_plot.x_left = left
                panel.macro_plot.x_right = right

    def x_right_right_arrow_pressed(self, event):

        current_x_right_date = dt.datetime.fromordinal(int(self.ax.get_xlim()[1])).date()
        self.x_right = current_x_right_date + relativedelta(months=1)

        self.set_ticks_everywhere(self.x_left, self.x_right)

    def x_right_left_arrow_pressed(self, event):

        current_x_right_date = dt.datetime.fromordinal(int(self.ax.get_xlim()[1])).date()
        self.x_right = current_x_right_date - relativedelta(months=1)

        self.set_ticks_everywhere(self.x_left, self.x_right)

    def x_left_right_arrow_pressed(self, event):

        current_x_left_date = dt.datetime.fromordinal(int(self.ax.get_xlim()[0])).date()
        self.x_left = current_x_left_date + relativedelta(months=1)

        self.set_ticks_everywhere(self.x_left, self.x_right)

    def x_left_left_arrow_pressed(self, event):

        current_x_left_date = dt.datetime.fromordinal(int(self.ax.get_xlim()[0])).date()
        self.x_left = current_x_left_date - relativedelta(months=1)

        self.set_ticks_everywhere(self.x_left, self.x_right)

    def x_right_right_arrow_left_down(self, event):
        self.panel.timer_x_right_right.Start(50)

    def x_right_right_arrow_left_up(self, event):
        self.panel.timer_x_right_right.Stop()

    def x_right_left_arrow_left_down(self, event):
        self.panel.timer_x_right_left.Start(50)

    def x_right_left_arrow_left_up(self, event):
        self.panel.timer_x_right_left.Stop()

    def x_left_right_arrow_left_down(self, event):
        self.panel.timer_x_left_right.Start(50)

    def x_left_right_arrow_left_up(self, event):
        self.panel.timer_x_left_right.Stop()

    def x_left_left_arrow_left_down(self, event):
        self.panel.timer_x_left_left.Start(50)

    def x_left_left_arrow_left_up(self, event):
        self.panel.timer_x_left_left.Stop()
    # </editor-fold>

    def save_macro_plot(self, event):

        default_name = self.column + ', ' + str(dt.datetime.today().strftime('%Y-%m-%d'))

        with wx.FileDialog(self.main_frame, "Save PNG file", wildcard="PNG files (*.png)|*.png",
                           defaultFile = default_name,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            # save the current contents in the file
            pathname = fileDialog.GetPath()

            try:
                try:
                    self.cursor.mute()
                except:
                    pass

                with open(pathname, 'wb') as file:
                    self.figure.savefig(file, format='png')

                try:
                    self.cursor.unmute()
                except:
                    pass

            except IOError:
                wx.LogError("Cannot save current data in file '%s'." % pathname)
