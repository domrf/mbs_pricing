# -*- coding: utf-8 -*-

import wx
import matplotlib.dates as mdates

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d


class MovingCursorInsidePlot(object):

    def __init__(self, ax, real, mean, text_controls_list, draw_line=False, first_time_drawn=False):

        self.ax = ax
        self.mean = mean
        self.realized = real
        self.text_controls_list = text_controls_list
        self.draw_line = draw_line
        self.first_time_drawn = first_time_drawn
        self.list_dates_real = list(self.realized.index.values)
        self.list_dates_sim = list(self.mean.index.values)

        self.combined_list_of_dates = sorted(list(set(self.list_dates_real) | set(self.list_dates_sim)))
        self.intersection_interval = sorted(list(set(self.list_dates_real) & set(self.list_dates_sim)))
        self.only_real_part = sorted(list(set(self.list_dates_real) - set(self.intersection_interval)))
        self.only_sim_part = sorted(list(set(self.list_dates_sim) - set(self.intersection_interval)))

        self.date = self.combined_list_of_dates[0]

        if self.draw_line and self.first_time_drawn:
            self.ax.ly = ax.axvline(self.date, color='darkgray', alpha=0.75, lw=1)

        if self.date in self.intersection_interval:

            self.mean_on = round(float(self.mean[self.list_dates_sim.index(self.date)]), 3)
            self.realized_on = round(float(self.realized[self.list_dates_real.index(self.date)]), 3)

            self.text_controls_list[0].SetValue(str(self.realized_on))
            self.text_controls_list[1].SetValue(str(self.mean_on))

        elif self.date in self.only_real_part:

            self.realized_on = round(float(self.realized[0]), 3)
            self.text_controls_list[0].SetValue(str(self.realized_on))

        elif self.date in self.only_sim_part:

            self.mean_on = round(float(self.mean[0]), 3)
            self.text_controls_list[1].SetValue(str(self.mean_on))

    def mouse_move(self, event):
        if event.inaxes != self.ax:
            return

        x = mdates.num2date(event.xdata)
        x = x.date().replace(day=1)

        self.redraw(x)

    def redraw(self, x):

        indx_all = np.where(np.array(self.combined_list_of_dates) == x)[0]
        indx_real = np.where(np.array(self.list_dates_real) == x)[0]
        indx_sim = np.where(np.array(self.list_dates_sim) == x)[0]

        if indx_all.size != 0:

            indx_all = int(indx_all)
            self.date = self.combined_list_of_dates[indx_all]

            if self.draw_line:
                self.ax.ly.set_xdata(self.date)
                self.ax.figure.canvas.draw()

            if self.date in self.intersection_interval:

                indx_real = int(indx_real)
                indx_sim = int(indx_sim)
                mean = self.mean[indx_sim]
                realized = self.realized[indx_real]

                self.mean_on = round(float(mean), 3)
                self.realized_on = round(float(realized), 3)

                self.text_controls_list[0].SetValue(str(self.realized_on))
                self.text_controls_list[1].SetValue(str(self.mean_on))

            elif self.date in self.only_real_part:

                indx_real = int(indx_real)
                realized = self.realized[indx_real]
                self.realized_on = round(float(realized), 3)

                self.text_controls_list[0].SetValue(str(self.realized_on))
                self.text_controls_list[1].Clear()

            elif self.date in self.only_sim_part:

                indx_sim = int(indx_sim)
                mean = self.mean[indx_sim]
                self.mean_on = round(float(mean), 3)

                self.text_controls_list[0].Clear()
                self.text_controls_list[1].SetValue(str(self.mean_on))

        else:

            if self.draw_line:
                self.ax.ly.set_xdata(x)
                self.ax.figure.canvas.draw()

            self.text_controls_list[0].Clear()
            self.text_controls_list[1].Clear()


class InteractiveMacro(object):

    def __init__(self, macro_plot):

        self.macro_plot = macro_plot

        self.ly_vert = self.macro_plot.ax.axvline(self.macro_plot.macro_series.index[0], color='darkgray', alpha=0.4, lw=1)
        self.ly_hor = self.macro_plot.ax.axhline(self.macro_plot.macro_series[0], color='darkgray', alpha=0.4, lw=1)

        self.marker_green, = self.macro_plot.ax.plot(self.macro_plot.macro_series.index[0], self.macro_plot.macro_series[0],
                               marker="o", markerfacecolor="cyan", markeredgecolor='teal',
                               markersize= 5, zorder=10)
        self.marker_red, = self.macro_plot.ax.plot(self.macro_plot.macro_series.index[0], self.macro_plot.macro_series[0],
                               marker="o", markerfacecolor="lightsalmon", markeredgecolor='darkred',
                               markersize= 5, zorder=10)

        self.date = self.macro_plot.macro_series.index[0]
        self.value = self.macro_plot.macro_series[0]
        self.date_on = pd.to_datetime(self.date).date()
        self.series_on = round(float(self.value), 3)

        self.macro_plot.panel.text_date_on.SetValue(str(self.date_on))
        self.macro_plot.panel.text_series_on.SetValue(str(self.series_on))

        self.press = False
        self.move = False
        self.just_series = None
        self.inside = True

    def mouse_move_first_day(self, event):
        if event.inaxes != self.macro_plot.ax:
            return

        x = mdates.num2date(event.xdata).date().replace(day=1)
        indx = np.where(self.macro_plot.macro_series.index.values == np.datetime64(x))[0]

        if indx.size != 0:

            self.inside = True

            indx = int(indx)
            self.value = self.macro_plot.macro_series[indx]
            self.date = self.macro_plot.macro_series.index[indx]
            self.ly_vert.set_xdata(self.date)
            self.ly_hor.set_ydata(float(event.ydata))

            self.marker_green.set_data([self.date], [self.value])
            self.marker_red.set_data([self.date], [event.ydata])

            self.date_on = pd.to_datetime(self.date).date()

            self.macro_plot.panel.text_date_on.SetValue(str(self.date_on))

            if self.press == True:
                self.marker_green.set_visible(False)
                self.series_on = round(float(event.ydata), 3)
            else:
                self.marker_green.set_visible(True)
                self.macro_plot.panel.text_series_on.SetForegroundColour(wx.BLACK)
                self.series_on = round(float(self.value), 3)

            self.macro_plot.panel.text_series_on.SetValue(str(self.series_on))

        else:

            self.inside = False

            self.marker_green.set_visible(False)
            self.ly_vert.set_xdata(x)
            self.ly_hor.set_ydata(float(event.ydata))
            self.marker_red.set_data([x], [event.ydata])
            self.macro_plot.panel.text_date_on.SetValue('')
            self.macro_plot.panel.text_series_on.SetForegroundColour(wx.RED)
            self.macro_plot.panel.text_series_on.SetValue(str(round(float(event.ydata), 3)))

        self.macro_plot.ax.figure.canvas.draw()

        if self.macro_plot.main_frame.synchronize_mode_on:
            for panel in self.macro_plot.main_frame.scrolled_panel_for_variables.Children:
                if not panel.is_empty and panel.variable_plot.plot_is_active:
                    if panel.variable_plot.base_cursor is not None:
                        panel.variable_plot.base_cursor.redraw(x)
                    if panel.variable_plot.modified_cursor is not None:
                        panel.variable_plot.modified_cursor.redraw(x)


    def on_press(self, event):
        try:
            if self.move is True: return
            if event.inaxes != self.macro_plot.ax: return

            x = mdates.num2date(event.xdata)
            y = event.ydata

            self.marker_red.set_ydata([y])
            self.just_series = pd.Series([y], index=[x])
            self.macro_plot.just_line.set_ydata(self.just_series.values)
            self.macro_plot.just_line.set_xdata(self.just_series.index)
            self.marker_green.set_visible(False)

            self.macro_plot.ax.figure.canvas.draw()

            self.press = True
            self.macro_plot.panel.text_series_on.SetForegroundColour(wx.RED)

            self.move = True

        except:
            pass

    def on_motion(self, event):
        try:
            if self.press is False: return
            if event.inaxes != self.macro_plot.ax: return

            x = mdates.num2date(event.xdata)
            y = event.ydata

            if x > self.just_series.index[-1]:
                self.just_series.at[x] = y

            self.macro_plot.just_line.set_ydata(self.just_series.values)
            self.macro_plot.just_line.set_xdata(self.just_series.index)
            self.macro_plot.ax.figure.canvas.draw()

        except:
            pass

    def on_release(self, event):
        try:
            if self.press == False: return

            self.press = False
            self.macro_plot.panel.text_series_on.SetForegroundColour(wx.BLACK)

            self.move = False
            self.macro_plot.just_line.set_data([], [])
            self.macro_plot.ax.figure.canvas.draw()

            if event.inaxes != self.macro_plot.ax: return

            start_date = self.just_series.index[0].date().replace(day=1)
            finish_date = self.just_series.index[-1].date().replace(day=1)
            range = [x.replace(day=1).date() for x in pd.date_range(start_date, finish_date, freq='M')]
            range_ordinal = [x.toordinal() for x in range]

            x = [x.toordinal() for x in self.just_series.index]
            function = interp1d(x, list(self.just_series.values), kind='linear', fill_value='extrapolate')
            self.new_macro = pd.Series(function(range_ordinal), index=[np.datetime64(d) for d in range])

            self.macro_plot.macro_series.update(self.new_macro)

            self.macro_plot.macro_line.set_data(self.macro_plot.macro_series.index, self.macro_plot.macro_series.values)
            self.just_series = None

            self.macro_plot.ax.figure.canvas.draw()

        except:
            pass

    def update(self):

        if self.inside:

            indx = int(np.where(self.macro_plot.macro_series.index.values == np.datetime64(self.date))[0])
            self.value = self.macro_plot.macro_series[indx]

            self.marker_green.set_data([self.date], [self.value])
            self.series_on = round(float(self.value), 3)
            self.macro_plot.panel.text_series_on.SetValue(str(self.series_on))
            self.macro_plot.ax.figure.canvas.draw()

    def mute(self):
        ''' In order to save graph without interactive widgets '''

        self.ly_vert.set_visible(False)
        self.ly_hor.set_visible(False)
        self.marker_green.set_visible(False)
        self.marker_red.set_visible(False)

    def unmute(self):

        self.ly_vert.set_visible(True)
        self.ly_hor.set_visible(True)
        self.marker_green.set_visible(True)
        self.marker_red.set_visible(True)





