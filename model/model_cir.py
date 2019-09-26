# -*- coding: utf8 -*-
#import time
import numpy as np
import pandas as pd

# utility subsystems
from common import profiling
from common import debugging

from impdata.names import CFN


y2000 = np.datetime64('2000-01-01', 'M')
month = np.timedelta64(1, 'M')

def spread_proxy(t):
    K1 = -0.14
    K2 = 0.0271
    K3 = 0.0444

    return K2 + K3 * np.exp(K1 * t)


# --- zcyc modeling functions ---

# G curve

def G_t(t, **kwargs):
    return G_t_fixed(t, kwargs['TAU'], kwargs['B0'], kwargs['B1'], kwargs['B2'], kwargs['G1'], kwargs['G2'],
                     kwargs['G3'], kwargs['G4'], kwargs['G5'], kwargs['G6'], kwargs['G7'], kwargs['G8'], kwargs['G9'])


def G_t_fixed(t, tau, b0, b1, b2, g1, g2, g3, g4, g5, g6, g7, g8, g9):
    '''
    k = 1.6
    a = np.zeros(9)
    b = np.zeros(9)

    a[0] = 0.0
    a[1] = 0.6
    for i in range(1, 8):
        a[i + 1] = a[i] + a[1] * k**i

    b[0] = a[1]
    for i in range(0, 8):
        b[i + 1] = b[i] * k
    '''

    a = np.array([0., 0.6, 1.56, 3.096, 5.5536, 9.48576, 15.777216, 25.8435456, 41.94967296])
    b = np.array([0.6, 0.96, 1.536, 2.4576, 3.93216, 6.291456, 10.0663296, 16.10612736, 25.769803776]) ** 2

    exp_t_tau = np.exp(-t / tau)
    return (
            b0
            + (b1 + b2) * (tau / t) * (1.0 - exp_t_tau)
            - b2 * exp_t_tau
            + g1 * np.exp(-(t - a[0]) ** 2 / b[0])
            + g2 * np.exp(-(t - a[1]) ** 2 / b[1])
            + g3 * np.exp(-(t - a[2]) ** 2 / b[2])
            + g4 * np.exp(-(t - a[3]) ** 2 / b[3])
            + g5 * np.exp(-(t - a[4]) ** 2 / b[4])
            + g6 * np.exp(-(t - a[5]) ** 2 / b[5])
            + g7 * np.exp(-(t - a[6]) ** 2 / b[6])
            + g8 * np.exp(-(t - a[7]) ** 2 / b[7])
            + g9 * np.exp(-(t - a[8]) ** 2 / b[8])
    )



def ZCYC(t, **kwargs):
    return G_t(t, **kwargs) * 0.0001


# --- Модель CIR ---
class CCIR(object):

    def __init__(self, start_date, n_year=30, n_sim=1000, delta=0, r_rate=ZCYC, forward_shift_months=0, verbose_log=False, **kbd_params):
        # forward_shift months forward evaluation
        self.forward_shift = forward_shift_months

        # Шаг модели, 1 месяц
        self.dt = 1.0 / 12.0
        self.period_num = int(1.0 / self.dt)

        # Мера моделирования, Q (spot), T (forward)
        self.t_measure = float(False)

        # Параметры сетки
        # Шаг модели
        self.dt = 1.0 / 12.0

        self.r_rate = r_rate
        self.r_args = kbd_params
        self.r_delta = float(delta) * 0.0001

        self.r = lambda t: self.r_rate(t, **self.r_args) + self.r_delta

        # --- Количество периодов всего (T), долгосрочной ставки (LT), количество симуляций (S) ---

        self.LT = 5 * self.period_num
        self.T = n_year * self.period_num + self.LT + self.forward_shift

        self.T1 = self.T + 1
        self.S = n_sim
        self.P = 2

        self.eps = 1e-6
        # self.t = np.linspace(0.0, self.T * self.dt, self.T1)
        self.t = np.zeros(int(self.T1))
        for i in xrange(self.T):
            self.t[i + 1] = self.t[i] + self.dt

        # --- Множитель единиц измерения величин вне модели (в базе данных, up/downstream моделях)
        # --- относительно используемых в модели абсолютных величин, т.е. при U = 100 1% в базе записывается как 1.0
        self.U = 100.0

        # --- Результаты моделирования (X), вектор независимых переменных VAR (Y) ---
        self.X = np.zeros((self.T1, self.P + 1, self.S))
        self.Y = np.zeros((5, self.S))
        self.xs = np.zeros((self.T1, self.S))

        # --- Параметры VAR ---
        self.var_i = None
        self.var_a = None
        self.var_s = None
        self.Corr = None
        self.Chol = None

        # --- Случайные шоки ---
        self.dz = np.zeros((self.T, self.P, self.S))

        # --- Параметры модели CIR ---
        self.cir_a = None
        self.cir_s = None
        self.cir_h = None
        self.a = None
        self.s = None
        self.h = None
        self.theta0 = None

        self.theta = np.zeros(self.T)
        self.dB = np.zeros((self.T1, self.T1))

        self.A = np.zeros(self.T1)
        self.B = np.zeros(self.T1)
        self.phi = np.zeros(self.T1)


        # Генерация сетки модели
        self.t = np.zeros(self.T1)
        for i in xrange(self.T):
            self.t[i + 1] = self.t[i] + self.dt
        self.eps = 1e-6


        # Рыночные ставки
        self.rt = np.zeros(self.T1)
        self.f_mkt = np.zeros(self.T1)

        self.rt[0] = self.r(self.eps)
        self.rt[1:] = self.r(self.t[1:])

        self.f_mkt[0] = self.rt[0]
        self.f_mkt[1:] = self.f_market(self.t[1:])

        self.Start_Date = np.datetime64(start_date, 'M')
        self.M1_Date = np.timedelta64(1, 'M')
        self.Period_Date = np.arange(self.Start_Date, self.Start_Date + (self.T1 - self.forward_shift) * self.M1_Date, self.M1_Date)
        self.Period_Date_Max = self.Start_Date + (self.T1 - self.forward_shift - self.LT) * self.M1_Date - 1

        self.verbose = verbose_log

    # Обобщенная функция рыночной процентной ставки
    def r_func(self, *args, **kwargs):
        return self.r_rate(*args, **kwargs) + self.r_delta

    def set_cir(self, cir_a=None, cir_s=None, cir_ax=None, cir_sx=None, cir_tx=None):

        # дрифт
        if cir_a is not None:
            self.cir_a = cir_a
            self.a = cir_ax if cir_ax is not None else cir_a / self.dt

        # волатильность
        if cir_s is not None:
            self.cir_s = cir_s

            self.s = cir_sx if cir_sx is not None else cir_s * np.sqrt(self.period_num)


        self.theta0 = cir_tx

        # вспомогательные переменные
        self.ss = self.s * self.s
        self.cir_h = (self.cir_a * self.cir_a + 2.0 * self.cir_s * self.cir_s) ** 0.5
        self.h = (self.a * self.a + 2.0 * self.ss) ** 0.5

        # theta0 > sigma^2 / 2a для обеспечения положительности симуляционных ставок
        if self.theta0 is None:
            self.theta0 = np.max([self.ss / (2 * self.a), self.rt[0], (self.a + self.h) * self.rt[0] / (2 * self.a)])
        self.atheta = self.a * self.theta0

        # начальное значение для моделируемого ряда x(t), где r(t) = x(t) + phi(t) согласовано с рыночной кривой
        self.x0 = self.f_mkt[0]


    # Параметры VAR
    def set_var(self, ms6_s=None, hpr_s=None):

        # --- VAR params ---
        # Свободный член авторегрессии
        self.var_i = np.array(
            # ln(MS6)       ln(HPR)
            [+1.271334735, +0.005556998]
        ).reshape((-1, 1))
        # Зависимости
        self.var_a = np.array([
            # ln(MS6)       ln(HPR)       Avg7_MS6      ln(r)         ln(r)/ln(r(-1)) # SPT
            [+0.600000000, +0.000000000, +0.000000000, +0.406818616, -1.137522747],  # ln(MS6)
            [+0.000000000, +0.600000000, -0.051671129, +0.000000000, +0.000000000],  # ln(HPR)
        ])
        # Волатильность при шоках
        self.var_s = np.array(
            # ln(MS6)       ln(HPR)
            [+0.113231275, +0.002281727]
        ).reshape((-1, 1))
        # Корреляции шоков
        self.Corr = np.array([
            # ln(MS6)       ln(HPR)
            [+1.000000000, -0.201899576],  # ln(MS6)
            [-0.201899576, +1.000000000],  # ln(HPR)
        ])

        self.Chol = np.linalg.cholesky(self.Corr)

        if ms6_s is not None:
            self.var_s[0, 0] = ms6_s

        if hpr_s is not None:
            self.var_s[1, 0] = hpr_s

    # Загрузка рядов исторических данных, изменение от месяца к месяцу
    def set_hst(self, df):
        #   Ставка поставляется в непрерывном (ранее в годовом) начислении из исторических данных как G(1./self.period_num),
        #   модель CIR настраивается на тех же данных, т.е. КБД в аннуализированном непрерывном начислении.
		#	Исторические значения и модельные генерации переводятся в годовое начисление и передаются в модель кэшфлоу
        self.SPT = np.interp(
            x=(np.arange(self.Period_Date[0] - np.timedelta64(5, 'M'), self.Period_Date[0] + np.timedelta64(1, 'M')) - y2000) / month,
            xp=(df.index.values.astype('datetime64[M]') - y2000) / month,
            fp=df[CFN.SPT].values / self.U
        )

        self.HPR = np.log(df.loc[:self.Period_Date[0], CFN.HPR].values)

        #   6-месячный моспрайм используется везде
        self.MS6 = df.loc[:self.Period_Date[0], CFN.MS6].values / self.U

        #   ставка по ипотеке используется везде и по логике это должна быть страновая ставка
        self.MIR = df.loc[:self.Period_Date[0], CFN.MIR].values / self.U

    def set_verbose(self, verbose):
        self.verbose = verbose

    # Vector autoregression
    # should be run after short rate X[:, 0, :] is simulated
    def run_VAR(self):

        for i in range(1, self.T1):
            # --- Подготовка к VAR, вспомогательный расчет 1 ---
            # self.Y[0, :] = self.X[i][0, :] # <- SPT
            self.Y[0, :] = self.X[i - 1][1, :]  # <- ln(MS6)
            self.Y[1, :] = self.X[i - 1][2, :]  # <- ln(HPR)
            # Avg7_MS6
            if i < 7:
                w1 = float(7 - i) / 7
                w2 = float(i) / 7
                self.Y[2, :] = w1 * np.average(self.MS6[-(7 - i):]) + w2 * np.average(np.exp(self.X[range(max(i - 7, 0), i), 1, :]), axis=0)
            else:
                self.Y[2, :] = np.average(np.exp(self.X[range(i - 7, i), 1, :]), axis=0)
            self.Y[3, :] = np.log(self.X[i][0, :])  # <- ln(SPT(t))
            self.Y[4, :] = self.Y[3, :] / np.log(self.X[i - 1][0, :])  # <- ln(SPT(t))/ln(SPT(t-1))

            # --- Расчет VAR ---
            # [.][P x S] = [P x 1] + dot([P x 5], [.][5 x S]) + [P x 1 ] * dot([P x P], [P x S])
            self.X[i, 1:, :] = self.var_i + np.dot(self.var_a, self.Y) + self.var_s * np.dot(self.Chol, self.dz[i - 1])


    # ставка срочности 1 месяц
    def f_monthly(self, t):
        if t <= self.eps:
            return self.rt[0]
        shift = self.eps if t == self.dt else 0.0

        return (self.r(t) * (t + self.dt) - self.r(t - self.dt + shift) * t) * self.period_num

    # self.dt форвард на момент времени t_i
    def f_i_dt(self, i):
        return (self.rt[i + 1] * self.t[i + 1] - self.rt[i] * self.t[i]) / self.dt

    # self.dt форвард на момент времени t
    def f_t_dt(self, t):
        t1 = t + self.dt
        return (self.r(t1) * t1 - self.r(t) * t) / self.dt if t > self.eps else self.r(t1)

    # мгновенный форвард f(0, t) по рынку,
    # используется для вычисления phi в cir++ и theta в общей модели:
    # центральная производная функции r(t) в точке t > 0
    def f_market(self, t):
        return self.r(t) + 0.5 * t * (self.r(t + self.eps) - self.r(t - self.eps)) / self.eps

    # CIR++

    def calc_phi_AB(self):

        # общие переменные
        h2 = self.h * 2.0
        a_h = self.a + self.h
        exp_ht = np.exp(np.dot(self.h, self.t))  # T+1
        exp_ht_1 = exp_ht - 1
        denominator = h2 + a_h * exp_ht_1

        # Калибровка на рыночную форвардную кривую
        # phi(t) = f_market(0, t) - f_cir(0, t),
        # f_mkt, f_cir - мгновенные t форварда в момент времени 0

        const_part = (2.0 * self.atheta * exp_ht_1) / denominator
        x_part = (h2 * h2 * exp_ht) / (denominator ** 2)
        f_cir = const_part + self.x0 * x_part

        self.phi = self.f_mkt - f_cir

        # Аффинная временная структура
        # P(t, T) = A(t, T) * exp(-B(t, T) * r(t))

        # self.A(T-t) = A(t, T)
        self.A = (h2 * np.exp(np.dot(0.5 * a_h, self.t)) / denominator) ** (2 * self.atheta / self.ss)

        # self.B(T-t) = B(t, T)
        self.B = 2 * exp_ht_1 / denominator

        return 1

    # MC симуляция для модели CIR++
    def run_cirpp_mc(self):

        self.calc_phi_AB()

        self.xs[0][:] = self.X[0][0, :] - self.phi[0]  # = self.x0

        for i in range(1, self.T1):
            # Схема Эйлера, симуляция в Q (current account) или T (zero coupon bond) мере

            dz_per_dt = self.dz[i - 1][0, :] / np.sqrt(self.period_num)
            dz_part = self.s * self.xs[i - 1][:] ** 0.5 * dz_per_dt

            self.xs[i][:] = self.xs[i - 1] + \
                (self.atheta - (self.a + self.B[self.T - (i - 1)] * self.ss * self.t_measure) * self.xs[i - 1]) * self.dt + \
                dz_part  # + dz2_part

            # Ограничиваем ставки внутренней кривой x(t) снизу нулем
            self.xs[i][self.xs[i] < 0] = self.eps

            # r(t) = x(t) + phi(t)
            self.X[i][0, :] = self.xs[i] + self.phi[i]

        return 1


    def Run(self, mir_k, new_seed=None, scr_seeds=None, method='cir++'):

        # Генерация шоков
        if scr_seeds is None:
            np.random.seed(new_seed)
            self.dz[:] = np.random.normal(size=(self.T, self.P, self.S))
        else:
            for i in range(self.S):
                np.random.seed(scr_seeds[i])
                self.dz[:, :, i] = np.random.normal(size=(self.T, self.P))

        # Начальные значения
        self.X[0][0, :] = self.rt[0]  # short rate
        self.X[0][1, :] = np.log(self.MS6[-1])  # ln(MosPrime 6m)
        self.X[0][2, :] = self.HPR[-1]  # ln(Home Price index growth Rate)

        # Симуляции Монте-Карло
        self.run_cirpp_mc()

        # Векторная авторегрессия
        self.run_VAR()

        # Переход к форвардной оценке: забываем симуляции до форвадной даты, укорачиваем время
        if self.forward_shift > 0:
            self.T -= self.forward_shift
            self.T1 -= self.forward_shift
            self.t = self.t[:-self.forward_shift]
            self.rt = self.rt[self.forward_shift:]
            self.A = self.A[self.forward_shift:]
            self.B = self.B[self.forward_shift:]
            self.phi = self.phi[self.forward_shift:]
            self.X = self.X[self.forward_shift:, :, :]

        # --- Подготовка результатов для модели досрочных погашений ---
        # Ряды для каждого из MC сценариев c шагом self.dt по времени
        size_S_T1 = (self.S, self.T1)

        # S(t) - сценарная реализованная короткая ставка (годовое начисление)
        St = np.zeros(size_S_T1)
        St[:] = np.exp(self.X[:, 0, :].T) - 1.0

        # G(t) - сценарная реализованная доходность (непрерывное начисление)
        Gt = np.zeros(size_S_T1)
        Gt[:, 0] = self.X[0][0, :]
        for i in range(1, self.T1):
            Gt[:, i] = (self.t[i - 1] * Gt[:, i - 1] + self.dt * self.X[i][0, :]) / self.t[i]

        # Y(t) - сценарная реализованная доходность (годовое начисление)
        Yt = np.zeros(size_S_T1)
        Yt[:] = np.exp(Gt[:, :]) - 1.0

        # D(t) - сценарное реализованное дисконтирование
        Dt = np.zeros(size_S_T1)
        Dt[:] = (1.0 + Yt[:, :]) ** -self.t

        # L(t) - сценарная реализованная доходность за LT / self.period_num лет вперед (непрерывное начисление)
        Lt = np.zeros(size_S_T1)
        for j in range(0, self.T - self.LT + 1):
            Lt[:, j] = np.sum(self.X[j + 1: j + self.LT + 1, 0, :], axis=0) / self.LT

        # R(t) - спот ставка в момент времени t_i = i * self.dt на LT / self.period_num лет (непрерывное начисление)
        if method == 'cir++':
            Rt = np.zeros(size_S_T1)
            t_range = range(self.T - self.LT + 1)
            ln_up = Dt[:, t_range] * self.A[self.LT:] * np.exp(-self.B[self.LT:] * self.x0)
            ln_down = Dt[:, self.LT:] * self.A[self.LT] * self.A[t_range] * np.exp(-self.B[t_range] * self.x0)
            Rt[:, t_range] = (np.log(ln_up / ln_down) - self.B[self.LT] * (self.phi[t_range] - self.rt[t_range])) / self.t[self.LT]

        # Ипотечные ставки
        Mt = np.zeros(size_S_T1)
        i0 = (self.Period_Date[0] - np.datetime64('2006-09-01', 'M')) / np.timedelta64(1, 'M')
        MIR_SPEED_DT = self.dt
        MIR_SPEED_UP = 1.12
        MIR_SPEED_DOWN = 2.73
        Mt[:, 0] = self.MIR[-1]
        for i in range(1, self.T1):
            loan_vs_mkt_diff = Mt[:, i - 1] - (Lt[:, i - 1] + spread_proxy(float(i - 1 + i0) * self.dt))
            speed_scale = loan_vs_mkt_diff * (MIR_SPEED_UP * (loan_vs_mkt_diff > 0) + MIR_SPEED_DOWN * (loan_vs_mkt_diff < 0))
            Mt[:, i] = Mt[:, i - 1] - speed_scale * MIR_SPEED_DT * mir_k

        # # L(t) - сценарная реализованная доходность за LT / self.period_num лет вперед (годовое начисление)
        # Lt[:, :] = np.exp(Lt) - 1.0

        # Z(t) - рыночная безрисковая доходность (годовое начисление)
        Zt = np.zeros(size_S_T1)
        Zt[:] = np.exp(self.rt.reshape((1, -1))) - 1.0

        #DZt - рыночный дисконтирование
        DZt = np.zeros(size_S_T1)
        DZt[:] = (1.0 + Zt[:, :]) ** -self.t

		# Присоединяем исторические данные к сгенерированным рядам ставки в годовом начислении
        sp6 = np.concatenate([self.SPT] * self.S).reshape((self.S, -1)) if len(self.SPT) > 0 else St[:, [0, 0, 0, 0, 0, 0]]
        sp6 = np.exp(sp6) - 1.0
        sp6 = np.concatenate([sp6, St[:, :-6]], axis=1).ravel()

        # Структура результатов модели процентных ставок, вход для модели досрочных погашений
        df = pd.DataFrame({
            CFN.SCN: np.repeat(np.arange(self.S), self.T1),  # np.array([range(self.S)] * self.T1).T.ravel(),
            CFN.DAT: np.tile(self.Period_Date, self.S),  # self.Period_Date.tolist() * self.S,
            CFN.ZCY: Zt.ravel() * self.U,
            CFN.SCR: Yt.ravel() * self.U,
            CFN.SPT: St.ravel() * self.U,
            CFN.MIR: Mt.ravel() * self.U,
            # CFN.R5Y: Lt.ravel()*self.U,
            CFN.MS6: np.exp(self.X[:, 1, :].T.ravel()) * self.U,
            CFN.HPR: np.exp(self.X[:, 2, :].T.ravel()),
            CFN.HPI: np.exp(self.X[:, 2, :]).cumprod(axis=0).T.ravel(),
            CFN.SP6: sp6 * self.U,
            CFN.SNG: np.concatenate([(np.mod((self.Period_Date - np.datetime64('2000-01-01', 'M')).astype(int), 12) + 1).astype(float)] * self.S),
        })
        df.set_index([CFN.SCN, CFN.DAT], inplace=True)
        df.sort_index(inplace=True)
        df = df.loc[pd.IndexSlice[:, :self.Period_Date_Max], :].copy(deep=True)

        return df


# Создание и запуск модели

def get_cir_obj(inputs, n_sim=None, delta=0, max_years=15, verbose=False):
    if n_sim is None:
        n_sim = inputs.Parameters.NumberOfMonteCarloScenarios

    forward = int(round(inputs.Parameters.ForwardMonths)) if hasattr(inputs.Parameters, 'ForwardMonths') else 0

    # --- run cir model ---
    if inputs.Parameters.UseStandartZCYC:
        # Рыночная кривая
        CIR = CCIR(
            start_date=inputs.Parameters.EvaluationDate,
            n_year=max_years,
            n_sim=n_sim,
            delta=delta,
            r_rate=ZCYC,
            forward_shift_months=forward,
            **inputs.Coefficients._asdict()
        )
    else:
        # Стрессированная кривая
        CIR = CCIR(
            start_date=inputs.Parameters.EvaluationDate,
            n_year=max_years,
            n_sim=n_sim,
            delta=delta,
            r_rate=np.interp,
            forward_shift_months=forward,
            **inputs.Parameters.ZCYCValues
        )
    # logging level
    CIR.set_verbose(verbose)

    # set rate volatility and average params
    CIR.set_cir(
        cir_a=inputs.Macromodel.cir_a,
        cir_s=inputs.Macromodel.cir_s * inputs.Parameters.BetModels,
        cir_ax=inputs.Macromodel.cir_ax,
        cir_sx=inputs.Macromodel.cir_sx * inputs.Parameters.BetModels,
        cir_tx=inputs.Macromodel.cir_tx,
    )

    CIR.set_var(
        ms6_s=inputs.Macromodel.ms6_s,
        hpr_s=inputs.Macromodel.hpr_s * inputs.Parameters.RealEstateIndexModels
    )

    # set history dataset to start from
    CIR.set_hst(inputs.Datasets.History)

    return CIR

# @debugging
@profiling
def run(inputs, delta=0, max_years=15, seeds=None, method='cir++', verbose=False, **kwargs):
    if isinstance(seeds, list):
        # list of seed - len(seeds) scenario number with one seed for each scenario
        return get_cir_obj(inputs, n_sim=len(seeds), delta=delta, max_years=max_years, verbose=verbose)\
            .Run(inputs.Parameters.MortgageRatesModels, scr_seeds=seeds, method=method)
    else:
        # no seed - default scenario number without any seed
        return get_cir_obj(inputs, delta=delta, max_years=max_years, verbose=verbose)\
            .Run(inputs.Parameters.MortgageRatesModels, new_seed=seeds, method=method)

