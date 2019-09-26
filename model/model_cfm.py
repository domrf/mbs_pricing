# -*- coding: utf8 -*-
import numpy as np
import pandas as pd

# utility subsystems
from common import profiling
from common import debugging

from common import xparse, ALF
from impdata.names import CFN, LLD


class CModel(object):

    def __init__(self, parent, m_type, avg_level=1.0):

        self.parent = parent
        self.m_type = m_type

        self.model__xname = getattr(LLD, m_type)

        self._lambda = np.zeros((self.parent.N, self.parent.T))
        self._cond_prob = np.zeros((self.parent.N, self.parent.T))
        self._cond_migr = np.zeros((self.parent.N, self.parent.T))

        self.avg = avg_level
        self.baseline = np.ones((self.parent.N, self.parent.T))
        self.comb = np.ones((self.parent.N, self.parent.T))
        self.pers = np.ones((self.parent.N, 1))

        self.c_name = []
        self.c_func = []
        self.c_args = []
        self.c_kwgs = []

    def Free(self):
        for key in vars(self):
            setattr(self, key, None)

    def SetBaseline(self, func, *args, **kwargs):

        max_t = np.max(self.parent.Seasoning) + self.parent.T

        # fill baseline risks
        templine = func(np.array(range(max_t))/12.0, *args, **kwargs)

        # get into account credit seasoning
        for i in range(self.parent.N):
            self.baseline[i, :] = templine[self.parent.Seasoning[i]:self.parent.Seasoning[i] + self.parent.T]

        # fill bucket adjustments
        self.pers[:, :] = self.parent.PoolParams[self.model__xname].values.reshape((-1, 1))


    def SetCombrisk(self, name, func, *args, **kwargs):

        # self.c_numb += 1
        self.c_name.append(name)
        self.c_func.append(func)
        self.c_args.append(args)
        self.c_kwgs.append(kwargs)

    def _getdata(self, name):

        if name in self.parent.PoolParams.columns:
            return self.parent.PoolParams[name].values.reshape((-1, 1))
        elif name in self.parent.TimeParams.columns:
            return self.parent.TimeParams[name].values[:self.parent.T].reshape((1, -1))
        else:
            return 1.0

    def UpdateRisks(self):

        self.comb[:, :] = 1.0
        for name, func, args, kwargs in zip(self.c_name, self.c_func, self.c_args, self.c_kwgs):
            expr, exargs = xparse(name)
            if expr is None:
                cov = self._getdata(exargs[ALF[0]])
            else:
                for i in range(len(exargs)):
                    # Note: This only works in Python 2.x.
                    locals()[ALF[i]] = self._getdata(exargs[ALF[i]])
                cov = eval(expr)
            self.comb[:, :] *= func(cov, *args, **kwargs)

    def _debug_risks(self, debug_name):

        self.comb[:, :] = 1.0
        for name, func, args, kwargs in zip(self.c_name, self.c_func, self.c_args, self.c_kwgs):
            if name == debug_name:
                continue
            expr, exargs = xparse(name)
            if expr is None:
                cov = self._getdata(exargs[ALF[0]])
            else:
                for i in range(len(exargs)):
                    # Note: This only works in Python 2.x.
                    locals()[ALF[i]] = self._getdata(exargs[ALF[i]])
                cov = eval(expr)
            self.comb[:, :] *= func(cov, *args, **kwargs)

        return self.avg * self.baseline * self.comb * self.pers

    # @profiling
    def GetProb(self, **kwargs):

        # risk
        self._lambda[:, :] = self.avg * self.baseline * self.comb * self.pers

        # probability
        self._cond_prob[:, :] = self._lambda / (1.0 + self._lambda)

class CState(object):

    def __init__(self, parent):

        self.parent = parent

        # data handlers
        self.O_Models = []
        self.I_Models = []

        self.size_N_T1 = (self.parent.N, self.parent.T + 1)
        self.weights = np.zeros(self.size_N_T1)
        self.Survival = np.zeros(self.size_N_T1)

    def Free(self):
        for key in vars(self):
            setattr(self, key, None)

    def Compute_Survival(self):

        # --- flush temp arrays ---
        _cond_surv = np.ones(self.size_N_T1)

        # --- compute hazards & apply mods---
        for m in self.O_Models:
            m.GetProb()

        # --- cumulative survival func for each period ---
        _cond_surv[:, 1:] = np.cumprod(1.0 - np.sum([m._cond_prob for m in self.O_Models], axis=0), axis=1)

        # --- [a_prob to migrate] = [c_prob to survive] * [c_prob to migrate]
        for m in self.O_Models:
            m._cond_migr[:, :] = self.weights[:, [0]] * _cond_surv[:, :-1] * m._cond_prob[:, :]

        self.Survival[:, :] += _cond_surv * self.weights[:, [0]]

    def Accumulate_Inflows(self):

        for m in self.I_Models:
            self.weights[:, 1:] += m._cond_migr[:, :]


class CSimulatePool(object):

    # @profiling
    def __init__(self, parent, replace_pp=None, **kwargs):

        self.parent = parent

        self.Proba_Plan = self.parent.ppp
        self.Proba_Part = None
        self.Share_Part = None
        self.Volum_Part = None

        self.size_T1_S_N = (self.parent.T + 1, self.parent.S, self.parent.N, )
        self.size_T_S_N = (self.parent.T, self.parent.S, self.parent.N, )
        self.size_S_N = (self.parent.S, self.parent.N)
        self.size_1_N = (1, self.parent.N )
        self.size_N_T1 = (self.parent.N, self.parent.T + 1)

        self.PR = np.random.random(self.size_T_S_N)          # partial prepayment
        self.AP = np.zeros(self.size_S_N, dtype=np.float64)  # installment

        self.IRP = np.zeros(self.size_1_N, dtype=np.float64) # interest rate

        self.L0 = np.empty(self.size_S_N, dtype=np.bool)


    # @profiling
    def Free(self, **kwargs):
        for key in vars(self):
            setattr(self, key, None)

    # @profiling
    def StaticPrepaymentRate(self, *args, **kwargs):

        # get partial prepayment events
        self.PR[:] = self.PR < self.parent.Models['P', 'P']._cond_prob.T[:, np.newaxis, :]

        # multiply events by prepayment size and original debt
        self.PR[:] *= self.parent.Models['P', 'R']._cond_prob.T[:, np.newaxis, :] * self.parent.PoolParams[LLD.Original_Debt].values[np.newaxis, np.newaxis, :]

    # @profiling
    def DynamicPrepaymentRate(self, *args, **kwargs):

        t = args[0]

        # get partial prepayment events
        self.PR[t] = self.PR[t] < self.parent.Models['P', 'P']._cond_prob[:, [t]].T

        # multiply events by prepayment size and original debt
        self.PR[t] *= self.parent.Models['P', 'R']._cond_prob[:, [t]].T * self.parent.PoolParams[[LLD.Original_Debt]].values.T

    # @profiling
    def AnalyticPrepaymentRate(self, *args, **kwargs):

        t = args[0]
        value = args[1]

        # partial prepayment simulation
        self.PR[t] = value * self.Debt_BoP[t]

    # @profiling
    def Run(self, debt, irate, mat, payment, **kwargs):

        # average partial prepayment params
        Debt_BoP = np.zeros(self.size_T1_S_N, dtype=np.float64)  # debt at the beginning of a period
        Debt_Int = np.zeros(self.size_T1_S_N, dtype=np.float64)  # payed interest
        Debt_Amt = np.zeros(self.size_T1_S_N, dtype=np.float64)  # debt amortization by plan
        Debt_Pmt = np.zeros(self.size_T1_S_N, dtype=np.float64)  # debt amortization by part prepayment        
        Plan_EoP = np.zeros(self.size_S_N, dtype=np.float64)

        #t0 = time.clock()

        # average debt size in bucket
        Debt_BoP[0] = debt.reshape((1, -1))
        # weighted average interest percent in bucket
        self.IRP[:, :] = irate.reshape((1, -1)) / 100.0 * self.parent.dt
        # average payment amount in bucket
        self.AP[:, :] = payment.reshape((1, -1)) * 12.0 * self.parent.dt

        #print('SR init: {0:.3f} sec'.format(time.clock() - t0)); t0 = time.clock()
        
        r2 = np.random.random(self.size_T_S_N) < self.Proba_Plan                                                         
        for t in range(self.parent.T):

            # partial prepayment rate
            # self.PrepaymentFunction(t, self.pp_value)
            # self.DynamicPrepaymentRate(t)

            # payed interests
            Debt_Int[t] = Debt_BoP[t] * self.IRP

            # amortization by plan
            Debt_Amt[t] = np.minimum(self.AP - Debt_Int[t], Debt_BoP[t])

            # general debt at the end of the period planned, i.e. before unscheduled partial prepayment
            Plan_EoP[:, :] = Debt_BoP[t] - Debt_Amt[t]

            # partial prepayment - only for normal amortization
            Debt_Pmt[t] = np.minimum(self.PR[t], Plan_EoP)
            # # for z-spread
            # Debt_Pmt[t] = 0.017344417 * Debt_BoP[t]

            # general debt at the end of period t, which is start of period t + 1
            Debt_BoP[t + 1] = Plan_EoP - Debt_Pmt[t]
            if not np.any(Debt_BoP[t + 1]):
                break

            # dynamic payment amount
            # self.AP[Debt_BoP[t] == 0] = 0.0            
            self.L0[:, :] = Debt_BoP[t + 1] > 0
            self.L0[:, :] &= r2[t] & (Debt_Pmt[t] > 0)
            self.AP[self.L0] *= (Debt_BoP[t + 1][self.L0] / Plan_EoP[self.L0])

        # save i-th bucket history
        self.parent.NUM[:, :] = (Debt_BoP > 0.0).mean(axis=1).T
        self.parent.DBT[:, :] = (Debt_BoP).mean(axis=1).T
        self.parent.YLD[:, :] = (Debt_Int).mean(axis=1).T
        self.parent.AMT[:, :] = (Debt_Amt).mean(axis=1).T
        self.parent.PMT[:, :] = (Debt_Pmt).mean(axis=1).T


class CPool(object):

    # @profiling
    def __init__(self, inputs, lld_df, **kwargs):

        self.N = len(lld_df) # number of buckets
        self.S = inputs.Parameters.NumberOfEventSimulations # number of bucket simulations
        Period = np.timedelta64(1, 'M')
        Day = np.timedelta64(1, 'D')

        # number of period and modeling start/end dates
        self.ModelStart = inputs.Parameters.EvaluationMonth
        self.ModelEnd = inputs.Parameters.EvaluationMonth + 12 * inputs.Parameters.ModelingHorizon * Period

        T = ((self.ModelEnd - self.ModelStart) / Period).astype(int) + 1
        self.ModelEnd = self.ModelStart + Period * T
        self.Period_Date = np.arange(start=self.ModelStart, stop=self.ModelEnd + Period, step=Period, dtype='datetime64[M]')

        self.dt = Period / np.timedelta64(12, 'M')

        # modeling start/end dates
        self.T = len(self.Period_Date) - 1
        
        # service params
        self.ServPerc = 0.0
        self.ServFix = 0.0

        self.ppp = inputs.Adjusts.PartChange

        self.replacePrepayments = inputs.Parameters.ReplacePrepayments
        # self.Sim = CSimulatePool(self, replace_pp=inputs.Parameters.ReplacePrepayments)
        
        # create states
        self.States = dict()

        # this is dependent on migration-model structure
        for sname in ['C','D','F']:
            self.States[sname] = CState(self)

        # create models
        self.Models = dict()

        # ... and connect them to states
        self.Models['C', 'D'] = CModel(self, 'Default', avg_level=inputs.Adjusts.Default * inputs.Parameters.ModelsOfDefaults)
        if self.replacePrepayments is None:
            self.Models['C', 'F'] = CModel(self, 'Prepayment', avg_level=inputs.Adjusts.Prepayment * inputs.Parameters.EarlyPaymentModels)
            self.Models['P', 'P'] = CModel(self, 'Partial', avg_level=inputs.Adjusts.Partial * inputs.Parameters.ModelsOfAcceleratedDepreciation**0.5)
            self.Models['P', 'R'] = CModel(self, 'PartSize', avg_level=inputs.Adjusts.PartSize * inputs.Parameters.ModelsOfAcceleratedDepreciation**0.5)
        else:
            self.Models['C', 'F'] = CModel(self, 'Prepayment', avg_level=0.0)
            self.Models['P', 'P'] = CModel(self, 'Partial', avg_level=0.0)
            self.Models['P', 'R'] = CModel(self, 'PartSize', avg_level=0.0)

        # create connections between states
        for _out, _inp in self.Models.keys():
            try:
                self.States[_out].O_Models.append(self.Models[_out, _inp])
                self.States[_inp].I_Models.append(self.Models[_out, _inp])
            except KeyError:
                pass

        self.TimeParams = None
        
        # init lld data
        self.PoolParams = lld_df
        self.Seasoning = self.PoolParams[LLD.Seasoning].values.astype(int)

        # read adjustment structure
        models = {
            'Default': self.Models['C', 'D'],
            'Prepayment': self.Models['C', 'F'],
            'Partial': self.Models['P', 'P'],
            'PartSize': self.Models['P', 'R'],
        }

        # --- set baselines ---
        for col in inputs.Adjusts.base.columns[1:]:
            models[col].SetBaseline(np.interp, xp=inputs.Adjusts.base['Period'].values, fp=inputs.Adjusts.base[col].values)

        # --- set comb adjusts ---
        adjusts = inputs.Adjusts.comb_adjusts
        for key in adjusts:
            for col in adjusts[key].columns[1:]:
                models[col].SetCombrisk(key, np.interp, xp=adjusts[key]['Value'].values, fp=adjusts[key][col].values)

        # allocate objects
        self.size_N_T1 = (self.N, self.T + 1)
        self.NUM = np.zeros(self.size_N_T1)
        self.DBT = np.zeros(self.size_N_T1)
        self.YLD = np.zeros(self.size_N_T1)
        self.AMT = np.zeros(self.size_N_T1)
        self.PMT = np.zeros(self.size_N_T1)

    # @profiling
    def Free(self, **kwargs):

        # clear states
        for _state in self.States.values():
            _state.Free()

        # clear models
        for _model in self.Models.values():
            _model.Free()

        # clear other attributes
        for key in vars(self):
            setattr(self, key, None)

    # @profiling
    def FlushWeights(self, timedata, **kwargs):

        if isinstance(timedata.index, pd.core.index.MultiIndex):
            self.TimeParams = timedata.loc[pd.IndexSlice[:, :self.ModelEnd], :]
        else:
            self.TimeParams = timedata.loc[:self.ModelEnd, :]

        # clear arrays
        for _state in self.States.values():
            _state.weights[:] = 0.0
            _state.Survival[:] = 0.0

        # init lld data
        self.States['C'].weights[:, 0] = self.PoolParams[LLD.Mortgage_Num].values

        # recompute risks
        for _model in self.Models.values():
            _model.UpdateRisks()

    # @profiling
    def Survival(self, **kwargs):

        # compute survival functions
        for sname in ['C']:
            self.States[sname].Compute_Survival()

        # accumulate conditional migrations
        for sname in ['F', 'D']:
            self.States[sname].Accumulate_Inflows()

        # finalize survival functions t = T
        for sname in ['F', 'D']:
            self.States[sname].Survival[:, self.T] += self.States[sname].weights[:, self.T]

    # @profiling
    def _accumulate_results(self, verbose=False, **kwargs):

        # --- allocate result dataframe ---
        res = pd.DataFrame(
            0.0,
            index=pd.MultiIndex.from_product([range(1), self.Period_Date], names=[CFN.SCN, CFN.DAT]),
            columns=[CFN.NUM, CFN.DBT, CFN.YLD, CFN.AMT, CFN.ADV] + ([CFN.PLN, CFN.PRT, CFN.FLL, CFN.PDL, CFN.SRV] if verbose else [])
        )

        # --- save result cashflows ---
        res[CFN.NUM] = np.sum(self.NUM * self.States['C'].Survival, axis=0)
        res[CFN.DBT] = np.sum(self.DBT * self.States['C'].Survival, axis=0)
        res[CFN.YLD] = np.sum(self.YLD * self.States['C'].Survival, axis=0)

        if verbose:
            res[CFN.PLN] = np.sum(self.AMT * self.States['C'].Survival, axis=0)
            res[CFN.PRT] = np.sum(self.PMT * self.States['C'].Survival, axis=0)
            res[CFN.FLL].values[:-1] = np.sum(self.DBT * self.States['F'].weights, axis=0)[1:]
            res[CFN.PDL].values[:-1] = np.sum(self.DBT * self.States['D'].weights, axis=0)[1:]
            res[CFN.AMT] = res[CFN.PLN] + res[CFN.PRT] + res[CFN.FLL] + res[CFN.PDL]
            res[CFN.SRV] = self.ServPerc * res[CFN.DBT] + self.ServFix * res[CFN.NUM]

        res[CFN.ADV] = np.sum(self.PMT * self.States['C'].Survival, axis=0)
        res[CFN.ADV].values[:-1] += np.sum(self.DBT * self.States['F'].weights, axis=0)[1:] + np.sum(self.DBT * self.States['D'].weights, axis=0)[1:]
        res[CFN.AMT] = res[CFN.ADV] + np.sum(self.AMT * self.States['C'].Survival, axis=0)

        return res

    # @profiling
    def RunScenario(self, **kwargs):

        self.Models['P', 'P'].GetProb(**kwargs)
        self.Models['P', 'R'].GetProb(**kwargs)

        sim = CSimulatePool(self, replace_pp=self.replacePrepayments, **kwargs)
        
        # this cannot be reused
        sim.StaticPrepaymentRate(**kwargs)

        # compute payment plan for an average credit in bucket
        sim.Run(
            self.PoolParams[LLD.Current_Debt].values,
            self.PoolParams[LLD.Interest_Percent].values,
            self.PoolParams[LLD.Maturity].values,
            self.PoolParams[LLD.Payment].values,
            **kwargs
        )

        sim.Free(**kwargs)

        # --- compute survival probabilities ---
        self.Survival(**kwargs)

        return self._accumulate_results(**kwargs)

@profiling
def run(inputs, lld_df, scr_df, seed=None, **kwargs):

    np.random.seed(seed)

    # --- build pool structure ---
    pool = CPool(inputs, lld_df, **kwargs)

    # --- update model risks with new time params ---
    pool.FlushWeights(scr_df, **kwargs)

    # --- compute cashflow forecast ---
    result = pool.RunScenario(**kwargs)

    # --- clear memory ---
    pool.Free(**kwargs)

    return result
