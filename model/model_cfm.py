# -*- coding: utf8 -*-
import time
import types
import numpy as np
import pandas as pd


from names import CFN, LLD

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
        self.time = np.ones((1, self.parent.T))
        self.pers = np.ones((self.parent.N, 1))

        self.bY = 20.0
        self.bY_len = int(self.bY * 12 + 1)

        self.t_numb = 0
        self.t_name = []
        self.t_func = []
        self.t_args = []

        self.c_numb = 0
        self.c_name = []
        self.c_func = []
        self.c_args = []

    def Free(self):
        for key in vars(self):
            setattr(self, key, None)

    def SetBaseline(self, func, *args):

        temp = np.zeros((10 * self.parent.T,))
        # fill baseline risks
        temp[:self.bY_len] = func(np.linspace(0.0, self.bY, self.bY_len), *args)
        temp[self.bY_len:] = temp[self.bY_len-1]

        # get into account credit seasoning
        for i in range(self.parent.N):
            self.baseline[i, :] = temp[self.parent.Seasoning[i]:self.parent.Seasoning[i] + self.parent.T]

        # fill bucket adjustments
        self.pers[:, :] = self.parent.PoolParams[self.model__xname].values.reshape((-1, 1))

    def SetTimerisk(self, name, func, args):

        self.t_numb += 1
        self.t_name.append(name)
        self.t_func.append(func)
        self.t_args.append(args)

    def SetCombrisk(self, name, func, args):

        self.c_numb += 1
        self.c_name.append(name)
        self.c_func.append(func)
        self.c_args.append(args)

    def UpdateRisks(self):


            # --- update time risks ---
            #  * OFZ_SPOT = CFN.SPT
            #  * SEASON   = CFN.SNG
            self.time[:, :] = 1.0
            for i in range(self.t_numb):
                cov = self.parent.TimeParams[self.t_name[i]].values[:self.parent.T].reshape((1, -1))
                self.time[:, :] *= self.t_func[i](cov, *self.t_args[i])

            # --- update comb risks ---
            #  * MIR(t) - INTEREST_PERCENT
            #  * IPV * HPI
            #  * |MIR(t) - MIR(0)|
            self.comb[:, :] = 1.0
            for i in range(self.c_numb):
                name_time, name_pers, operation = self.c_name[i]
                if operation not in ['diff', 'mult', 'abs_diff']:
                    print('unknown operation {}'.format(operation))
                    break          
                x = self.parent.PoolParams[name_pers].values.reshape((-1, 1))
                y = self.parent.TimeParams[name_time].values[:self.parent.T].reshape((1, -1))
                y_mat = np.dot(np.ones((x.shape[0], 1)), y)
                x_mat = np.dot(x, np.ones((1, y.shape[1])))
                if operation == 'diff':
                    # difference
                    cov = y_mat - x_mat
                elif operation == 'mult':
                    # multiplication
                    cov = y_mat * x_mat
                elif operation == 'abs_diff':
                    # abs(difference)
                    cov = np.abs(y_mat - x_mat)

                self.comb[:, :] *= self.c_func[i](cov, *self.c_args[i])

    def GetProb(self):

        # monthly risk
        self._lambda[:, :] = self.avg * self.baseline * self.comb * self.time * self.pers

        # probability
        self._cond_prob[:, :] = self._lambda / (1.0 + self._lambda)

    def GetProb_Partitial(self):

        self._lambda[:, :] = self.avg * self.baseline * self.comb * self.time * self.pers

        self._cond_prob[:, :] = self._lambda / (1.0 + self._lambda)

    def GetProb_PartSize(self):

        self._lambda[:, :] = self.avg * self.baseline * self.comb * self.time * self.pers

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

    def __init__(self, parent, replace_pp=None):

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
        
        # allocate objects
        self.NUM = np.zeros(self.size_N_T1, dtype=np.float64)
        self.DBT = np.zeros(self.size_N_T1, dtype=np.float64)
        self.YLD = np.zeros(self.size_N_T1, dtype=np.float64)
        self.AMT = np.zeros(self.size_N_T1, dtype=np.float64)
        self.PMT = np.zeros(self.size_N_T1, dtype=np.float64)       

        self.PR = np.random.random(self.size_T_S_N)          # partial prepayment
        self.AP = np.zeros(self.size_S_N, dtype=np.float64)  # installment

        self.IRP = np.zeros(self.size_1_N, dtype=np.float64) # interest rate

        self.L0 = np.empty(self.size_S_N, dtype=np.bool)

    def Free(self):
        for key in vars(self):
            setattr(self, key, None)

    def StaticPrepaymentRate(self, *args, **kwargs):

        # get partial prepayment events
        self.PR[:] = self.PR < self.parent.ModelPP._cond_prob.T[:, np.newaxis, :]
        # multiply events by prepayment size and original debt
        self.PR[:] *= self.parent.ModelPR._cond_prob.T[:, np.newaxis, :] * self.parent.PoolParams[LLD.Original_Debt].values[np.newaxis, np.newaxis, :]

    def DynamicPrepaymentRate(self, *args, **kwargs):

        t = args[0]

        # partial prepayment simulation
        self.PR[t] = (self.PR[t] < self.parent.ModelPP._cond_prob[:, [t]].T) * (self.parent.ModelPR._cond_prob[:, [t]].T * self.parent.PoolParams[[LLD.Original_Debt]].values.T)

    def AnalyticPrepaymentRate(self, *args, **kwargs):

        t = args[0]
        value = args[1]

        # partial prepayment simulation
        self.PR[t] = value * self.Debt_BoP[t]

    def Run(self, debt, irate, mat, payment):

        # average partial prepayment params
        
        Debt_BoP = np.zeros(self.size_T1_S_N, dtype=np.float64)  # debt at the beginning of a period
        Debt_Int = np.zeros(self.size_T1_S_N, dtype=np.float64)  # payed interest
        Debt_Amt = np.zeros(self.size_T1_S_N, dtype=np.float64)  # debt amortization by plan
        Debt_Pmt = np.zeros(self.size_T1_S_N, dtype=np.float64)  # debt amortization by part prepayment        
        Plan_EoP = np.zeros(self.size_S_N, dtype=np.float64)

        # average debt size in bucket
        Debt_BoP[0] = debt.reshape((1, -1))
        # weighted average interest percent in bucket
        self.IRP[:, :] = irate.reshape((1, -1)) / 100.0 * self.parent.dt
        # average payment amount in bucket
        self.AP[:, :] = payment.reshape((1, -1)) * 12.0 * self.parent.dt

        r2 = np.random.random(self.size_T_S_N) < self.Proba_Plan
        for t in range(self.parent.T):

            # payed interests
            Debt_Int[t] = Debt_BoP[t] * self.IRP

            # amortization by plan
            Debt_Amt[t] = np.minimum(self.AP - Debt_Int[t], Debt_BoP[t])

            # general debt at the end of the period planned, i.e. before unscheduled partial prepayment
            Plan_EoP[:, :] = Debt_BoP[t] - Debt_Amt[t]

            # partial prepayment - only for normal amortization
            Debt_Pmt[t] = np.minimum(self.PR[t], Plan_EoP)

            # general debt at the end of period t, which is start of period t + 1
            Debt_BoP[t + 1] = Plan_EoP - Debt_Pmt[t]
            if not np.any(Debt_BoP[t + 1]):
                break

            # dynamic payment amount
            self.L0[:, :] = Debt_BoP[t + 1] > 0
            self.L0[:, :] &= r2[t] & (Debt_Pmt[t] > 0)
            self.AP[self.L0] *= (Debt_BoP[t + 1][self.L0] / Plan_EoP[self.L0])
        
        # save i-th bucket history
        self.NUM[:, :] = (Debt_BoP > 0.0).mean(axis=1).T
        self.DBT[:, :] = (Debt_BoP).mean(axis=1).T
        self.YLD[:, :] = (Debt_Int).mean(axis=1).T
        self.AMT[:, :] = (Debt_Amt).mean(axis=1).T
        self.PMT[:, :] = (Debt_Pmt).mean(axis=1).T
      

class CPool(object):

    def __init__(self, inputs, lld_df):

        self.N = len(lld_df) # number of buckets
        self.S = inputs.Parameters.NumberOfEventSimulations # number of bucket simulations
        Period = np.timedelta64(1, 'M')

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

        # create states
        self.StateC = CState(self)
        self.StateD = CState(self)
        self.StateP = CState(self)
        
        # create models and connect them to states
        self.ModelCD = CModel(self, 'Default', avg_level=inputs.Adjusts.Default * inputs.Parameters.ModelsOfDefaults)
        if self.replacePrepayments is None:
            self.ModelCP = CModel(self, 'Prepayment', avg_level=inputs.Adjusts.Prepayment * inputs.Parameters.EarlyPaymentModels)
            self.ModelPP = CModel(self, 'Partial', avg_level=inputs.Adjusts.Partial * inputs.Parameters.ModelsOfAcceleratedDepreciation**0.5)
            self.ModelPR = CModel(self, 'PartSize', avg_level=inputs.Adjusts.PartSize * inputs.Parameters.ModelsOfAcceleratedDepreciation**0.5)
        else:
            self.ModelCP = CModel(self, 'Prepayment', avg_level=0.0)
            self.ModelPP = CModel(self, 'Partial', avg_level=0.0)
            self.ModelPR = CModel(self, 'PartSize', avg_level=0.0)

        # create connections
        self.StateC.O_Models.append(self.ModelCD)
        self.StateC.O_Models.append(self.ModelCP)
        self.StateD.I_Models.append(self.ModelCD)
        self.StateP.I_Models.append(self.ModelCP)
        
        self.TimeParams = None
        
        # init lld data
        self.PoolParams = lld_df
        self.Seasoning = self.PoolParams[LLD.Seasoning].values.astype(int)
        self.StateC.weights[:, 0] = self.PoolParams[LLD.Mortgage_Num].values
        
        models = {
            'Default': self.ModelCD,
            'Prepayment': self.ModelCP,
            'Partial': self.ModelPP,
            'PartSize': self.ModelPR,
        }

        # set baselines
        for col in inputs.Adjusts.base.columns[1:]:
            models[col].SetBaseline(np.interp, inputs.Adjusts.base['Period'].values, inputs.Adjusts.base[col].values)

        # set time adjusts
        adjusts = inputs.Adjusts.time_adjusts
        for cov in adjusts:
            for col in adjusts[cov].columns[1:]:
                models[col].SetTimerisk(cov, np.interp, (adjusts[cov]['Value'].values, adjusts[cov][col].values,))

        # set comb adjusts
        adjusts = inputs.Adjusts.comb_adjusts
        for cov in adjusts:
            for col in adjusts[cov].columns[1:]:
                models[col].SetCombrisk(cov, np.interp, (adjusts[cov]['Value'].values, adjusts[cov][col].values,))

        # allocate objects
        self.size_N_T1 = (self.N, self.T + 1)
        self.NUM = np.zeros(self.size_N_T1)
        self.DBT = np.zeros(self.size_N_T1)
        self.YLD = np.zeros(self.size_N_T1)
        self.AMT = np.zeros(self.size_N_T1)
        self.PMT = np.zeros(self.size_N_T1)


    def Free(self):

        # clear states
        self.StateC.Free()
        self.StateD.Free()
        self.StateP.Free()

        # clear models
        self.ModelCD.Free()
        self.ModelCP.Free()
        self.ModelPP.Free()
        self.ModelPR.Free()

        for key in vars(self):
            setattr(self, key, None)

    def FlushWeights(self, timedata):

        if isinstance(timedata.index, pd.core.index.MultiIndex):
            self.TimeParams = timedata.loc[pd.IndexSlice[:, :self.ModelEnd], :]
        else:
            self.TimeParams = timedata.loc[:self.ModelEnd, :]

        # clear start debt and weights
        self.StateC.weights[:, 1:] = 0
        self.StateC.Survival[:, :] = 0

        self.StateD.weights[:, :] = 0
        self.StateD.Survival[:, :] = 0

        self.StateP.weights[:, :] = 0
        self.StateP.Survival[:, :] = 0

        self.ModelCD.UpdateRisks()
        self.ModelCP.UpdateRisks()
        self.ModelPP.UpdateRisks()
        self.ModelPR.UpdateRisks()

    def Survival(self):

        # compute survival functions
        self.StateC.Compute_Survival()

        # accumulate conditional migrations
        self.StateP.Accumulate_Inflows()
        self.StateD.Accumulate_Inflows()

        # finalize survival functions t = T
        self.StateD.Survival[:, self.T] += self.StateD.weights[:, self.T]
        self.StateP.Survival[:, self.T] += self.StateP.weights[:, self.T]

    def RunScenario(self):

        self.ModelPP.GetProb_Partitial()
        self.ModelPR.GetProb_PartSize()

        sim = CSimulatePool(self, replace_pp=self.replacePrepayments)
        
        sim.StaticPrepaymentRate()

        # compute payment plan for an average credit in bucket
        sim.Run(
            self.PoolParams[LLD.Current_Debt].values,
            self.PoolParams[LLD.Interest_Percent].values,
            self.PoolParams[LLD.Maturity].values,
            self.PoolParams[LLD.Payment].values
        )
        
        # save i-th bucket history
        self.NUM[:, :] = sim.NUM
        self.DBT[:, :] = sim.DBT
        self.YLD[:, :] = sim.YLD
        self.AMT[:, :] = sim.AMT
        self.PMT[:, :] = sim.PMT

        sim.Free()

        # --- compute survival probabilities ---
        self.Survival()

        res = pd.DataFrame(
            0.0,
            index=pd.MultiIndex.from_product([range(1), self.Period_Date], names=[CFN.SCN, CFN.DAT]),
            columns=[CFN.NUM, CFN.DBT, CFN.YLD, CFN.AMT, CFN.ADV]
        )

        # --- save result cashflows ---
        res[CFN.NUM] = np.sum(self.NUM * self.StateC.Survival, axis=0)
        res[CFN.DBT] = np.sum(self.DBT * self.StateC.Survival, axis=0)
        res[CFN.YLD] = np.sum(self.YLD * self.StateC.Survival, axis=0)

        res[CFN.ADV] = np.sum(self.PMT * self.StateC.Survival, axis=0)
        res[CFN.ADV].values[:-1] += np.sum(self.DBT * self.StateP.weights, axis=0)[1:] + np.sum(self.DBT * self.StateD.weights, axis=0)[1:]
        res[CFN.AMT] = res[CFN.ADV] + np.sum(self.AMT * self.StateC.Survival, axis=0)

        return res

def run(inputs, lld_df, scr_df, seed=None):

    np.random.seed(seed)

    # --- build pool structure ---
    pool = CPool(inputs, lld_df)

    # --- update model risks with new time params ---
    pool.FlushWeights(scr_df)

    # --- compute cashflow forecast ---
    result = pool.RunScenario()

    # --- clear memory ---
    pool.Free()

    return result

