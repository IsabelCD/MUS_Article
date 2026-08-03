
import pandas as pd
import numpy as np
from scipy.stats import norm
from .inclusion_probability import assign_hv_by_method
from .selection import select_sample
from .precision_estimation import precision_estimator

CONSERVATIVE_SAMPLING = ['con_systematic_sampling', 'sequential_list']

class Sample:
    """
    XX

    Parameters
    ----------
    population : pd.DataFrame
        The actual population, with columns BV, E, ER, Q 
    sample_size : int
        Intended sample size
    CL : float
        Confidence interval. Default is 80%
    method : str
        Sampling method. Can be MUS or MRS
    selection_type : str 
        Method for selecting with PPS
    bound_estimator : str
        Bound estimator used. Default is HH
    """

    def __init__(
        self,
        population: pd.DataFrame,
        N: int,
        EE: float,
        BV: float,
        sample_size: int,
        z_score: float,
        method: str = "MUS",
        hv_selection: str = "iterative",
        selection_type: str = "systematic_sampling",
        bound_estimator: str = "HH",
        random_state: int = 120
    ):
        #Simulation parameters
        self.population = population
        self.sample_size = sample_size 
        self.z_score = z_score
        self.rng = np.random.default_rng(random_state)

        #Type of method to apply
        self.method = method
        self.hv_selection = hv_selection
        self.selection_type = selection_type
        self.bound_estimator = bound_estimator
        
        #Population characteristics
        self.N = N
        self.EE = EE
        self.BV = BV

        #Sample characteristics
        self.sample = None
        self.sample_s = None
        self.real_n = None #Real sample size
        self.BVs = None
        self.ns = None
        self.SI = None

        #Prediction results
        self.EE_pred = None
        self.SE = None
        self.VAR = None
        self.LLE = None
        self.ULE = None


    def assign_hv(self):
        kwargs = {"population": self.population,}

        if self.hv_selection == "iterative":
            kwargs.update({
                "BV": self.BV,
                "n": self.sample_size,
            })

        self.population = assign_hv_by_method(
            hv_selection = self.hv_selection,
            **kwargs,
            )

        self.BVs = self.population[self.population["HV"] != 1]["BV"].sum()
        self.ns = self.sample_size - int((self.population["HV"] == 1).sum())
        self.SI = self.BVs / self.ns

    def select_sample(self):
        if self.selection_type == "systematic_sampling":
            kwargs = {"SI": self.SI}

        elif self.selection_type == "python":
            kwargs = {"n": self.sample_size}

        self.sample = select_sample(selection_type = self.selection_type,
                                    population = self.population,
                                    rng = self.rng,
                                    **kwargs,)

        #Update sample information
        self.real_n = self.sample.shape[0]
        self.ns = self.real_n - int((self.population["HV"] == 1).sum()) 
        self.BVs = self.population[self.population["HV"] != 1]["BV"].sum()
        self.sample['HV'] = np.where(self.sample['BV']>self.SI, 1, self.sample['HV'])         #TODO: see for no HV separation if ns and BVs is updated, rn it is not


    
    def estimate_error(self):
        self.sample_s = self.sample[self.sample["HV"] != 1].copy()

        # error in certainty stratum
        EEe = sum(self.sample[self.sample["HV"]==1]['E'])         
        # projected error in sampling stratum
        self.sample_s["EQ_ratio"] = self.sample_s["E"] / self.sample_s["BV"]
        EEs = self.SI * self.sample_s["EQ_ratio"].sum() 
        # error estimation    
        self.EE_pred = EEe + EEs


    def obtain_precision(self):
        kwargs = {
            "sample_s": self.sample_s,
            "EE": self.EE_pred,
        }

        if self.bound_estimator == "HH" or self.bound_estimator == "Mod_HH":
            kwargs.update({
                "BVs": self.BVs,
                "ns": self.ns,
                "z_score": self.z_score,
            })

        elif self.bound_estimator == "Con":
            kwargs.update({
                "SI": self.SI,
            })

        elif self.bound_estimator == "Ratio":
            kwargs.update({
                "BVs": self.BVs,
                "ns": self.ns,
                "z_score": self.z_score,
            })

        self.SE, self.VAR, self.LLE, self.ULE = precision_estimator(
            bound_estimator=self.bound_estimator,
            **kwargs,
        )

    def run(self):
        self.assign_hv()
        self.select_sample()
        self.estimate_error()
        self.obtain_precision()

        return self
    
    def get_results(self):
        return {
            "EE_pred": self.EE_pred,
            "SE_pred": self.SE,
            "VAR_pred": self.VAR,
            "LLE_pred": self.LLE,
            "ULE_pred": self.ULE,
            "real_n": self.real_n
        }