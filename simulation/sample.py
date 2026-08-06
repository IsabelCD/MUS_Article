
import pandas as pd
import numpy as np
from scipy.stats import norm
from .inclusion_probability import assign_hv_by_method
from .selection import select_sample
from .precision_estimation import precision_estimator

CONSERVATIVE_SAMPLING = ['con_systematic_sampling', 'sequential_list']

class Sample:
    """
    Draws and evaluates one sample from a population for one Monte Carlo
    iteration: assigns certainty (HV) units, selects the sample, projects
    the population error, and estimates precision.

    Parameters
    ----------
    population : pd.DataFrame
        The population for this iteration, with columns BV, E, ER.
    N : int
        Population size.
    EE : float
        True population error (used only for reporting, not estimation).
    BV : float
        Total population book value.
    sample_size : int
        Intended sample size.
    cl : float
        Confidence level (e.g. 0.80).
    z_score : float
        Normal critical value corresponding to `cl`.
    hv_selection : str
        Certainty-unit assignment method. "nothing" or "iterative".
    selection_type : str
        Method for selecting with PPS. "systematic_sampling" or "python".
    bound_estimator : str
        Precision/bound estimator used. See simulation/precision_estimation.py
        for the implemented options.
    hv_lookup : pd.Series, optional
        Precomputed "HV" values indexed by the population's row index, for
        hv_selection="iterative". The certainty-unit assignment depends only
        on BV and sample_size, not on row order, so it is identical for
        every Monte Carlo iteration of the same (population, sample_size);
        passing it in lets the caller compute it once per combination
        instead of recomputing it from scratch on every iteration.
    """

    def __init__(
        self,
        population: pd.DataFrame,
        N: int,
        EE: float,
        BV: float,
        sample_size: int,
        cl: float,
        z_score: float,
        hv_selection: str = "iterative",
        selection_type: str = "systematic_sampling",
        bound_estimator: str = "HH",
        random_state: int = 120,
        hv_lookup: pd.Series | None = None,
    ):
        #Simulation parameters
        self.population = population
        self.sample_size = sample_size
        self.z_score = z_score
        self.cl = cl
        self.rng = np.random.default_rng(random_state)

        #Type of method to apply
        self.hv_selection = hv_selection
        self.selection_type = selection_type
        self.bound_estimator = bound_estimator
        self.hv_lookup = hv_lookup

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
        self.ULE = None
        self.number_errors = None


    def assign_hv(self):
        if self.hv_selection == "iterative" and self.hv_lookup is not None:
            # HV assignment depends only on BV/sample_size, not row order,
            # so reuse the precomputed lookup instead of recomputing it.
            self.population = self.population.copy()
            self.population["HV"] = self.hv_lookup.reindex(self.population.index).to_numpy()
        else:
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
            "cl": self.cl
        }

        if self.bound_estimator == "HH" or self.bound_estimator == "Mod_HH":
            kwargs.update({
                "BVs": self.BVs,
                "ns": self.ns,
                "z_score": self.z_score,
                "SI": self.SI,
            })

        elif self.bound_estimator == "Poisson_Stringer":
            kwargs.update({
                "SI": self.SI,
                "cl": self.cl
            })

        elif self.bound_estimator == "Binomial_Stringer":
            kwargs.update({
                "BV": self.BV,
                "n": self.real_n
            })

        elif self.bound_estimator == "Moment":
            del kwargs["cl"]
            kwargs.update({
                "BVs": self.BVs,
                "ns": self.ns,
                "z_score": self.z_score,
            })

        self.SE, self.VAR, self.ULE = precision_estimator(
            bound_estimator=self.bound_estimator,
            **kwargs,
        )

    def number_of_sample_errors(self):
        self.number_errors = (self.sample_s["E"] > 0).sum()

    def run(self):
        self.assign_hv()
        self.select_sample()
        self.estimate_error()
        self.obtain_precision()
        self.number_of_sample_errors()

        return self
    
    def get_results(self):
        return {
            "EE_pred": self.EE_pred,
            "SE_pred": self.SE,
            "VAR_pred": self.VAR,
            "ULE_pred": self.ULE,
            "real_n": self.real_n, 
            "number_errors": self.number_errors,
        }