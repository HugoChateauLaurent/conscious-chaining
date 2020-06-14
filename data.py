import pyreadr
import pandas
import matplotlib.pyplot as plt
import scipy
from scipy.stats import sem
import numpy as np


## suj: participant
## ses: ("session"): training = 1 / main = 2
## block: larger blocks (1 -- 10)
## miniblock: smaller blocks of 20 trials (1 -- 30)
## respside: (1/2) withinn participant counterbalancing of response side
## rule: -2, 0 (simple comparison), +2
## stim: stimulus (2, 4, 6, 8)
## acc: accuracy (0: error / 1: correct)
## rt: response times (ms --- values of 0 are probably anticipations)
## cg: congruency (1: congruent / -1 incongruent); is the result of the application of the rule one the same side of 5 as the stimulus itself? meaningful only for rule !=0
## target: result of the application of the rule to the stimulus
## code: ?
## begend: ?
## newCode: ? 


class Data():
    def __init__(self, df):
        self.df = df

    @staticmethod
    def kl_div(a,b):
        q, bin_edges = np.histogram(b)
        p, bin_edges = np.histogram(a, bins=bin_edges)

        # normalize histograms
        p = p / p.sum()
        q = q / q.sum()

        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    @property
    def main(self):
        return self.df.loc[self.df['ses'] == 2]
    
    @property
    def training(self):
        return self.df.loc[self.df['ses'] == 2]

    @property
    def simple(self):
        return self.df.loc[self.df['rule'] == 0]

    @property
    def chained_add(self):
        return self.df.loc[self.df['rule'] == 2]

    @property
    def chained_sub(self):
        return self.df.loc[self.df['rule'] == -2]

    @property
    def stimuli(self):
        return [2,4,6,8]
    

    @property
    def simple_accuracies(self):
        return [self.simple.loc[self.simple['stim'] == s]['acc'].tolist() for s in self.stimuli]

    @property
    def simple_error_rates(self):
        return 100 - 100*self.simple_accuracies

    @property
    def simple_RTs(self):
        simples = [self.simple.loc[self.simple['stim'] == s]['rt'].tolist() for s in self.stimuli]
        return simples

    @property
    def chained_add_accuracies(self):
        return [self.chained_add.loc[self.chained_add['stim'] == s]['acc'].tolist() for s in self.stimuli]

    @property
    def chained_add_error_rates(self):
        return 100 - 100*self.chained_add_accuracies

    @property
    def chained_add_RTs(self):
        chained_adds = [self.chained_add.loc[self.chained_add['stim'] == s]['rt'].tolist() for s in self.stimuli]
        return chained_adds

    @property
    def chained_sub_accuracies(self):
        return [self.chained_sub.loc[self.chained_sub['stim'] == s]['acc'].tolist() for s in self.stimuli]

    @property
    def chained_sub_error_rates(self):
        return 100 - 100*self.chained_sub_accuracies

    @property
    def chained_sub_RTs(self):
        return [self.chained_sub.loc[self.chained_sub['stim'] == s]['rt'].tolist() for s in self.stimuli]

    @property
    def fitness_error(self):
        empirical = Data(SackurData().main)
        error = 0

        
        to_compare = [
            [
                self.simple_accuracies, 
                self.simple_RTs, 
                self.chained_add_accuracies, 
                self.chained_add_RTs, 
                self.chained_sub_accuracies, 
                self.chained_sub_RTs
            ],

            [
                empirical.simple_accuracies, 
                empirical.simple_RTs, 
                empirical.chained_add_accuracies, 
                empirical.chained_add_RTs, 
                empirical.chained_sub_accuracies, 
                empirical.chained_sub_RTs
            ]
        ]

        assert len(to_compare[0]) == len(to_compare[1])

        for i in range(len(to_compare[0])):
            model_data = to_compare[0][i]
            empirical_data = to_compare[1][i]
            for digit in range(4):
                error += Data.kl_div(model_data[digit], empirical_data[digit])

        return error
    

class SackurData(Data):
    def __init__(self):
        super().__init__(pyreadr.read_r('../sackur_data/exp1.R')['data'])