import pyreadr
import pandas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
        return self.df.loc[self.df['ses'] == 1]
    
    @property
    def exclusion(self):
        assert type(self) is SackurData
        return self.df.loc[self.df['suj'] != 6]

    @property
    def simple(self):
        return self.df.loc[self.df['rule'] == 0]

    @property
    def chained(self):
        return self.df.loc[self.df['rule'] != 0]

    @property
    def chained_add(self):
        return self.chained.loc[self.chained['rule'] == 2]

    @property
    def chained_sub(self):
        return self.chained.loc[self.chained['rule'] == -2]

    @property
    def congruent(self):
        return self.chained.loc[self.chained['cg'] == 1]

    @property
    def incongruent(self):
        return self.chained.loc[self.chained['cg'] == -1]

    @property
    def stimuli(self):
        return np.array([2,4,6,8])

    def accuracies(self, data):
        return [data.loc[data['stim'] == s]['acc'].tolist() for s in self.stimuli]

    def error_rates(self, data):
        return 100 - 100*np.mean(self.accuracies(data), axis=1)

    def RTs(self, data):
        return [data.loc[(data['stim'] == s) & (data['acc'] == 1)]['rt'].tolist() for s in self.stimuli]

    def plot_fig2_simple(self, plot_humans=False, save_file=None, rts=True, errorates=True, show=True):

        if plot_humans:
            humans = Data(SackurData().exclusion)
        
        if rts:
            if show:
                plt.figure(figsize=(6,4))

            # plt.subplot(1,2,1)
            plt.errorbar(self.stimuli, [np.median(s) for s in self.RTs(self.simple)], [sem(s) for s in self.RTs(self.simple)], color='black', capsize=3, fmt="o", markerfacecolor="white", linestyle='-')
            handles, labels = plt.gca().get_legend_handles_labels()
            if plot_humans:
                plt.errorbar(humans.stimuli, [np.median(s) for s in humans.RTs(humans.simple)], [sem(s) for s in humans.RTs(humans.simple)], color='gray', capsize=3, fmt="o", markerfacecolor="white", linestyle='-')

                # add model and human to the legend
                model_patch = mpatches.Patch(color='black', label='model')
                humans_patch = mpatches.Patch(color='gray', label='humans')
                handles = [model_patch, humans_patch] + handles
            
            plt.legend(handles=handles)

            # plt.xticks(self.stimuli)
            # plt.xlabel('Stimuli')
            plt.xticks([])
            plt.ylabel('Median reaction times (ms)')
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            

            if save_file is not None:
                plt.savefig(save_file+'_simple_rts.eps', format='eps')
                plt.title(save_file)

            if show:
                plt.show()

        if errorates:
            if show:
                plt.figure(figsize=(6,4))

            # plt.subplot(1,2,2)
            plt.bar(self.stimuli, self.error_rates(self.simple), color='black')
            plt.xticks(self.stimuli)
            plt.xlabel('Stimuli')
            plt.ylabel('Error rates (%)')
            ax = plt.gca()
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.spines['left'].set_color('none')
            ax.spines['top'].set_color('none')
            if save_file is not None:
                plt.savefig(save_file+'_simple_errorrates.eps', format='eps')
            if show:
                plt.show()

    def plot_fig2_chained(self, plot_humans=False, save_file=None, rts=True, errorates=True, show=True):

        if plot_humans:
            humans = Data(SackurData().exclusion)

        if rts:
            if show:
                plt.figure(figsize=(6,4))

            plt.plot(self.stimuli, [np.median(s) for s in self.RTs(self.congruent)], linestyle=(0, (5, 5)), color='black', label='congruent')
            plt.plot(self.stimuli, [np.median(s) for s in self.RTs(self.incongruent)], linestyle='dotted', color='black', label='incongruent')
            plt.errorbar(self.stimuli, [np.median(s) for s in self.RTs(self.chained_add)], [sem(s) for s in self.RTs(self.chained_add)], color='black', ls='none', capsize=3, fmt='^', label='add')
            plt.errorbar(self.stimuli, [np.median(s) for s in self.RTs(self.chained_sub)], [sem(s) for s in self.RTs(self.chained_sub)], color='black', ls='none', capsize=3, fmt='s', label='sub', markerfacecolor='white')
            handles, labels = plt.gca().get_legend_handles_labels()

            if plot_humans:
                plt.plot(humans.stimuli, [np.median(s) for s in humans.RTs(humans.congruent)], linestyle=(0, (5, 5)), color='gray')
                plt.plot(humans.stimuli, [np.median(s) for s in humans.RTs(humans.incongruent)], linestyle='dotted', color='gray')
                plt.errorbar(humans.stimuli, [np.median(s) for s in humans.RTs(humans.chained_add)], [sem(s) for s in humans.RTs(humans.chained_add)], color='gray', ls='none', capsize=3, fmt='^')
                plt.errorbar(humans.stimuli, [np.median(s) for s in humans.RTs(humans.chained_sub)], [sem(s) for s in humans.RTs(humans.chained_sub)], color='gray', ls='none', capsize=3, fmt='s', markerfacecolor='white')

                # add model and human to the legend
                model_patch = mpatches.Patch(color='black', label='model')
                humans_patch = mpatches.Patch(color='gray', label='humans')
                handles = [model_patch, humans_patch] + handles
            
            plt.legend(handles=handles)
            plt.xticks([])
            plt.xlabel('Stimuli')
            plt.ylabel('Median reaction times (ms)')
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')

            if save_file is not None:
                plt.savefig(save_file+'_chained_rts.eps', format='eps')
                plt.title(save_file)
            if show:
                plt.show()


        if errorates:
            if show:
                plt.figure(figsize=(6,4))
            bar_width = .5
            bar_x_offset = bar_width + .1
            plt.bar(self.stimuli, self.error_rates(self.chained_sub), width=bar_width, label='sub', color='white', edgecolor='black')
            plt.bar(self.stimuli+bar_x_offset, self.error_rates(self.chained_add), width=bar_width, label='add', color='black')

            congruent_x_offset = [bar_x_offset if s in [2,6] else 0 for s in self.stimuli]
            incongruent_x_offset = [bar_x_offset if s not in [2,6] else 0 for s in self.stimuli]
            plt.plot(self.stimuli + congruent_x_offset, self.error_rates(self.congruent), linestyle=(0, (5, 5)), color='black', label='congruent')
            plt.plot(self.stimuli + incongruent_x_offset, self.error_rates(self.incongruent), linestyle='dotted', color='black', label='incongruent')

            plt.xticks(self.stimuli)
            plt.xlabel('Stimuli')
            plt.ylabel('Error rates (%)')
            ax = plt.gca()
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.spines['left'].set_color('none')
            ax.spines['top'].set_color('none')
            if save_file is not None:
                plt.savefig(save_file+'_chained_errorrates.eps', format='eps')
                plt.title(save_file)
            if show:
                plt.show()

    @property
    def error_rate(self):
        return 1-self.df['acc'].mean()

    @property
    def kl_fitness_error(self):
        humans = Data(SackurData().exclusion)
        error = 0

        
        to_compare = [
            [
                self.accuracies(self.simple), 
                self.RTs(self.simple), 
                self.accuracies(self.chained_add), 
                self.RTs(self.chained_add), 
                self.accuracies(self.chained_sub), 
                self.RTs(self.chained_sub)
            ],

            [
                humans.accuracies(humans.simple), 
                humans.RTs(humans.simple), 
                humans.accuracies(humans.chained_add), 
                humans.RTs(humans.chained_add), 
                humans.accuracies(humans.chained_sub), 
                humans.RTs(humans.chained_sub)
            ]
        ]

        assert len(to_compare[0]) == len(to_compare[1])

        for i in range(len(to_compare[0])):
            model_data = to_compare[0][i]
            humans_data = to_compare[1][i]
            for digit in range(4):
                error += Data.kl_div(model_data[digit], humans_data[digit])

        return error

    def rmse_fitness_error(self, compare_errorrates, compare_RTs, tasks): # tasks should be range(N_DIFFERENT_OPERATIONS)
        humans = Data(SackurData().exclusion)
        errorrate_errors = []
        RT_errors = []

        if compare_errorrates:
            errorrates_to_compare = [
                [
                    self.accuracies(self.simple),
                    self.accuracies(self.chained_add),
                    self.accuracies(self.chained_sub),
                ],

                [
                    humans.accuracies(humans.simple),
                    humans.accuracies(humans.chained_add),
                    humans.accuracies(humans.chained_sub),
                ]
            ]

        if compare_RTs:
            RTs_to_compare = [
                [
                    self.RTs(self.simple), 
                    self.RTs(self.chained_add), 
                    self.RTs(self.chained_sub)
                ],

                [
                    humans.RTs(humans.simple), 
                    humans.RTs(humans.chained_add), 
                    humans.RTs(humans.chained_sub)
                ]
            ]

        for i in tasks:

            if compare_errorrates:
                model_data = errorrates_to_compare[0][i]
                humans_data = errorrates_to_compare[1][i]
                for digit in range(4):
                    humans_mean = np.mean(humans_data[digit])
                    for data in model_data[digit]:
                        errorrate_errors.append((humans_mean - data)**2)

            if compare_RTs:
                model_data = RTs_to_compare[0][i]
                humans_data = RTs_to_compare[1][i]
                for digit in range(4):
                    humans_mean = np.mean(humans_data[digit])
                    for data in model_data[digit]:
                        RT_errors.append((humans_mean - data)**2)

        return np.sqrt(np.mean(errorrate_errors+RT_errors)), errorrate_errors, RT_errors

    def mean_differences_error(self, compare_errorrates, compare_RTs, tasks): # tasks should be range(N_DIFFERENT_OPERATIONS)
        humans = Data(SackurData().exclusion)
        errorrate_errors = []
        RT_errors = []

        if compare_errorrates:
            errorrates_to_compare = [
                [
                    self.accuracies(self.simple),
                    self.accuracies(self.chained_add),
                    self.accuracies(self.chained_sub),
                ],

                [
                    humans.accuracies(humans.simple),
                    humans.accuracies(humans.chained_add),
                    humans.accuracies(humans.chained_sub),
                ]
            ]

        if compare_RTs:
            RTs_to_compare = [
                [
                    self.RTs(self.simple), 
                    self.RTs(self.chained_add), 
                    self.RTs(self.chained_sub)
                ],

                [
                    humans.RTs(humans.simple), 
                    humans.RTs(humans.chained_add), 
                    humans.RTs(humans.chained_sub)
                ]
            ]

        for i in tasks:

            if compare_errorrates:
                model_data = errorrates_to_compare[0][i]*100
                humans_data = errorrates_to_compare[1][i]*100
                for digit in range(4):
                    humans_mean = np.mean(humans_data[digit])
                    model_mean = np.mean(model_data[digit])
                    # if model_mean == 0:
                    #     return np.inf
                    errorrate_errors.append((humans_mean - model_mean)**2)

            if compare_RTs:
                model_data = RTs_to_compare[0][i]
                humans_data = RTs_to_compare[1][i]
                for digit in range(4):
                    humans_mean = np.mean(humans_data[digit])
                    model_mean = np.mean(model_data[digit])
                    if np.isnan(model_mean):
                        model_mean = 2000 # arbitrary
                    RT_errors.append((humans_mean - model_mean)**2)

        error = np.sqrt(np.mean(errorrate_errors+RT_errors))
        return error

    # def custom_error(self, compare_errorrates, compare_RTs, tasks): # tasks should be range(N_DIFFERENT_OPERATIONS)
    #     humans = Data(SackurData().exclusion)

    #     errorrate_mean_differences = []
    #     rt_mean_differences = []

    #     if compare_errorrates:
    #         errorrates_to_compare = [
    #             [
    #                 self.accuracies(self.simple),
    #                 self.accuracies(self.chained_add),
    #                 self.accuracies(self.chained_sub),
    #                 self.accuracies(self.congruent),
    #                 self.accuracies(self.incongruent),
    #             ],

    #             [
    #                 humans.accuracies(humans.simple),
    #                 humans.accuracies(humans.chained_add),
    #                 humans.accuracies(humans.chained_sub),
    #                 humans.accuracies(humans.congruent),
    #                 humans.accuracies(humans.incongruent),
    #             ]
    #         ]

    #     if compare_RTs:
    #         RTs_to_compare = [
    #             [
    #                 self.RTs(self.simple), 
    #                 self.RTs(self.chained_add), 
    #                 self.RTs(self.chained_sub)
    #                 self.RTs(self.congruent)
    #                 self.RTs(self.incongruent)
    #             ],

    #             [
    #                 humans.RTs(humans.simple), 
    #                 humans.RTs(humans.chained_add), 
    #                 humans.RTs(humans.chained_sub)
    #                 humans.RTs(humans.congruent)
    #                 humans.RTs(humans.incongruent)
    #             ]
    #         ]

    #     for i in tasks:

    #         if compare_errorrates:
    #             model_data = errorrates_to_compare[0][i]
    #             humans_data = errorrates_to_compare[1][i]
    #             for digit in range(4):
    #                 humans_mean = np.mean(humans_data[digit])
    #                 model = np.mean(model_data[digit])

    #                 for data in model_data[digit]:
    #                     errorrate_errors.append((humans_mean - data)**2)

    #         if compare_RTs:
    #             model_data = RTs_to_compare[0][i]
    #             humans_data = RTs_to_compare[1][i]
    #             for digit in range(4):
    #                 humans_mean = np.mean(humans_data[digit])
    #                 for data in model_data[digit]:
    #                     RT_errors.append((humans_mean - data)**2)

    #     return np.sqrt(np.mean(errorrate_errors+RT_errors)), errorrate_errors, RT_errors
    

class SackurData(Data):
    def __init__(self):
        super().__init__(pyreadr.read_r('../sackur_data/exp1.R')['data'])



