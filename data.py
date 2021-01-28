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
                plt.figure(figsize=(4,3))

            # plt.subplot(1,2,1)
            handles = []
            if plot_humans:
                plt.errorbar(humans.stimuli, [np.median(s) for s in humans.RTs(humans.simple)], [sem(s) for s in humans.RTs(humans.simple)], color='gray', capsize=3, fmt="o", markerfacecolor="white", linestyle='-')

                # add model and human to the legend
                model_patch = mpatches.Patch(color='black', label='model')
                humans_patch = mpatches.Patch(color='gray', label='humans')
                handles = [model_patch, humans_patch]
            plt.errorbar(self.stimuli, [np.median(s) for s in self.RTs(self.simple)], [sem(s) for s in self.RTs(self.simple)], color='black', capsize=3, fmt="o", markerfacecolor="white", linestyle='-')
            handles_model, labels = plt.gca().get_legend_handles_labels()
            handles += handles_model
            
            
            plt.legend(handles=handles, ncol=2)

            
            
            plt.ylabel('Median reaction times (ms)', size=12)
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            
            if errorates:
                ax.spines['bottom'].set_color('none')
                plt.xticks([])
            else:
                plt.xticks(self.stimuli)
                plt.xlabel('Stimuli', size=12)

            

            if save_file is not None:
                plt.savefig(save_file+'_simple_rts.svg')
                plt.title(save_file)

            if show:
                plt.show()

        if errorates:
            if show:
                plt.figure(figsize=(4,1))

            # plt.subplot(1,2,2)
            bar_width = .5
            bar_x_offset = bar_width #+ .1
            plt.bar(self.stimuli-.5*bar_x_offset, self.error_rates(self.simple), color='black', width=bar_width)
            if plot_humans:
                plt.bar(humans.stimuli+.5*bar_x_offset, humans.error_rates(humans.simple), color='gray', width=bar_width)
            plt.xticks(self.stimuli)
            plt.xlabel('Stimuli', size=12)
            plt.ylabel('Error rates (%)', size=12)
            ax = plt.gca()
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.spines['left'].set_color('none')
            ax.spines['top'].set_color('none')
            if save_file is not None:
                plt.savefig(save_file+'_simple_errorrates.svg')
            if show:
                plt.show()

    def plot_fig2_chained(self, plot_humans=False, save_file=None, rts=True, errorates=True, show=True):

        if plot_humans:
            humans = Data(SackurData().exclusion)

        if rts:
            if show:
                plt.figure(figsize=(4,3))

            handles = []
            if plot_humans:
                plt.plot(humans.stimuli, [np.median(s) for s in humans.RTs(humans.congruent)], linestyle=(0, (5, 5)), color='gray')
                plt.plot(humans.stimuli, [np.median(s) for s in humans.RTs(humans.incongruent)], linestyle='dotted', color='gray')
                plt.errorbar(humans.stimuli, [np.median(s) for s in humans.RTs(humans.chained_add)], [sem(s) for s in humans.RTs(humans.chained_add)], color='gray', ls='none', capsize=3, fmt='^')
                plt.errorbar(humans.stimuli, [np.median(s) for s in humans.RTs(humans.chained_sub)], [sem(s) for s in humans.RTs(humans.chained_sub)], color='gray', ls='none', capsize=3, fmt='s', markerfacecolor='white')

                # add model and human to the legend
                model_patch = mpatches.Patch(color='black', label='model')
                humans_patch = mpatches.Patch(color='gray', label='humans')
                handles = [model_patch, humans_patch]

            plt.plot(self.stimuli, [np.median(s) for s in self.RTs(self.congruent)], linestyle=(0, (5, 5)), color='black', label='congruent')
            plt.plot(self.stimuli, [np.median(s) for s in self.RTs(self.incongruent)], linestyle='dotted', color='black', label='incongruent')
            plt.errorbar(self.stimuli, [np.median(s) for s in self.RTs(self.chained_add)], [sem(s) for s in self.RTs(self.chained_add)], color='black', ls='none', capsize=3, fmt='^', label='add')
            plt.errorbar(self.stimuli, [np.median(s) for s in self.RTs(self.chained_sub)], [sem(s) for s in self.RTs(self.chained_sub)], color='black', ls='none', capsize=3, fmt='s', label='sub', markerfacecolor='white')
            handles_model, labels = plt.gca().get_legend_handles_labels()
            handles += handles_model

            
            
            plt.legend(handles=handles, ncol=3)
            plt.xlabel('Stimuli', size=12)
            plt.ylabel('Median reaction times (ms)', size=12)
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            if errorates:
                ax.spines['bottom'].set_color('none')
                plt.xticks([])
            else:
                plt.xticks(self.stimuli)
                plt.xlabel('Stimuli', size=12)

            if save_file is not None:
                plt.savefig(save_file+'_chained_rts.svg')
                plt.title(save_file)
            if show:
                plt.show()


        if errorates:
            if show:
                plt.figure(figsize=(4,1))
            bar_width = .25
            bar_x_offset = bar_width #+ .1
            plt.bar(self.stimuli-1.5*bar_x_offset, self.error_rates(self.chained_sub), width=bar_width, label='sub', color='white', edgecolor='black')
            plt.bar(self.stimuli-.5*bar_x_offset, self.error_rates(self.chained_add), width=bar_width, label='add', color='black')

            congruent_x_offset = np.asarray([-.5*bar_x_offset if s in [2,6] else -1.5*bar_x_offset for s in self.stimuli])
            incongruent_x_offset = np.asarray([-.5*bar_x_offset if s not in [2,6] else -1.5*bar_x_offset for s in self.stimuli])
            plt.plot(self.stimuli + congruent_x_offset, self.error_rates(self.congruent), linestyle=(0, (5, 5)), color='black', label='congruent')
            plt.plot(self.stimuli + incongruent_x_offset, self.error_rates(self.incongruent), linestyle='dotted', color='black', label='incongruent')

            if plot_humans:
                congruent_x_offset = np.asarray([1.5*bar_x_offset if s in [2,6] else .5*bar_x_offset for s in self.stimuli])
                incongruent_x_offset = np.asarray([1.5*bar_x_offset if s not in [2,6] else .5*bar_x_offset for s in self.stimuli])
                plt.bar(humans.stimuli+.5*bar_x_offset, humans.error_rates(humans.chained_sub), width=bar_width, label='sub', color='white', edgecolor='gray')
                plt.bar(humans.stimuli+1.5*bar_x_offset, humans.error_rates(humans.chained_add), width=bar_width, label='add', color='gray')
                plt.plot(humans.stimuli + congruent_x_offset, humans.error_rates(humans.congruent), linestyle=(0, (5, 5)), color='gray', label='congruent')
                plt.plot(humans.stimuli + incongruent_x_offset, humans.error_rates(humans.incongruent), linestyle='dotted', color='gray', label='incongruent')


            plt.xticks(self.stimuli)
            plt.xlabel('Stimuli', size=12)
            plt.ylabel('Error rates (%)', size=12)
            ax = plt.gca()
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.spines['left'].set_color('none')
            ax.spines['top'].set_color('none')
            if save_file is not None:
                plt.savefig(save_file+'_chained_errorrates.svg')
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

    def mean_differences_error(self, compare_errorrates, compare_RTs, tasks): # tasks should be range(N_DIFFERENT_OPERATIONS)

        humans = Data(SackurData().exclusion)
        errorrate_errors = []
        RT_errors = []

        if compare_errorrates:
            errorrates_to_compare = [
                [
                    self.error_rates(self.simple),
                    self.error_rates(self.chained_add),
                    self.error_rates(self.chained_sub),
                ],

                [
                    humans.error_rates(humans.simple),
                    humans.error_rates(humans.chained_add),
                    humans.error_rates(humans.chained_sub),
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
                    errorrate_errors.append((humans_data[digit] - model_data[digit])**2)

            if compare_RTs:
                model_data = RTs_to_compare[0][i]
                humans_data = RTs_to_compare[1][i]
                for digit in range(4):
                    humans_median = np.median(humans_data[digit])
                    model_median = np.median(model_data[digit])
                    if np.isnan(model_median):
                        model_median = 2000 # arbitrary
                    RT_errors.append((humans_median - model_median)**2)

        error = np.sqrt(np.mean(errorrate_errors+RT_errors))
        return error
    

class SackurData(Data):
    def __init__(self, directory='..'):
        super().__init__(pyreadr.read_r(directory+'/sackur_data/exp1.R')['data'])



