"""plot_cscw.py

Plot for cscw.

"""

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 42})

import pandas as pd
import seaborn as sns

mpl.rc('legend', fontsize=16)
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rc('axes', labelsize=18)
mpl.rc('figure.subplot', left=0.15, bottom=0.15)


f_in = 'config.json_policies.json-n_authors_400.csv'
f_out = 'cscw.png'
final_policies = {
        'Random': '{"author_order": "highest_prob", "type": "random"}',
        'Greedy': '{"author_order": "highest_prob", "ties": "random", "type": "greedy"}',
        'Decision-theoretic Optimization': '{"type": "dt"}'}


def final_plot():

    df = pd.read_csv(f_in)
    df = df[df.policy.isin(final_policies.values())]
    
    for p in final_policies:
        df = df.replace(final_policies[p], p)

    column_v = 'u_true'
    ax = sns.tsplot(df, time='t', unit='iteration', condition='policy',
                       value=column_v, ci=95)


    h, l = ax.get_legend_handles_labels()
    d = {label: line for line, label in zip(h,l)}
    d['Greedy'].set_linestyle('--')
    d['Random'].set_linestyle(':')
    labels = ['Decision-theoretic Optimization',
              'Greedy',
              'Random']
    ax.legend([d[l] for l in labels], labels, loc='lower right')

    #plt.legend(title=None, loc='lower right')
    plt.xlabel('Contribution requests issued')
    plt.ylabel('Expected utility')



    plt.savefig(f_out)

if __name__ == '__main__':
    final_plot()
