"""plot.py"""
import os
import itertools
import collections
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import scipy.stats as ss
import csv

def significance_x(df, policy, v):
    """Find how many fewer requests policy needs to reach max value of other
    policies.
    
    Uses a unequal-variance paired t-test.
    
    """
    all_policies = df['policy'].unique()
    if policy not in all_policies:
        raise Exception('Could not find policy')
    other_policies = [p for p in all_policies if p != policy]
    max_iter = df['iteration'].max()
    max_t = df['t'].max()
    values = collections.defaultdict(list)
    for p in other_policies:
        for i in xrange(max_iter + 1):
            max_value = df[(df.iteration == i) & (df.policy == p)][v].max()
            t = 0
            vals = df[(df.iteration == i) & (df.policy == policy)].sort('t')[v]
            vals = list(vals)
            max_value_p = vals[t]
            while max_value_p < max_value:
                t += 1
                print i, t, p, max_value_p, max_value
                max_value_p = vals[t]

            values[p].append((t, max_t))

    res = []
    for p in other_policies:
        v1, v2 = zip(**values[p])
        diffs = [x2 - x1 for x1, x2 in zip(v1, v2)]
        u1 = np.mean(v1)
        u2 = np.mean(v2)
        tval, pval = ss.ttest_ind(v1, v2, equal_var=False)

        res.append({'pol1': pol1,
                    'pol2': pol2,
                    'diff': np.mean(diffs),
                    'u1': u1,
                    'u2': u2,
                    'tval': tval,
                    'pval': pval})
    return res

def significance(df, t, v):
    """Find significance of all pairs of policies at the given timestep.
    
    Uses a unequal-variance paired t-test.
    
    """
    df = df[df.t == t].sort('iteration')
    policies = df['policy'].unique()
    max_iter = df['iteration'].max()
    res = []
    for pol1, pol2 in itertools.combinations(policies, 2):
        v1 = df[df.policy == pol1][v]
        v2 = df[df.policy == pol2][v]
        u1 = np.mean(v1)
        u2 = np.mean(v2)
        tval, pval = ss.ttest_ind(v1, v2, equal_var=False)

        res.append({'pol1': pol1,
                    'pol2': pol2,
                    't': t,
                    'u1': u1,
                    'u2': u2,
                    'tval': tval,
                    'pval': pval})
    return res

def plot_from_file(f, outname):
    df = pd.read_csv(f)
    for s in ['true', 'estimated']:
        column_v = 'u_{}'.format(s)
        sns.tsplot(df, time='t', unit='iteration', condition='policy',
                   value=column_v, ci=95)
        plt.xlabel('Contribution requests issued')
        plt.ylabel('Expected utility')
        plt.savefig('{}_{}.png'.format(outname, s))
        plt.close()
        
        # Save raw values
        df.groupby(['policy', 't'])[column_v].agg(np.mean).reset_index().sort(['policy', 't']).to_csv('{}_{}_means.csv'.format(outname, s), index=False)

        sigs = significance(df, df['t'].max(), column_v)
        with open('{}_{}_sig.csv'.format(outname, s), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'pol1', 'pol2', 't', 'u1', 'u2', 'tval', 'pval'])
            writer.writeheader()
            writer.writerows(sigs)


        """
        sigs = significance_x(df, '{"type": "dt"}', column_v)
        with open('{}_{}_sig_x.csv'.format(outname, s), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'pol1', 'pol2', 'diff', 'u1', 'u2', 'tval', 'pval'])
            writer.writeheader()
            writer.writerows(sigs)
        """

    print df.groupby('policy')['iteration'].agg(lambda x: len(set(x)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', '-i', type=argparse.FileType('r'),
                        required=True)
    parser.add_argument('--outfile', '-o', type=str)
    args = parser.parse_args()

    if args.outfile is None:
        args.outfile = os.path.basename(args.infile.name)
    plot_from_file(args.infile, args.outfile)
