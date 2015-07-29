"""plot.py"""
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse

def plot_from_file(f, outname):
    df = pd.read_csv(f)
    for s in ['true', 'estimated']:
        sns.tsplot(df, time='t', unit='iteration', condition='policy',
                   value='u_{}'.format(s), ci=95)
        plt.xlabel('Contribution requests issued')
        plt.ylabel('Expected utility')
        plt.savefig('{}_{}.png'.format(outname, s))
        plt.close()

    print df.groupby('policy')['iteration'].agg(lambda x: len(set(x)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', '-i', type=argparse.FileType('r'),
                        required=True)
    parser.add_argument('--outfile', '-o', type=str, required=True)
    args = parser.parse_args()
    plot_from_file(args.infile, args.outfile)
