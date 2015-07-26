"""plot.py"""
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_from_file(f, outname):
    df = pd.read_csv(f)
    sns.tsplot(df, time='t', unit='iteration', condition='policy', value='v',
               ci=95)
    plt.savefig(outname)

if __name__ == '__main__':
    plot_from_file('out.csv', 'out.png')
