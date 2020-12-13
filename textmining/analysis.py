import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import os
from matplotlib import pyplot

import textwrap
import numpy as np
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
sns.set(style="whitegrid")

def get_train_test(df):
    training = (df[df['split_label'] == 'train']).copy()
    test = (df[df['split_label'] == 'test']).copy()

    return training, test


def get_df_high_effect(df, threshold=0.10):
    print(threshold)
    if threshold > 0:
        res_df = df[((df >= threshold) | (df <= (threshold * -1)))].copy()
        res_df.reset_index(inplace=True)
        res_df.drop_duplicates(keep='first', inplace=True)
        res_df.set_index('feature', inplace=True)
    else: res_df = df.copy()
    #res_df.dropna(axis=0, how='all', inplace=True)


    return res_df


def clean_feature_col(row):
    col = row['feature']
    col = col.replace('_scores_categories_', ' ')
    col = col.replace('liberal_', '')
    col = col.replace('conservative_', '')
    col = col.replace('_scores_percentiles_', ' % ')
    col = col.replace('_scores_raw_scores_', ' ')
    col = col.replace('emotional_analysis_emotional_tone_score', 'receptiviti emotional_tone')
    col = col.replace('_', ' ')
    col = col.replace('%', '')
    col = col.replace('mpqa arg lexicon ratio', 'arguing ratio')
    col = col.replace('sadness', 'emotion sadness')
    col = col.replace('negative', 'emotion:negative')
    col = col.replace('  ', ' ')
    col = col.replace(' ', ':')
    col = col.replace('nrc:', 'emotion:')
    col = col.replace('self:conscious', 'self_conscious')
    row['feature'] = col
    return row


def unify_df(df):
    for col in df.columns.values:

        effects = [x.strip() for x in col.split(' ')]

        if 'reinforcing' in effects and 'no_effect' in effects:
            new_name = 'reinforcing vs. ineffective'

            if col.startswith('no_effect') or col.startswith('ineffective'):
                df[col] = df[col] * -1

            df = df.rename(columns={col: new_name})
            continue

        if 'reinforcing' in effects and 'challenging' in effects:
            new_name = 'challenging vs. reinforcing'

            if col.startswith('reinforcing'):
                df[col] = df[col] * -1

            df = df.rename(columns={col: new_name})
            continue

        if 'challenging' in effects and 'no_effect' in effects:
            new_name = 'challenging vs. ineffective'

            if col.startswith('no_effect') or col.startswith('ineffective'):
                df[col] = df[col] * -1

            df = df.rename(columns={col: new_name})

            continue
    return df[['reinforcing vs. ineffective', 'challenging vs. ineffective', 'challenging vs. reinforcing']]

def get_sign_features_hm(orig_effect_sizes_df, threshold=0.10, drop_na=True):
    effect_sizes_df = orig_effect_sizes_df.copy()
    if drop_na:
        effect_sizes_df.dropna(axis=0, how='all', inplace=True)

    effect_sizes_df.reset_index(inplace=True)
    effect_sizes_df = effect_sizes_df.apply(clean_feature_col, axis=1)
    effect_sizes_df.set_index(['feature'], inplace=True)
    high_effect_sizes_df = get_df_high_effect(effect_sizes_df, threshold) if threshold > 0 else effect_sizes_df
    high_effect_sizes_df.fillna(0, inplace=True)

    high_effect_sizes_df.reset_index(inplace=True)

    high_effect_sizes_df.set_index(['feature'], inplace=True)
    high_effect_sizes_df =unify_df(high_effect_sizes_df)

    return high_effect_sizes_df




# sns.s
def plot_hm(df, filename=None):
    fig, ax = pyplot.subplots(figsize=(5, 14))


    max_width = 17
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    sns.set(style="whitegrid")


    # create s

    data = df

    ax = sns.heatmap(df, center=0, cmap="YlGnBu", xticklabels=True, yticklabels=True,
                     linewidths=0.05, square=True, cbar_kws={'label': 'Effect Size'},
                     vmin=-0.23, vmax=0.37, robust=False)

    ax.set_ylabel("Feature", fontsize=11
                  )

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    ax.figure.axes[-1].yaxis.label.set_size(10)

    x_axis = ['reinforcing vs. ineffective',
              'reinforcing vs. challenging',
              'ineffective vs. challenging']

    ax.set_xticklabels([textwrap.fill(x, max_width) for x in x_axis],
                       rotation=60, horizontalalignment="center", fontsize=10)

    for i in range(len(data.columns.values)):
        ax.axvline(i, color='white', lw=3)

    cbar = ax.collections[0].colorbar
    
    cbar.ax.tick_params(labelsize=10)

    plt.show()
    if filename is not None:
        root = '/'.join(os.getcwd().split('\\')[:-1]) + '/out/heatmaps/'
        Path(root).mkdir(parents=True, exist_ok=True)
        fig.savefig(root + filename, bbox_inches='tight')



