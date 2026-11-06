"""Create three slide charts from training records only. AI-assisted code.
Run: .venv/Scripts/python.exe src/create_charts.py
"""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, MaxNLocator
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/train.csv'
OUTPUT = ROOT / 'charts'
PURPLE = '#5D65BB'
LIGHT = '#BCC1E6'
INK = '#181818'
BACKGROUND = '#F3F3F3'


def figure(title, subtitle):
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    fig.subplots_adjust(left=0.19, right=0.92, bottom=0.17, top=0.75)
    fig.text(0.06, 0.92, title, fontsize=25, weight='bold', color=INK)
    fig.text(0.06, 0.85, subtitle, fontsize=14, color='#555555')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#B9B9B9')
    ax.tick_params(axis='both', labelsize=14, length=0, pad=10)
    ax.set_axisbelow(True)
    return fig, ax


def main():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    data = pd.read_csv(SOURCE, dtype=str)
    assert data['number'].is_unique
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial', 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': '#555555',
                         'ytick.color': INK, 'svg.fonttype': 'none'})

    order = ['1 - Critical', '2 - High', '3 - Moderate', '4 - Low']
    counts = data['priority'].value_counts().reindex(order, fill_value=0)
    labels = [label.split(' - ')[1] for label in order]
    fig, ax = figure('Priority imbalance', f'Training incidents only (n = {counts.sum():,})')
    bars = ax.barh(labels, counts.values, color=PURPLE, height=0.56)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.28)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.grid(True, color='#DCDCDC', linewidth=0.7)
    ax.set_xlabel('Training incidents', fontsize=15, labelpad=15)
    for bar, count in zip(bars, counts):
        ax.text(count + counts.max() * 0.025, bar.get_y() + bar.get_height()/2,
                f'{count:,}  ({count / counts.sum():.1%})', va='center', fontsize=15)
    fig.savefig(OUTPUT / 'priority_imbalance.png', dpi=200, facecolor=BACKGROUND)
    plt.close(fig)

    groups = data['assignment_group'].dropna().value_counts()
    bins = pd.cut(groups, bins=[0, 4, 19, 99, 499, float('inf')],
                  labels=['1–4', '5–19', '20–99', '100–499', '500+'])
    support = bins.value_counts(sort=False)
    rare = int((groups < 20).sum())
    fig, ax = figure('Rare assignment groups',
                     f'{rare} of {len(groups)} training groups have fewer than 20 examples')
    fig.subplots_adjust(left=0.10)
    bars = ax.bar(support.index.astype(str), support.values,
                  color=[PURPLE, PURPLE, LIGHT, LIGHT, LIGHT], width=0.58)
    ax.yaxis.grid(True, color='#DCDCDC', linewidth=0.7)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.set_ylim(0, support.max() * 1.3)
    ax.set_xlabel('Training examples per group', fontsize=15, labelpad=15)
    ax.set_ylabel('Number of support groups', fontsize=15, labelpad=14)
    ax.bar_label(bars, labels=[f'{n} groups' for n in support], padding=10, fontsize=15)
    fig.savefig(OUTPUT / 'rare_assignment_groups.png', dpi=200, facecolor=BACKGROUND)
    plt.close(fig)

    top = groups.head(10)
    fig, ax = figure('Largest assignment groups',
                     f'Top 10 of {len(groups)} groups · {groups.sum():,} labelled training incidents')
    bars = ax.barh(top.index, top.values, color=PURPLE, height=0.64)
    ax.invert_yaxis()
    ax.set_xlim(0, top.max() * 1.28)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.grid(True, color='#DCDCDC', linewidth=0.7)
    ax.set_xlabel('Training incidents', fontsize=15, labelpad=15)
    for bar, count in zip(bars, top):
        ax.text(count + top.max() * 0.025, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=14)
    fig.text(0.06, 0.045, 'Remaining groups are not shown. Missing assignment labels are excluded.',
             fontsize=12, color='#555555')
    fig.savefig(OUTPUT / 'largest_assignment_groups.png', dpi=200, facecolor=BACKGROUND)
    plt.close(fig)

    assert support.sum() == len(groups)
    assert support.iloc[:2].sum() == rare
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == digest
    print('Priority training counts:', dict(counts))
    print('Routing training groups:', len(groups), '| Rare groups:', rare)
    print('Support bins:', dict(support))
    print('Training source SHA-256:', digest)
    print('Saved 3 PNG charts in:', OUTPUT)
    print('No data changes, model scores or resampling.')


if __name__ == '__main__':
    main()
