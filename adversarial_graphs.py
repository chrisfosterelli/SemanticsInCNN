#!/usr/bin/env python

""" Generate results graphs for the adversarial results """

import io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

results = """
0.5353535354	0.7070707071	0.69	0.6767676768	0.67	0.6060606061
0.5454545455	0.6666666667	0.75	0.7070707071	0.68	0.7474747475
0.5454545455	0.6363636364	0.72	0.7171717172	0.68	0.7373737374
0.5454545455	0.6565656566	0.62	0.6565656566	0.6	0.4949494949
0.5050505051	0.6565656566	0.59	0.6161616162	0.58	0.4545454545
0.5353535354	0.6616161616	0.595	0.6060606061	0.56	0.4494949495
0.5656565657	0.6666666667	0.6	0.595959596	0.54	0.4444444444
0.5151515152	0.6767676768	0.6	0.6666666667	0.59	0.5757575758
0.5353535354	0.6767676768	0.58	0.6161616162	0.57	0.4545454545
0.5656565657	0.6565656566	0.69	0.6161616162	0.58	0.5555555556
0.5656565657	0.5858585859	0.66	0.6161616162	0.55	0.5454545455
0.6161616162	0.6565656566	0.64	0.6464646465	0.59	0.5757575758
0.5656565657	0.5555555556	0.61	0.595959596	0.52	0.3333333333
0.6262626263	0.595959596	0.57	0.6262626263	0.5	0.404040404
0.5252525253	0.5757575758	0.52	0.5252525253	0.46	0.3737373737
0.5454545455	0.404040404	0.42	0.4444444444	0.41	0.3535353535
0.4646464646	0.3636363636	0.33	0.3939393939	0.33	0.3434343434
"""

if __name__ == '__main__':

    results = io.StringIO(results)
    results = np.loadtxt(results, delimiter="\t", skiprows=1)

    sns.set()
    sns.set(font_scale=1.5)
    sns.set_palette('plasma')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    layer_labels = [
        'ACT1', 'ACT2', 'ACT3', 'ACT4', 'ACT5', 'Mixed0', 'Mixed1', 'Mixed2',
        'Mixed3', 'Mixed4', 'Mixed5', 'Mixed6', 'Mixed7', 'Mixed8', 'Mixed9',
        'Mixed10', 'avg_pool'
    ]

    line_labels = (
        'Most similar', 'More similar', 'Mildly similar', 'Somewhat similar',
        'Less similar', 'Least similar', 'Average'
    )

    l1, = ax1.plot(results[:, 0])
    l2, = ax1.plot(results[:, 1])
    l3, = ax1.plot(results[:, 2])
    l4, = ax1.plot(results[:, 3])
    l5, = ax1.plot(results[:, 4])
    l6, = ax1.plot(results[:, 5])
    l7, = ax2.plot(np.average(results, axis=1), linestyle='--')
    ticks = list(range(len(layer_labels)))
    lines = (l1, l2, l3, l4, l5, l6, l7)

    ax1.set_xticks(ticks)
    ax1.set_xticklabels(layer_labels, rotation=45)
    ax1.set_ylabel('1 vs 2 Accuracy', labelpad=4.0, fontsize=20)
    ax1.axhline(y=0.50, color='#999999', linestyle=':')

    ax2.set_xticks(ticks)
    ax2.set_xticklabels(layer_labels, rotation=45)
    ax2.axhline(y=0.50, color='#999999', linestyle=':')
    plt.tick_params(axis='y', labelleft=False, labelright=True)

    fig.legend(lines, line_labels, ncol=7, bbox_to_anchor=(0.974, 0.945))
    plt.suptitle('Adversarial 1 vs 2 Accuracy Through Layers of InceptionV3')
    plt.subplots_adjust(wspace=0.03, left=0.045, right=0.955)
    plt.savefig('adversarial.png')
    plt.show()
