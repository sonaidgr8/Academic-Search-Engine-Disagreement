"""
Report results of all test run that followed our project structure.
Plot Box plots for development set best performance on single metric.
From best on development set select the best and report results on test set.
"""

import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
mpl.use('Agg')

def box_plot_matplotlib(result_dictionary, plot_order, filename, y_label):
    """
    Final plot using matplotlib.
    """
    group = []
    for name in plot_order:
        group.append([x for x in result_dictionary[name] if not math.isnan(x)])

    plt.clf()
    plt.rc('font', size=13)

    fig, axes = plt.subplots(ncols=1, sharey=True)
    fig.set_size_inches(20, 7)
    axes.boxplot(group,
                    widths=0.5,
                    whis='range')
    axes.set(xticklabels=order)
    axes.grid(True)
    axes.set_ylabel(y_label)
    axes.set_ylim([-0.05,0.6])
#    axes.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

    # adjust space at bottom
    fig.subplots_adjust(left=0.05, top=0.98, right=0.98, bottom=0.08, wspace=0)

    #plt.show()
    plt.savefig(filename)

#pylint: disable=invalid-name
if __name__ == '__main__':
    input_file = sys.argv[1]
    data = pd.read_csv(input_file)
    order = ['GS+SS', 'GS+MA', 'GS+SC', 'SS+MA', 'SS+SC', 'MA+SC']
    #print(data.to_dict(orient='list'))
    box_plot_matplotlib(data.to_dict(orient='list'),
                        order,
                        'box_plot.svg',
                        'intersection')
