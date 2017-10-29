import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm

from analyse import computeROC

def plotROCs(rates_list, plot):
    fig = plt.figure(0)
    ax = fig.gca()
    # Random classifier
    ax.plot([0, 1], [0, 1], linestyle = '--', color = 'black')
    # Bound the curve above
    ax.plot([0, 1], [1, 1], linestyle = '--', color = 'black')
    # Label the graph
    ax.set_title('ROC Curves')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    # Set limits
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([-0.1, 1.1])
    # Turn on grid, add x-axis and y-axis
    ax.grid()
    ax.plot([-0.1, 1.1], [0, 0], linestyle = '--', linewidth = 2, color = 'black')
    ax.plot([0, 0], [-0.1, 1.1], linestyle = '--', linewidth = 2,  color = 'black')
    # Specify colors
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'purple']
    # Plot the ROC curves
    for model in range(6): 
        rates = rates_list[model]
        ax.plot(rates[0], rates[1], linewidth = 1, color = colors[model])
    red = plt.Line2D((0,1), (0,0), color = 'red', marker='o', linestyle = '')
    green = plt.Line2D((0,1), (0,0), color = 'green', marker='o', linestyle = '')
    blue = plt.Line2D((0,1), (0,0), color = 'blue', marker='o', linestyle = '')
    yellow = plt.Line2D((0,1), (0,0), color = 'yellow', marker='o', linestyle = '')
    cyan = plt.Line2D((0,1), (0,0), color = 'cyan', marker='o', linestyle = '')
    purple = plt.Line2D((0,1), (0,0), color = 'purple', marker='o', linestyle = '')
    ax.legend([red, green, blue, yellow, cyan, purple], ['1_10', '3_6', '4_16', '8_5', '16_8', '16_10'])
    plt.savefig(plot)

def getNormalDeviate(array):
    vals = []
    for i in range(len(array)):
        val = norm.ppf(array[i])
        if val > 2:
            val = 2
        elif val < -2:
            val = -2
        vals.append(val)
    return vals

def plotDETs(rates_list, plot):

    fig = plt.figure(1)
    ax = fig.gca()
    # Label the graph
    ax.set_title('DET Curves')
    ax.set_xlabel('FPR in %')
    ax.set_ylabel('FNR in %')
    # Set ticks
    vals = np.arange(0.0, 110.0, 10.0)
    ticks_name = [str(int(val)) for val in vals]
    ticks_pos = [norm.ppf(val / 100) for val in vals]
    ticks_pos[0], ticks_pos[-1] = -2, 2
    ax.xaxis.set_major_locator(ticker.FixedLocator((ticks_pos)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((ticks_name)))
    ax.yaxis.set_major_locator(ticker.FixedLocator((ticks_pos)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter((ticks_name)))
    # Turn on grid, add x-axis and y-axis
    ax.set_xlim([-2.1, 2])
    ax.set_ylim([-2.1, 2.1])
    ax.grid()
    # Set x-axis and y-axis
    ax.plot([-2.1, 2], [-2.0, -2.0], linestyle = '--', linewidth = 2, color = 'black')
    ax.plot([-2.0, -2.0], [-2.1, 2.1], linestyle = '--', linewidth = 2,  color = 'black')
    # Bound the curve above
    ax.plot([-2.1, 2.0], [2.0, 2.0], linestyle = '--', color = 'black')
    # Specify colors
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'purple']
    for model in range(6):
        fpr, tpr = rates_list[model]
        fnr = [1 - t for t in tpr]
        x, y = [], []
        x = getNormalDeviate(fpr)
        y = getNormalDeviate(fnr)   
        # Plot the DET curve 
        ax.plot(x, y, linewidth = 1, color = colors[model])

    red = plt.Line2D((0,1), (0,0), color = 'red', marker='o', linestyle = '')
    green = plt.Line2D((0,1), (0,0), color = 'green', marker='o', linestyle = '')
    blue = plt.Line2D((0,1), (0,0), color = 'blue', marker='o', linestyle = '')
    yellow = plt.Line2D((0,1), (0,0), color = 'yellow', marker='o', linestyle = '')
    cyan = plt.Line2D((0,1), (0,0), color = 'cyan', marker='o', linestyle = '')
    purple = plt.Line2D((0,1), (0,0), color = 'purple', marker='o', linestyle = '')
    ax.legend([red, green, blue, yellow, cyan, purple], ['1_10', '3_6', '4_16', '8_5', '16_8', '16_10'])
    plt.savefig(plot)


if __name__ == '__main__':
    rates_list = []
    for model in ['1_10', '3_6', '4_16', '8_5', '16_8', '16_10']:
        # Get the ROC data-points
        rates = computeROC('results/hmm/%s.txt' % model)
        # Add to rates_list
        rates_list.append(rates)
    plotROCs(rates_list, 'plots/hmm/roc.jpg')
    plotDETs(rates_list, 'plots/hmm/det.jpg')

