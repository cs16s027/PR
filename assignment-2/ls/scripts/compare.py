import h5py
from matplotlib import pyplot as plt

from helpers import loadModel
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
    colors = ['red', 'green', 'blue', 'yellow', 'cyan']
    # Plot the ROC curves
    for model in range(5): 
        rates = rates_list[model]
        ax.plot(rates[0], rates[1], linewidth = 1, color = colors[model])
    red = plt.Line2D((0,1), (0,0), color = 'red', marker='o', linestyle = '')
    green = plt.Line2D((0,1), (0,0), color = 'green', marker='o', linestyle = '')
    blue = plt.Line2D((0,1), (0,0), color = 'blue', marker='o', linestyle = '')
    yellow = plt.Line2D((0,1), (0,0), color = 'yellow', marker='o', linestyle = '')
    cyan = plt.Line2D((0,1), (0,0), color = 'cyan', marker='o', linestyle = '')
    ax.legend([red, green, blue, yellow, cyan], ['C-1', 'C-2', 'C-3', 'C-4', 'C-5'])
    plt.savefig(plot)

def compare(data, roc_plot):
    print 'Compare models on validation data'
    rates_list = []
    for model in ['C-1', 'C-2', 'C-3', 'C-4', 'C-5']:
        params = loadModel('models/%s/model.h5' % model)
        # Get the ROC data-points
        rates = computeROC(data, params)
        # Add to rates_list
        rates_list.append(rates)
    plotROCs(rates_list, roc_plot)

if __name__ == '__main__':
    data = h5py.File('data/valid.h5', 'r')
    compare(data, 'plots/compare.jpg')

