import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm

# ROC computations
def computeROC(results):
    mapping = {'1' : 0, '5' : 1, 'z' : 2}
    #mapping = {'0' : 0, '1' : 1, '2' : 2}
    results = [line.strip().split() for line in open(results, 'r').readlines()]
    for index in range(len(results)):
        results[index][0] = mapping[results[index][0]]
        for i in [1, 2, 3]:    
            results[index][i] = float(results[index][i])

    # ROC curve computation
    thresholds_left = [1.0 / 10**i for i in np.arange(16, 3, -1)]
    thresholds_central = list(np.arange(0.001, 1.0, 0.001))
    thresholds_right = [1 - l for l in thresholds_left[::-1]]
    thresholds = [0] + thresholds_left + thresholds_central + thresholds_right + [1.0]
    tprs, fprs = [], []

    # Choose one threshold - one point on the ROC curve
    epsilon = 0.00001
    for threshold in thresholds:
        roc = {}
        # Choose the target class 
        for roc_label in [0, 1, 2]:
            tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
            # Run through the data
            for result in results:
                prob = result[roc_label + 1]
                ground_truth = result[0]
                # roc_label_int is positive (+1), the other two classes are negative (-1)
                if ground_truth == roc_label:
                    if prob > threshold:
                        tp += 1
                    else:
                        fn += 1
                if ground_truth != roc_label:
                    if prob > threshold:
                        fp += 1
                    else:
                        tn += 1
            
            roc[roc_label] = (tp / (tp + fn + epsilon), fp / (fp + tn + epsilon))
        # Macro average of the ROC values
        tprs.append(np.mean([roc[roc_label][0] for roc_label in [0, 1, 2]]))
        fprs.append(np.mean([roc[roc_label][1] for roc_label in [0, 1, 2]]))
        
    return (fprs, tprs)

def plotROC(rates, plot, title, figure):
    fig = plt.figure(figure)
    ax = fig.gca()
    # Random classifier
    ax.plot([0, 1], [0, 1], linestyle = '--', color = 'black')
    # Bound the curve above
    ax.plot([0, 1], [1, 1], linestyle = '--', color = 'black')
    # Label the graph
    ax.set_title(title)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    # Set limits
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([-0.1, 1.1])
    # Turn on grid, add x-axis and y-axis
    ax.grid()
    ax.plot([-0.1, 1.1], [0, 0], linestyle = '--', linewidth = 2, color = 'black')
    ax.plot([0, 0], [-0.1, 1.1], linestyle = '--', linewidth = 2,  color = 'black')
    # Plot the ROC curve 
    ax.plot(rates[0], rates[1], linewidth = 1)
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

def plotDET(rates, plot, title, figure):

    fpr, tpr = rates
    fnr = [1 - t for t in tpr]
    x, y = [], []
    x = getNormalDeviate(fpr)
    y = getNormalDeviate(fnr)
    fig = plt.figure(figure)
    ax = fig.gca()
    # Label the graph
    ax.set_title(title)
    ax.set_xlabel('FPR in %')
    ax.set_ylabel('FNR in %')
    # Set ticks
    vals = np.arange(0.0, 110.0, 10.0)
    ticks_name = [str(int(val)) for val in vals]
    print ticks_name
    ticks_pos = [norm.ppf(val / 100) for val in vals]
    ticks_pos[0], ticks_pos[-1] = -2, 2
    ax.xaxis.set_major_locator(ticker.FixedLocator((ticks_pos)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((ticks_name)))
    ax.yaxis.set_major_locator(ticker.FixedLocator((ticks_pos)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter((ticks_name)))
    # Turn on grid, add x-axis and y-axis
    print ticks_pos
    ax.set_xlim([-2.1, 2])
    ax.set_ylim([-2.1, 2.1])
    ax.grid()
    # Set x-axis and y-axis
    ax.plot([-2.1, 2], [-2.0, -2.0], linestyle = '--', linewidth = 2, color = 'black')
    ax.plot([-2.0, -2.0], [-2.1, 2.1], linestyle = '--', linewidth = 2,  color = 'black')
    # Bound the curve above
    ax.plot([-2.1, 2.0], [2.0, 2.0], linestyle = '--', color = 'black')
    # Plot the DET curve 
    ax.plot(x, y, linewidth = 1)
    plt.savefig(plot)

def printConfusionMatrix(data, params):
    predictions = predict(data, params)
    conmat = np.zeros((3, 3), dtype = np.int)
    for t_label in ['0', '1', '2']:
        pred_prob, pred_index = predictions[t_label]
        for p_label in pred_index:
            conmat[int(t_label), int(p_label)] += 1
    print '\t0\t1\t2'
    for t_label in ['0', '1', '2']:
        print t_label + '\t' + '\t'.join([str(conmat[int(t_label), j]) for j in [0, 1, 2]])

if __name__ == '__main__':
    Ks = [1, 3, 5, 10, 40]
    for K in Ks:
        rates = computeROC('results/dtw/dtw_%s.txt' % K)
        plotROC(rates, 'plots/dtw/roc.jpg', 'DTW and k-NN' % K, 0)
        
