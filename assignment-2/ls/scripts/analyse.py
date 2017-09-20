import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm

from helpers import multivariateNormalPDF

# First step in the prediction
def predict(data, params):
    # Separate params into mus and covs
    mus, covs = params
    # This dict will store class-wise predictions; one list for each target class 
    predictions = {}
    # Iterate through each target class data-points
    for t_label in ['0', '1', '2']:
        points = data[t_label][:]
        # zs will store the likelihood values for "predicted" class
        zs = []
        for p_label in [0, 1, 2]:
            mean, cov = mus[p_label], covs[p_label]
            zs.append(multivariateNormalPDF(points, mean, cov))
        # Convert it into a numpy-array
        zs = np.array(zs).T
        # Normalize to get probabilities
        probs = (zs.T / zs.sum(axis = 1)).T
        # Get the index of the most probable class 
        preds = np.argmax(probs, axis = 1)
        # Store the prediction probabilities and decisions in this class
        predictions[t_label] = (probs, preds)
   
    return predictions

# ROC computations
def computeROC(data, params):
    # Get the predictions
    predictions = predict(data, params)

    # ROC curve computation
    thresholds_left = [1.0 / 10**i for i in np.arange(16, 3, -1)]
    thresholds_central = list(np.arange(0.001, 1.0, 0.001))
    thresholds_right = [1 - l for l in thresholds_left[::-1]]
    thresholds = [0] + thresholds_left + thresholds_central + thresholds_right + [1.0]
    tprs, fprs = [], []

    # Choose one threshold - one point on the ROC curve
    for threshold in thresholds:
        roc = {}
        # Choose the target class 
        for roc_label in ['0', '1', '2']:
            roc_label_int = int(roc_label)
            tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
            # Run through the data
            for label in ['0', '1', '2']:
                label_int = int(label)
                # Predictions for' label' class
                probs, _ = predictions[label]
                # Run through each point in this class
                for index in range(probs.shape[0]):
                    prob = probs[index]
                    # roc_label_int is positive (+1), the other two classes are negative (-1)
                    if label_int == roc_label_int:
                        if prob[label_int] > threshold:
                            tp += 1
                        else:
                            fn += 1
                    if label_int != roc_label_int:
                        if prob[roc_label_int] > threshold:
                            fp += 1
                        else:
                            tn += 1
            
            roc[roc_label] = (tp / (tp + fn), fp / (fp + tn))
        # Macro average of the ROC values
        tprs.append(np.mean([roc[roc_label][0] for roc_label in ['0', '1', '2']]))
        fprs.append(np.mean([roc[roc_label][1] for roc_label in ['0', '1', '2']]))
        
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

