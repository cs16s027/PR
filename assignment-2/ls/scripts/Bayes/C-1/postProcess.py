import numpy as np
from matplotlib import pyplot as plt

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
        zs = np.array(zs)
        # Get the index of the most probable class 
        pred_index = np.argmax(zs, axis = 0)
        # Get the probability of the most probable class
        pred_prob = np.max(zs, axis = 0)
        # Store the prediction probabilities and decisions in this class
        predictions[t_label] = (pred_prob, pred_index)
   
    return predictions

# ROC computations
def computeROC(data, params):
    # Get the predictions
    predictions = predict(data, params)
    # Get the maximum prediction to set threshold
    max_prediction = max(np.max(predictions[label][0]) for label in ['0', '1', '2'])
    #print 'Max prediction =', max_prediction
    # Get the number of points in each class
    sizes = [data[label][:].shape[0] for label in ['0', '1', '2']]

    # ROC curve computation
    thresholds = np.arange(0, max_prediction, max_prediction / 10.0)
    tprs, fprs = [], []
    for threshold in thresholds:
        # Choose one threshold - one point on the ROC curve
        # Choose the target class 
        roc = {}
        for roc_label in ['0', '1', '2']:
            tpr, fpr = 0.0, 0.0
            # Calculate the TPR
            tpr_prob, tpr_index = predictions[roc_label]
            for index in range(tpr_prob.shape[0]):
                if tpr_index[index] == int(roc_label) and tpr_prob[index] >= threshold:
                    tpr += 1
            # Calculate the FPR
            fpr_labels = ['0', '1', '2']
            fpr_labels.remove(roc_label)
            for fpr_label in fpr_labels:
                fpr_prob, fpr_index = predictions[fpr_label]
                for index in range(fpr_prob.shape[0]):
                    if fpr_index[index] == int(roc_label) and fpr_prob[index] >= threshold:
                        fpr += 1
            
            roc[roc_label] = (tpr / sizes[int(roc_label)],\
                              fpr / sum([sizes[int(l)] for l in fpr_labels]))

        tprs.append(np.mean([roc[roc_label][0] for roc_label in ['0', '1', '2']]))
        fprs.append(np.mean([roc[roc_label][1] for roc_label in ['0', '1', '2']]))
        
    return (fprs, tprs)

def plotROC(rates, plot, title, figure):
    fig = plt.figure(figure)
    ax = fig.gca()
    ax.plot(rates[0], rates[1])
    ax.set_title(title)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_ylim([0, 1.0])
    ax.set_xlim([0, 1.0])
    plt.savefig(plot)

def printConfusionMatrix(data, params):
    predictions = predict(data, params)
    conmat = np.zeros((3, 3), dtype = np.int)
    for t_label in ['0', '1', '2']:
        pred_prob, pred_index = predictions[t_label]
        for p_label in pred_index:
            conmat[t_label, p_label] += 1
    print '\t0\t1\t2'
    for t_label in ['0', '1', '2']:
        print t_label + '\t' + '\t'.join([str(conmat[t_label, j]) for j in [0, 1, 2]])

