import h5py
import numpy as np

from estimate_parameters import estimateParams3
from plot_data import plotData
from plot_pdf import plotPDF
from plot_contours import plotContours
from plot_boundary import plotBoundaries
from helpers import saveModel, loadModel, multivariateNormalPDF
from analyse import computeROC, plotROC, printConfusionMatrix

# Train model 
def train(data, data_plot, pdf_plot, contour_plot, boundary_plot, choice):
    print 'Training model'
    # Plot the data
    plotData(data, data_plot, title = 'Training data', figure = 0)
    # Estimate the means and the covariance matrix for the data
    mus, covs = estimateParams3(data, choice)
    # Plot the PDF, contours and directions for each class
    plotPDF(mus, covs, pdf_plot, title = 'Class-conditional densities', figure = 1)
    # Plot the contours alone in a separate plot
    plotContours(mus, covs, contour_plot, title = 'Constant-density curves with directions', figure = 2)
    # Plot the decision boundaries
    plotBoundaries(mus, covs, boundary_plot, title = 'Decision surfaces', figure = 2)
    # Return the trained values
    return np.array(mus), np.array(covs)

def validate(data, params, roc_plot):
    print 'Validating model'
    # Get the ROC data-points
    rates = computeROC(data, params)
    # Plot the ROC curve
    plotROC(rates, roc_plot, 'ROC curve', figure = 4)
    # Print confusion matrix
    printConfusionMatrix(data, params)
    return rates

def test(data, params, plot):
    print 'Testing model'
    # Plot the data
    # Use the trained params
    mus, covs = params
    # Plot the contours alone in a separate plot
    plotContours(mus, covs, plot, title = 'Constant-density curves with directions', figure = 5)
    # Plot the decision boundaries
    plotBoundaries(mus, covs, plot, title = 'Decision surfaces', figure = 5)
    # Plot the data
    plotData(data, plot, title = 'Test data', figure = 5)

if __name__ == '__main__':
    data = h5py.File('data/train.h5', 'r')
    # Training
    mus, covs = train(data, 'plots/C-3/train/5/data.jpg',\
                            'plots/C-3/train/5/pdf.jpg',\
                            'plots/C-3/train/5/contours.jpg',\
                            'plots/C-3/train/5/boundaries.jpg',\
                             (2, 1))   
    # Save model
    saveModel([mus, covs], 'models/C-3/5/model.h5')
    # Load model
    mus, covs = loadModel('models/C-3/5/model.h5')
    # Validation
    data = h5py.File('data/valid.h5', 'r')
    rates = validate(data, [mus, covs], 'plots/C-3/valid/5/roc.jpg')

