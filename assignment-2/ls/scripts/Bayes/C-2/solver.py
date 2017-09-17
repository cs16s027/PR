import h5py
import numpy as np

from estimate_parameters import estimateParams
from plot_data import plotData
from plot_pdf import plotPDF
from plot_contours import plotContours
from plot_boundary import plotBoundaries
from helpers import saveModel, loadModel, multivariateNormalPDF, predict

# Train model 
def train(data, data_plot, pdf_plot, contour_plot, boundary_plot):
    print 'Training model'
    # Plot the data
    plotData(data, data_plot, title = 'Training data', figure = 0)
    # Estimate the means and the covariance matrix for the data
    mus, covs = estimateParams(data)
    # Plot the PDF, contours and directions for each class
    plotPDF(mus, covs, pdf_plot, title = 'Class-conditional densities', figure = 1)
    # Plot the contours alone in a separate plot
    plotContours(mus, covs, contour_plot, title = 'Constant-density curves with directions', figure = 2)
    # Plot the decision boundaries
    plotBoundaries(mus, covs, boundary_plot, title = 'Decision surfaces', figure = 3)
    # Return the trained values
    return np.array(mus), np.array(covs)

def validate(data, params):
    # Get the confusion matrix
    predict(data, params, 0.5)
    pass  
       

if __name__ == '__main__':
    '''
    data = h5py.File('data/train.h5', 'r')
    # Training
    mus, covs = train(data, 'plots/Bayes/C-2/train/data.jpg',\
                            'plots/Bayes/C-2/train/pdf.jpg',\
                            'plots/Bayes/C-2/train/contours.jpg',\
                            'plots/Bayes/C-2/train/boundaries.jpg') 
    # Save model
    saveModel([mus, covs], 'models/Bayes/C-2/model.h5')
    '''
    # Load model
    mus, covs = loadModel('models/Bayes/C-2/model.h5')
    # Validation
    data = h5py.File('data/valid.h5', 'r')
    validate(data, [mus, covs])
    # Testing

