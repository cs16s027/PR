import h5py
import numpy as np

from estimate_parameters import estimateParams
from plot_pdf import plotPDF

def solve(data, plot):
    # Estimate the means and the covariance matrix for the data
    mus, covs = estimateParams(data)
    # Plot the PDF, contours and directions for each class
    plotPDF(mus, covs, plot)

if __name__ == '__main__':
    data = h5py.File('data/train.h5', 'r')
    solve(data, 'plots/pdf.jpg')

