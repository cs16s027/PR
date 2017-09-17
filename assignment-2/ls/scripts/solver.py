import h5py
import numpy as np

from estimate_parameters import estimateParams
from plot_data import plotData
from plot_pdf import plotPDF
from plot_contours import plotContours

def solve(data, data_plot, pdf_plot, contour_plot):
    # Plot the data
    plotData(data, data_plot, figure = 0)
    # Estimate the means and the covariance matrix for the data
    mus, covs = estimateParams(data)
    # Plot the PDF, contours and directions for each class
    plotPDF(mus, covs, pdf_plot, figure = 1)
    # Plot the contours alone in a separate plot
    plotContours(mus, covs, contour_plot, figure = 2)

if __name__ == '__main__':
    data = h5py.File('data/train.h5', 'r')
    solve(data, 'plots/data.jpg', 'plots/pdf.jpg', 'plots/contours.jpg')

