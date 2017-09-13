import h5py
from matplotlib import pyplot as plt
from estimate_parameters import estimateParams
from plot_pdf import plotPDF

def solve(data):
    # Estimate the means and the covariance matrix for the data
    means, covs = estimateParams(data)
    # Plot the PDF, contours and directions for each class
    plt.figure(1)
    for label in [0, 1, 2]:
        mu, cov = means[label], covs[label]
        plotPDF(mu, cov)
    plt.savefig('plots/pdf.png')

if __name__ == '__main__':
    data = h5py.File('data/train.h5', 'r')
    solve(data)

