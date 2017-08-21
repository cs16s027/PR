Problem 1: Image Reconstruction 

Tasks to be performed:
1) Perform Singular Value Decomposition on both square and rectangular images (given below)

(a) by converting the image to grayscale.

(b) separately on each color bands.

(c) after concatenating the 8bit R,G,B channel to form a 24bit number. Experiment with the order of concatenation and analyze. 

Reconstruct the matrix using top N singular vectors corresponding to top N singular values. Experiment with the values of N. Also try random N singular values instead of top N.

2) Perform Eigen Value Decomposition (If image A is rectangular, use A'A) on both square and rectangular images given to your group

(a) by converting the image to grayscale.

(b) separately on each color bands.

(c) after concatenating the 8bit R,G,B channel to form a 24bit number. Experiment with the order of concatenation and analyze. 

Reconstruct the matrix using top N eigen vectors corresponding to top N eigen values. Experiment with the values of N. Also try random N eigen values instead of top N.


TASK REQUIREMENTS:

(a) Plot the reconstructed images along with their corresponding error image.

(b) A comparative graph of the reconstruction error vs N is required in each experiment.

(c) You can use ONLY inbuilt function of EVD in this whole task. No other inbuilt functions are allowed and you have to code everything yourself.

(d) A brief comparison of all the techniques and their analysis is required in your report.

(e) Extra credits for more experiments and observations.

