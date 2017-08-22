
**Problem 1 : Image Reconstruction**

Perform Eigen Value Decomposition (If image A is rectangular, use A'A) on both square and rectangular images given to your group

- by converting the image to grayscale.

- separately on each color bands.

- after concatenating the 8bit R,G,B channel to form a 24bit number. Experiment with the order of concatenation and analyze. 

Reconstruct the matrix using top N eigen vectors corresponding to top N eigen values. Experiment with the values of N. Also try random N eigen values instead of top N.

**Task Requirements**

- Plot the reconstructed images along with their corresponding error image.

- A comparative graph of the reconstruction error vs N is required in each experiment.

- You can use ONLY inbuilt function of EVD in this whole task. No other inbuilt functions are allowed and you have to code everything yourself.

- A brief comparison of all the techniques and their analysis is required in your report.

- Extra credits for more experiments and observations.

