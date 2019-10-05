# jtangTJC

-stochastic algorithm in Matlab.docx
  an exmpale code of modeling of 2D and 3D stochastic algorithm.

—R_kaiC_2parameters.R
	R file used to perform gradient matching with GP to infer rate constants
—KaiCv4.py
	This file performs the following:
	a. Solve O3 ODEs and generate the experimental data and the ground-truth curve
	b. Perform linear regression to infer parameters
	c. Read in inferred data from GP and RNN and compares the performance of linear regression, gp, and rnn. 
	d. Calculate the MSE of 3 models
-KaiC_RNN.py
	a. Solve KaiC ODEs and generate the experimental data and the ground-truth curve
	b. Perform RNN training to estimate rate constants
