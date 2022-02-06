# Python tool for ANOVA analysis and Linear regression

This repository provides a Python script for ANOVA and linear regression anaylses.

### Content

* The script, [functions.py](functions.py), is designed to provide necessary functions for one-way ANOVA, two-way ANOVA, simple and multiple linear regression. 
* [main.py](functions.py) file is a test script to demonstrate how to use functions with what kind of arguments. It could be used with *-w* parameter to show output of tests one by one, waiting an input from user.

### Example
* **ANOVA1_test_equality** function tests the equality of the means among the groups:
	* Input:
	>	
		Groups: [[6.9, 5.4, 5.8, 4.6, 4.0], 
				[8.3, 6.8, 7.8, 9.2, 6.5], 
				[8.0, 10.5, 8.1, 6.9, 9.3], 
				[5.8, 3.8, 6.1, 5.6, 6.2]]
		alpha: 0.05

	* ANOVA table:
	>
		| Source         |   df  |   SS  |   MS  |   F  |
		|----------------|:-----:|:-----:|:-----:|:----:|
		| Between groups |  3.00 | 38.82 | 12.94 | 9.72 |
		| Within groups  | 16.00 | 21.29 |  1.33 |      |
		| Total          | 19.00 |       |       |      |

	* P-value = 0.000684 < alpha = 0.05. Thus, **the null hypothesis** that the means of the groups are equal is **rejected at significance level alpha = 0.05.**

* **ANOVA1_CI_linear_combs** function calculates simultaneous confidence intervals of linear combinations given as a matrix, and returns a list of confidence intervals holding simultaneously with 1-alpha probability:
	*  The function offers different methods for calculating the interval:
		*  Scheffe's method
		* Tukey's pairwise comparison
		* Bonferroni correction
		* Sidak's method
		* And the best choice as the function finds the most suitable method for the given linear combinations.
	* Input:
	>
		Groups: [[6.9, 5.4, 5.8, 4.6, 4.0], 
				[8.3, 6.8, 7.8, 9.2, 6.5], 
				[8.0, 10.5, 8.1, 6.9, 9.3], 
				[5.8, 3.8, 6.1, 5.6, 6.2]]
		alpha: 0.05
		Linear combinations C: [[0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5]]
	* Output:
	>	
		Bonferroni:
			(-1.7758, 0.7758)
			(-0.9358, 1.6158)
		Scheffe 2.8:
			(-2.1081, 1.1081)
			(-1.2681, 1.9481)
		Sidak:
			(-1.7725, 0.7725)
			(-0.9325, 1.6125)
		Tukey:
		Error! Given C matrix is not valid for Tukey (expecting pairwise comparisons)
		
* **Mult_LR_Least_squares** function calculate the least square estimates for the beta vector and the biased and unbiased sigma estimators:
	* Input:
	>	
		|  x0 | Height | Weight | Distance |
		|:---:|:------:|:------:|:--------:|
		| 1.0 |  42.8  |  40.0  |   37.0   |
		| 1.0 |  63.5  |  93.5  |   49.5   |
		| 1.0 |  37.5  |  35.5  |   34.5   |
		| 1.0 |  39.5  |  30.0  |   36.0   |
		| 1.0 |  45.5  |  52.0  |   43.0   |
		| 1.0 |  38.5  |  17.0  |   28.0   |
		| 1.0 |  43.0  |  38.5  |   37.0   |
		| 1.0 |  22.5  |   8.5  |   20.0   |
		| 1.0 |  37.0  |  33.0  |   33.5   |
		| 1.0 |  23.5  |   9.5  |   30.5   |
		| 1.0 |  33.0  |  21.0  |   38.5   |
		| 1.0 |  58.0  |  79.0  |   47.0   |
	* Output:
	> 
		BetaHat: [21.00839777  0.19635663  0.19082778] 
		Biased Sigma Square Hat: 11.6594
		Unbiased Sigma Square Hat: 15.5459
