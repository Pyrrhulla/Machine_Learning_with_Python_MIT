# MITx 6.86x Machine Learning with Python-From Linear Models to Deep Learning

**Certificate received: 09.06.2020** [dc6250ab91f34ad8a3cdcde6e6e86da9](https://courses.edx.org/certificates/dc6250ab91f34ad8a3cdcde6e6e86da9)
__________
**Other certificates**
**6.431x: Probability - The Science of Uncertainty and Data**

**Certificate received: 29.05.2020** [ccb5b62a577a4394acb4731dec2ff2c4](https://courses.edx.org/certificates/ccb5b62a577a4394acb4731dec2ff2c4)

**14.310Fx: Data Analysis in Social Science-Assessing Your Knowledge**

**Certificate received: 14.12.2019** [08166bcfd5554faf983af4e6a82e2e2a](https://courses.edx.org/certificates/08166bcfd5554faf983af4e6a82e2e2a)

**18.6501x: Fundamentals of Statistics**

**Certificate received: 23.09.2020** [c2e8fb91eeaa4fd0b5b0d884a8a9350b](https://courses.edx.org/certificates/c2e8fb91eeaa4fd0b5b0d884a8a9350b)
____________


**Project 1: Automatic Review Analyzer** 

The goal of this project is to design a classifier to use for sentiment analysis of product reviews. Our training set consists of reviews written by Amazon customers for various food products. The reviews, originally given on a 5 point scale, have been adjusted to a +1 or -1 scale, representing a positive or negative review, respectively. 

**Project 2: Digit Recognition** 

Aim: create a numeric recognition algorithm, using a simple neural network.  

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this dataset. 

<img width="572" alt="Снимок экрана 2022-08-23 в 19 32 38" src="https://user-images.githubusercontent.com/55465730/186226216-f3c79675-07ff-4ad3-bfbc-35ad7c1a50ab.png">

**Project 3: Collaborative Filtering via Gaussian Mixtures** 

The task is to build a mixture model for collaborative filtering. A data matrix contains movie ratings made by users where the matrix is extracted from a much larger Netflix database. Any particular user has rated only a small fraction of the movies so the data matrix is only partially filled. The goal is to predict all the remaining entries of the matrix.

I used mixtures of Gaussians to solve this problem. The model assumes that each user's rating profile is a sample from a mixture model. In other words, we have  possible types of users and, in the context of each user, we must sample a user type and then the rating profile from the Gaussian distribution associated with the type. We will use the Expectation Maximization (EM) algorithm to estimate such a mixture from a partially observed rating matrix. The EM algorithm proceeds by iteratively assigning (softly) users to types (E-step) and subsequently re-estimating the Gaussians associated with each type (M-step). Once we have the mixture, we can use it to predict values for all the missing entries in the data matrix.

