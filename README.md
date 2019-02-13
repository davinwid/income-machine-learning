# SVM Gradient Descent Machine Learning Project
Create a training algorithm for a Support Vector Machine (SVM), that uses Gradient Descent model to predict future's data label. Since an SVM can only define two labels, it can only categorize the vector into 2 values, in this case of Adult income's data, <=50K as 1 and >50K as -1.

## Procedures
* ### Step 1 (Extract Training Data)
    Initialize the dataframe array by reading from a comma separated file (.csv) and giving labels to the data read. The data read in this case, since we are only considering continous data are, namely, age, final-weight, education-num, capital-gain, capital-loss, hours-per-week. And to that dataframe, I added a class column, that is the prediction of label of each of the given data.

    Note: Refer to source's data-set description for more information about the labels' description [here](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)

* ### Step 2 (Specify the Constants)
    Initialized constant values that are,
    * lda_list (values of regularization constants to consider)
    * ū = (ā, b) (original guess of the gradient descent classifier)
    * season (number of seasons the algorithm is set to run)
    * num_steps (number of steps ū value is changed every season)
    * step_divide (number of steps done before taking accuracy and magnitude values)
    * train_percentage (percentage used for the train data set (90%) vs. validation set (10%))

* ### Step 3 (Scale the data used)
    Take the mean and variance of the whole data set (before training/testing the data), then subtract from every data in the column, the mean. Divide that by the variance so that each value has a unit variance and zero mean.

* ### Step 4 (Split the Training Data Set)
    Split the training data set into the percentage given, and thus creating the validation set, and training set. Validation set is divided for the purpose of data to make sure that we have an unbiased data that is not used to train the classifier, to check its accuracy to. Go back to the classifier training algorithm if accuracy is not high enough for this data set.

* ### Step 5 (Start training the Data Set)
    Go through the specified the number of num_steps, number of season times. Create two arrays, one for held-out set accuracy and magnitude of a.
    Each season, create a new training set data to train the ū with (a new 90% of the whole training-set) and also separate out a 50 data from that training data set as a held-out set. For each steps taken during a single season, we take a random index from the "seasonal" training set and use that to update the value of ū. Repeat the steps step_divide times before
    1. Calculate the accuracy of the current classifier with the held-out set
    2. Calculate the magnitude of ā in ū

    After the number of steps taken is finished, we calculate the accuracy of the validation set before moving on to the next season and repeating this step (number of) season times.

* ### Step 5a (Calculating the accuracy)
    For each data-column in the data-set (*held-out* or *validation set*), create an **x** using the continous data, and with the known pair values of (ā, b), being ū, use the boolean to come up with a prediction for the data:
    * ā**x** + b > 0 having predicted value 1 (<=50K)
    * ā**x** + b < 0 having predicted value -1 (>50K)

    Calculating the percentage of each of the data-set accuracy (correct counter + 1 if correct guess) over the number of total data in the data-set.

* ### Step 5b (Calculate magnitude of ā)
    For each value in vector ā, take the square values and root the result to get the magnitude of ā.

* ### Step 5c (Updating ū)
    A few values and formulas to update ū

    * Learning rate model, η = 1/(s + 2) where s is the current season number.
    * The cost function to be minimized, p = -(∇gi(ū_(n))) - λū_(n)

      ∇gi(ū_(n)) is defined by the vector ū with (ā, b) as follows:
      ![picture alt](https://github.com/davinwid/income-machine-learning/blob/master/result_image/Cost%20Function%20Gradient.png)


    * The updated ū, ū_(n+1) = ηp + ū_(n), where ū_(n) is the current value of ū

    Note: λ is the regularization constant we chose from the list (iterate through one-by-one)

* ### Step 6 (Choosing the Best Regularization Constant, λ)
    Repeat step 5 for the possible λ values in lda_list, and plot the results of
     * Accuracy of the *held-out* set noted every step_divide for each corresponding ū
     * Magnitude of ā calculated every step_divide for each corresponding ū
     * Accuracy of the *validation set* for the final ū value for each corresponding λ

     in a side-by-side plot for all the λ values to compare their values with each other.
     The regularization constant with the highest **average** accuracy for the *held-out* and *validation set* will be the best regularization constant to set for the whole Gradient Descent algorithm. Now, using the final ū, iterate through the  given the continous test data to make predictions necessary. Happy Working!

## Graphs Created
  ### Accuracy of held-out set
  ![picture alt](https://github.com/davinwid/income-machine-learning/blob/master/result_image/Held-out%20Set%20Accuracy.png)
  ### Accuracy of validation set every season
  ![picture alt](https://github.com/davinwid/income-machine-learning/blob/master/result_image/Validation%20Set%20Accuracy.png)
  ### Magnitude of ā
  ![picture alt](https://github.com/davinwid/income-machine-learning/blob/master/result_image/Magnitude.png)

## Imports used
* [Pandas](https://pandas.pydata.org/) - Used to manipulate and hold data
* [Numpy](http://www.numpy.org/) - Used to hold values in vector, matrix format

## Data Provided by UCI Machine Learning Repository
```
[https://archive.ics.uci.edu/ml/datasets/Adult](https://archive.ics.uci.edu/ml/datasets/Adult)
```
   Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science
