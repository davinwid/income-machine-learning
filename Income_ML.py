import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Training, testing and whole data only with continous and class data
test_data = pd.read_csv('data_set/test.csv', sep=r',', \
names =["age", "workclass", "fnlwgt", "education", "education-num", \
"marital-status", "occupation", "relationship", "race", "sex", "capital-gain",\
"capital-loss", "hours-per-week", "native-country"], \
usecols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',\
'hours-per-week'])

train_data = pd.read_csv('data_set/train.csv', sep=r',', \
names =["age", "workclass", "fnlwgt", "education", "education-num", \
"marital-status", "occupation", "relationship", "race", "sex", "capital-gain",\
"capital-loss", "hours-per-week", "native-country", "class"], \
usecols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',\
'hours-per-week', "class"])

whole_data = pd.read_csv('data_set/whole_data.csv', sep=r',', \
names =["age", "workclass", "fnlwgt", "education", "education-num", \
"marital-status", "occupation", "relationship", "race", "sex", "capital-gain",\
"capital-loss", "hours-per-week", "native-country", "class"], \
usecols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',\
'hours-per-week', "class"])

# Constants
# List of regularization constants
lda_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

# Accuracy foe each of the lda values, thus, 1e-4 is
# the one with the largest accuracy based on the validation set,
# the best lambda is then chosen to be 1e-4
# [0.7647861692447679, 0.7752502274795269, 0.7552320291173794,
# 0.7595541401273885, 0.7672884440400364, 0.7529572338489536]
lda = 1e-4

# Initial a and b guess
u = np.array([np.array([0, 0, 0, 0, 0, 0]), 0])

# Number of times the sampling is done
season = 50

# Number of total steps for each sampling
num_steps = 300

# Steps needed before noting the accuracy and magnitude into the graph
step_divide = 30

# Training percentage compared to the whole set, (10% for validation)
train_percentage = 0.9

# Create a panda dataframe of the train data
df_test = pd.DataFrame(test_data)
df_train = pd.DataFrame(train_data)
df_whole = pd.DataFrame(whole_data)

# Initialize the training data by changing <=50K and >50K to 1 and -1 respectively
# Such that y_i calculation is represented correctly
df_train['class'] = df_train['class'].str.strip().replace('<=50K', 1).replace('>50K', -1)

# Iterate through the column, calculate mean var, and normalize the training data
def normalize_data_train(df_train, whole):
    for i in range(len(whole.columns) - 1):
        mean = whole.iloc[:, i].mean()
        var = whole.iloc[:, i].var()
        df_train.iloc[:, i] =  (df_train.iloc[:, i] - mean)/var
    return df_train

# Iterate through the column, calculate mean var, and normalize the test data
def normalize_data_test(df_test, whole):
    for i in range(len(df_test.columns)):
        mean = whole.iloc[:, i].mean()
        var = whole.iloc[:, i].var()
        df_test.iloc[:, i] =  (df_test.iloc[:, i] - mean)/var
    return df_test

# Split the data into validation and train, create a,b pairs
def split_train_validate(df, train_percentage):
    # Filter the class to only -1 or 1 based on the value needed
    train_set_size = int(train_percentage * len(df))
    valid_set_size = len(df) - train_set_size

    # Validate index possible choices
    valid_index = np.random.randint(len(df) - 1 - valid_set_size)
    valid = df[valid_index : valid_index + valid_set_size]
    train = df[0 : valid_index + 1]
    train = train.append(df[valid_index + valid_set_size : len(df) - 1])

    return (valid, train)

# Given the train_data with specified label already filtered,
# Update the model of u (given u numpy array)
# Returns the p for each u_n+1 = u_n + ηp
def update_model_factor(u, n_s, lda, x_i, y_i):
    # Initial a, b guess
    a = u[0]; b = u[1]; diff_a = 0; diff_b = 0

    # Diff a and b
    if (y_i * (a.T@x_i + b) >= 1):
        diff_a = lda * a
        diff_b = 0
    else:
        diff_a = (lda * a) - (y_i * x_i)
        diff_b = -1 * y_i

    g_i = np.array([diff_a, diff_b])
    return (-1 * g_i) - (lda * u)

# Model of the learning rate
def learning_rate(i):
    return 1/(i + 2)

# Returns the magnitude of a vector
def magnitude_vec(a):
    sum = 0
    for a_i in a:
        sum += a_i**2
    return sum**(0.5)

# Return accuracy given the data set and u value
def calc_accuracy(data_set, u):
    correct_count = 0; total_size = len(data_set)
    a_t = u[0].T
    b = u[1]

    # Iterate through the data set
    for i in range(len(data_set)):
        x_i = data_set.iloc[i, :6].values
        y_i = data_set.iloc[i, 6]

        # Positive and negative label correctly predicted using
        # updated a and b values
        if ((a_t @ x_i + b) > 0 and y_i == 1):
            correct_count += 1
        elif ((a_t @ x_i + b) < 0 and y_i == -1):
            correct_count += 1

    return correct_count/total_size

# Trains the classifier given the wanted class, number of season, steps,
# training data, lda value and the initial guess
def train_classifier(u, lda, season, num_steps, step_divide, df):
    # List for 30 step accuracy
    accuracy_steps = []
    magnitude_a = []
    validation_accuracy = []

    valid_train_tuple = split_train_validate(df, train_percentage)
    validate = valid_train_tuple[0]

    for s in range(season):
        # Calculate step size (arbitrary m and n)
        n_s = learning_rate(s)

        # Split the data into validate and training
        train = split_train_validate(df, train_percentage)[1]

        # Select batch_size total data from the training set (held-out set)
        random_data = train.sample(n = 50, replace = True)

        for step in range(num_steps):
            # Extract the xi matrix and yi label
            index = np.random.randint(0, len(train))
            x_i = train.iloc[index, :6].values
            y_i = train.iloc[index, 6]
            # Update the u_n -> u_n+1
            u += (n_s * (update_model_factor(u, n_s, lda, x_i, y_i)))
            # Compute accuracy every step_divide iteration
            if (((step + 1) % step_divide) == 0):
                accuracy_steps.append(calc_accuracy(random_data, u))
                magnitude_a.append(magnitude_vec(u[0]))
        validation_accuracy.append(calc_accuracy(validate, u))
    return (u, accuracy_steps, magnitude_a, validation_accuracy)

# Create a plot image of the graph given the y plots, x axis range
# title, axis labels and legend position.
def plot(y_plots, x, title, x_label, y_label, pos):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for graph in y_plots:
        plt.plot(x, graph)
    plt.legend(('lda: 1e-5', 'lda: 1e-4', 'lda: 1e-3', 'lda: 1e-2',
    'lda: 1e-1', 'lda: 1'), loc = pos)
    plt.show()

# Plot values of the given lda list based of the number of steps and accuracy
def plt_ldas(u, lda_list, season, num_steps, step_divide,  df):
    # List of of corresponding important data
    hold_acc_list = []
    mag_list = []
    val_acc_list = []

    # Add into the dictionaries of different lda values
    for lda in lda_list:
        classifier_tuple = train_classifier(u, lda, season, num_steps,
        step_divide, df)
        hold_acc_list.append(classifier_tuple[1])
        mag_list.append(classifier_tuple[2])
        val_acc_list.append(classifier_tuple[3])

    x_1 = np.linspace(0, season * num_steps/step_divide, 500)
    x_2 = np.linspace(0, season, 50)

    # Plot each corresponding lambda values on its varying magnitude,
    # held out set accuracy and validation set accuracy
    plot(mag_list, x_1, "Regularization constants' coefficient " +
    "magnitude (per 30 steps)", "Steps",'Magnitude of coefficient', 'center right')

    plot(hold_acc_list, x_1, "Regularization constants' accuracy for " +
    "season's hold-out set (per 30 steps)", "Steps", "Accuracy",'lower right')

    plot (val_acc_list, x_2, "Regularization constants' accuracy per " +
    "season's validation set", "Season", "Accuracy", 'lower right')

# Normalize the whole data
df_train = normalize_data_train(df_train, df_whole)
df_test = normalize_data_test(df_test, df_whole)

# Create a classifier that evaluates the prediciton for the test data
def test_result(df, u):
    file = open("submission.txt", "w")
    a_t = u[0].T
    b = u[1]
    for i in range(len(df)):
        x_i = df.iloc[i, :6].values
        if ((a_t @ x_i + b) > 0):
            file.write('<=50K\n')
        elif ((a_t @ x_i + b) < 0):
            file.write('>50K\n')
    file.close()

u_1e_4 = train_classifier(u, lda, season, num_steps, step_divide, df_train)[0]
plt_ldas(u, lda_list, season, num_steps, step_divide, df_train)
test_result(df_test, u_1e_4)
