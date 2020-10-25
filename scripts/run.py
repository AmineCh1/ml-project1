import numpy as np
from implementations import *
import argparse

def main(recomp_params=False):
    # Loading the training data
    DATA_TRAIN_PATH = '../data/train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

    # Loading the test data
    DATA_TEST_PATH = '../data/test.csv'
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    # Defining outputh path
    OUTPUT_PATH = '../data/submission-ridge.csv' 

    # Partitioning the training data
    index0, y0, x_train0 = partition_data(tX, y, 0)
    index1, y1, x_train1 = partition_data(tX, y, 1)
    index2, y2, x_train2 = partition_data(tX, y, 2)
    index3, y3, x_train3 = partition_data(tX, y, 3)

    # Partitioning the test data
    index0_t, y0_t, x_test0 = partition_data(tX_test, y_test, 0)
    index1_t, y1_t, x_test1 = partition_data(tX_test, y_test, 1)
    index2_t, y2_t, x_test2 = partition_data(tX_test, y_test, 2)
    index3_t, y3_t, x_test3 = partition_data(tX_test, y_test, 3)

    # Standardizing the training data
    std_train0, mean_x0, std_x0 = standardize_data(x_train0)
    std_train1, mean_x1, std_x1 = standardize_data(x_train1)
    std_train2, mean_x2, std_x2 = standardize_data(x_train2)
    std_train3, mean_x3, std_x3 = standardize_data(x_train3)

    # Adding a bias term to the training data
    std_train0 = np.hstack((np.ones((x_train0.shape[0], 1)), std_train0))
    std_train1 = np.hstack((np.ones((x_train1.shape[0], 1)), std_train1))
    std_train2 = np.hstack((np.ones((x_train2.shape[0], 1)), std_train2))
    std_train3 = np.hstack((np.ones((x_train3.shape[0], 1)), std_train3))

    # Adding a bias term and normalizing the test data with respect to
    # offset and scale found by normalizing the training data.
    std_test0 = np.hstack(
        (np.ones((x_test0.shape[0], 1)), (x_test0-mean_x0)/std_x0))
    std_test1 = np.hstack(
        (np.ones((x_test1.shape[0], 1)), (x_test1-mean_x1)/std_x1))
    std_test2 = np.hstack(
        (np.ones((x_test2.shape[0], 1)), (x_test2-mean_x2)/std_x2))
    std_test3 = np.hstack(
        (np.ones((x_test3.shape[0], 1)), (x_test3-mean_x3)/std_x3))

    if  recomp_params:
        # Finding the best parameters to run model with for each batch ( with 4-fold cross-validation)
        degree_lambda_0 = find_best_params(y0, std_train0, 4)
        degree_lambda_1 = find_best_params(y1, std_train1, 4)
        degree_lambda_2 = find_best_params(y2, std_train2, 4)
        degree_lambda_3 = find_best_params(y3, std_train3, 4)
    else:
        degree_lambda_0 = (11,0.0012915)
        degree_lambda_1 = (12,0.0021544)
        degree_lambda_2 = (13,0.0002782)
        degree_lambda_3 = (13,0.0007742)
        
    # Running ridge regression with found parameters
    w_0 = run_model(y0, build_poly(
        std_train0, degree_lambda_0[0]), model='ridge_reg', lambda_ = degree_lambda_0[1])
    w_1 = run_model(y1, build_poly(
        std_train1, degree_lambda_1[0]), model='ridge_reg', lambda_ = degree_lambda_1[1])
    w_2 = run_model(y2, build_poly(
        std_train2, degree_lambda_2[0]), model='ridge_reg', lambda_ = degree_lambda_2[1])
    w_3 = run_model(y3, build_poly(
        std_train3, degree_lambda_3[0]), model='ridge_reg', lambda_ = degree_lambda_3[1])

    #Predicting labels
    label_0 = predict_labels(w_0, build_poly(std_test0, degree_lambda_0[0]))
    label_1 = predict_labels(w_1, build_poly(std_test1, degree_lambda_1[0]))
    label_2 = predict_labels(w_2, build_poly(std_test2, degree_lambda_2[0]))
    label_3 = predict_labels(w_3, build_poly(std_test3, degree_lambda_3[0]))

    #Concatenating prediction ofr final prediction
    labels = np.empty(len(y_test))
    labels[index0_t] = label_0
    labels[index1_t] = label_1
    labels[index2_t] = label_2
    labels[index3_t] = label_3

    #Creating submission
    create_csv_submission(ids_test, labels, OUTPUT_PATH)

parser = argparse.ArgumentParser(prog='run',
                                    description='Best submission replication')
parser.add_argument("--recompute_params", help ="Use hardcoded optimal values to save time",action="store_true")
args = parser.parse_args()

main(args.recompute_params)
