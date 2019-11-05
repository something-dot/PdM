import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os.path
from os import path

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve

from sklearn.model_selection import train_test_split
from preprocessing import flatten
from sklearn.externals.joblib import dump, load
import h5py


def test_train_split_standardize(df, seed=456, data_split=0.2):
    df_train, df_test = train_test_split(df, test_size=data_split, random_state=seed)
    df_train, df_valid = train_test_split(df_train, test_size=data_split, random_state=seed)
    df_train_0 = df_train.loc[df['y'] == 0]

    df_train_0_x = df_train_0.drop(['y'], axis=1)
    df_valid_0 = df_valid.loc[df['y'] == 0]
    df_valid_0_x = df_valid_0.drop(['y'], axis=1)

    scaler = StandardScaler().fit(df_train_0_x)
    dump(scaler, "../output/std_scaler.bin", compress=True)

    df_train_0_x_rescaled = scaler.transform(df_train_0_x)
    df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)
    df_valid_x_rescaled = scaler.transform(df_valid.drop(['y'], axis=1))

    df_test_x_rescaled = scaler.transform(df_test.drop(['y'], axis=1))
    return df_train_0_x_rescaled, df_valid_0_x_rescaled, df_valid_x_rescaled, df_test_x_rescaled, df_valid, df_test


def auto_encoder_model(input_dim, path, train_test_data, nb_epoch=200, batch_size=128, encoding_dim=32,
                       learning_rate=0.01,
                       optimizer="Adam", activation="relu"):
    input_layer = Input(shape=(input_dim,))
    hidden_dim = int(encoding_dim / 2)
    encoder = Dense(encoding_dim, activation=activation, activity_regularizer=regularizers.l1(learning_rate))(
        input_layer)
    encoder = Dense(hidden_dim, activation=activation)(encoder)
    decoder = Dense(hidden_dim, activation=activation)(encoder)
    decoder = Dense(encoding_dim, activation=activation)(decoder)
    decoder = Dense(input_dim, activation="linear")(decoder)
    auto_encoder = Model(inputs=input_layer, outputs=decoder)
    auto_encoder.compile(metrics=['accuracy'],
                         loss='mean_squared_error',
                         optimizer=optimizer)

    cp = ModelCheckpoint(filepath=path,
                         save_best_only=True,
                         verbose=0)

    history = auto_encoder.fit(train_test_data[0], train_test_data[0],
                               epochs=nb_epoch,
                               batch_size=batch_size,
                               shuffle=True,
                               validation_data=(
                                   train_test_data[1], train_test_data[1]),
                               verbose=1,
                               callbacks=[cp]).history
    auto_encoder = load_model(path)
    return auto_encoder


def calculate_auc(rescaled, original, path, LSTM_ind=False):
    model = load_model(path)
    valid_x_predictions = model.predict(rescaled)
    if LSTM_ind:
        mse = np.mean(np.power(flatten(rescaled) - flatten(valid_x_predictions), 2), axis=1)
    else:
        mse = np.mean(np.power(rescaled - valid_x_predictions, 2), axis=1)
    if LSTM_ind:
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': list(original)})
    else:
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': original['y']})
    false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
    roc_auc = auc(false_pos_rate, true_pos_rate)
    return roc_auc


def chose_weights_test_results(negative_weight, positive_weight, path, rescaled, original, LSTM_ind=False):
    auto_encoder = load_model(path)
    # Predictions on validation set
    valid_x_predictions = auto_encoder.predict(rescaled)
    if LSTM_ind:
        mse = np.mean(np.power(flatten(rescaled) - flatten(valid_x_predictions), 2), axis=1)
    else:
        mse = np.mean(np.power(rescaled - valid_x_predictions, 2), axis=1)

    if LSTM_ind:
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': list(original)})
    else:
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': original['y']})
    prob = []
    cost_list = []
    fp_values = []
    fn_values = []
    # Chose the threshold based on the validation set
    for i in [x / 100.0 for x in range(0, 40, 1)]:
        pred_y = [1 if e > i else 0 for e in error_df.Reconstruction_error.values]
        true_y = list(map(int, error_df.True_class.values))
        c1 = [x and y for x, y in zip([x == 1 for x in pred_y], [x == 0 for x in true_y])]
        c2 = [x and y for x, y in zip([x == 0 for x in pred_y], [x == 1 for x in true_y])]
        fp_values.append(sum(c1))
        fn_values.append(sum(c2))
        # Calculate cost based on weights.
        cost = np.sum(np.array(c1) * negative_weight + np.array(c2) * positive_weight)
        prob.append(i)
        cost_list.append(cost)
    return prob[cost_list.index(min(cost_list))]


def roc_curve_plot(rescaled, original, path, LSTM_ind=False, ):
    auto_encoder = load_model(path)
    valid_x_predictions = auto_encoder.predict(rescaled)
    if LSTM_ind:
        mse = np.mean(np.power(flatten(rescaled) - flatten(valid_x_predictions), 2), axis=1)
    else:
        mse = np.mean(np.power(rescaled - valid_x_predictions, 2), axis=1)
    if LSTM_ind:
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': list(original)})
    else:
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': original['y']})
    false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
    roc_auc = auc(false_pos_rate, true_pos_rate)
    fig = plt.figure(figsize=(8, 8))
    plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], linewidth=5)
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='lower right')
    plt.title('Receiver operating characteristic curve (ROC) on Validation set')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig.savefig('../output/roc_curve.png')
    plt.show()


def test_metrics_print(path, rescaled, original, LSTM_ind=False, threshold_fixed=0.5):
    auto_encoder = load_model(path)
    # threshold_fixed = chose_weights_test_results(negative_weight,positive_weight,path,rescaled,original,LSTM_ind)
    valid_x_predictions = auto_encoder.predict(rescaled)
    if LSTM_ind:
        mse = np.mean(np.power(flatten(rescaled) - flatten(valid_x_predictions), 2), axis=1)
    else:
        mse = np.mean(np.power(rescaled - valid_x_predictions, 2), axis=1)
    if LSTM_ind:
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': list(original)})
    else:
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': original['y']})
    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
    predictions = pd.DataFrame({'true': error_df.True_class,
                                'predicted': pred_y})
    conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    fig = plt.figure(figsize=(8, 8))
    LABELS = ["Normal", "Break"]
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    fig.savefig('../output/confusion_matrix.png')
    plt.show()
    tn, fp, fn, tp = confusion_matrix(error_df.True_class, pred_y).ravel()
    print('sensitivity', tp / (tp + fn) * 100)
    print('specificity', tn / (tn + fp) * 100)
    print('precision', tp / (tp + fp) * 100)
    print('accuracy', (tp + tn) / (tp + tn + fp + fn) * 100)


def hyper_param_tune(df, activation_list, optimizer_list, learn_rate_list, path):
    dic = {}
    train_test_valid_standardised_data = test_train_split_standardize(df)
    input_dim = train_test_valid_standardised_data[0].shape[1]
    for k in learn_rate_list:
        for j in optimizer_list:
            for i in activation_list:
                autoencoder = auto_encoder_model(input_dim=input_dim, train_test_data=train_test_valid_standardised_data, nb_epoch=200, batch_size=128, encoding_dim=32,
                                                 learning_rate=k, optimizer=j, activation=i,
                                                 path=path)
                dic[(i, j, k)] = calculate_auc(train_test_valid_standardised_data[2],
                                               train_test_valid_standardised_data[4],
                                               path=path)
    results = max(dic, key=dic.get)
    auto_encoder = auto_encoder_model(input_dim=train_test_valid_standardised_data[0].shape[1], train_test_data=train_test_valid_standardised_data,
                                      nb_epoch=1000, batch_size=32, encoding_dim=200, learning_rate=results[2],
                                      activation=results[0], optimizer=results[1], path=path)
    return auto_encoder


def get_predictions(path, rescaled, original, LSTM_ind=False, threshold_fixed=0.5):
    auto_encoder = load_model(path)
    # threshold_fixed = chose_weights_test_results(negative_weight,positive_weight,path,rescaled,original,LSTM_ind)
    valid_x_predictions = auto_encoder.predict(rescaled)
    if LSTM_ind:
        mse = np.mean(np.power(flatten(rescaled) - flatten(valid_x_predictions), 2), axis=1)
    else:
        mse = np.mean(np.power(rescaled - valid_x_predictions, 2), axis=1)
    if LSTM_ind:
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': list(original)})
    else:
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': original['y']})

    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
    return pred_y


if __name__ == '__main__':
    # Read in data
    try:
        df = pd.read_csv(pd.read_csv(sys.argv[1]))  # "../../output/imputed_df.csv"
    except FileNotFoundError:
        df = pd.read_csv("../output/imputed_df.csv")

    # If no mode, just make a model
    if path.exists('../../output/autoencoder.h5') == False:
        activation_list = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        optimizer_list = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        learn_rate_list = [0.001, 0.01]
        # Best model for auto encoder
        path = '../output/autoencoder.h5'
        best_model = hyper_param_tune(df, activation_list, optimizer_list, learn_rate_list, path=path)
        best_model.save('../output/autoencoder.h5')

        train_test_valid_standardized_data = test_train_split_standardize(df)

        test_metrics_print(path, rescaled, original, LSTM_ind=False, threshold_fixed=0.5)
        test_metrics_print(path, train_test_valid_standardized_data[3], train_test_valid_standardized_data[3])
    else:
        path = '../output/autoencoder.h5'
        scaler = load("../output/std_scaler.bin")
        df_rescaled = scaler.transform(df.drop(['y'], axis=1))
        pred_y = get_predictions(path=path, rescaled=df_rescaled, original=df)
        pd.DataFrame(pred_y).to_csv("../../output/pred.csv")

