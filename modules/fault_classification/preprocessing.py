import pandas as pd
import numpy as np


def curve_shift(df, shift_by, target='y'):
    """
    This function will shift the binary labels in a dataframe.
    The curve shift will be with respect to the 1s.
    For example, if shift is -2, the following process
    will happen: if row n is labeled as 1, then
    - Make row (n+shift_by):(n+shift_by-1) = 1.
    - Remove row n.
    i.e. the labels will be shifted up to 2 rows up.

    :param df: DataFrame
    :param shift_by: An integer denoting the number of rows to shift.
    :param target: The target feature
    :return: curve shifted DataFrame
    """
    sign = lambda x: (1, -1)[x < 0]

    vector = df[target].copy()
    for _ in range(abs(shift_by)):
        tmp = vector.shift(sign(shift_by))
        tmp = tmp.fillna(0)
        vector += tmp

    labelcol = target
    # Add vector to the df
    df.insert(loc=0, column=labelcol + 'tmp', value=vector)
    # Remove the rows with labelcol == 1.
    df = df.drop(df[df[labelcol] == 1].index)
    # Drop labelcol and rename the tmp col as labelcol
    df = df.drop(labelcol, axis=1)
    df = df.rename(columns={labelcol + 'tmp': labelcol})
    # Make the labelcol binary
    df.loc[df[labelcol] > 0, labelcol] = 1
    return df


def flatten(x):
    """
    Flatten a 3D array.

    :param X: A 3D array for lstm, where the array is sample x timesteps x features.
    :return: A 2D array, sample x features.
    """
    flattened_x = np.empty((x.shape[0], x.shape[2]))  # sample x features array.
    for i in range(x.shape[0]):
        flattened_x[i] = x[i, (x.shape[1] - 1), :]
    return flattened_x

