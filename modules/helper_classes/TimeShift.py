class curveShift:
    '''
    This function will shift the binary labels in a dataframe.
    The curve shift will be with respect to the 1s.
    For example, if shift is -2, the following process
    will happen: if row n is labeled as 1, then
    - Make row (n+shift_by):(n+shift_by-1) = 1.
    - Remove row n.
    i.e. the labels will be shifted up to 2 rows up.

    Inputs:
    df       A pandas dataframe with a binary labeled column.
             This labeled column should be named as 'y'.
    shift_by An integer denoting the number of rows to shift.

    Output
    df       A dataframe with the binary labels shifted by shift.
    '''

    def __init__(self, df, shift_by):
        self.df = df
        self.shift_by = shift_by

    def shift(self, target):
        sign = lambda x: (1, -1)[x < 0]

        vector = self.df[target].copy()
        for s in range(abs(self.shift_by)):
            tmp = vector.shift(sign(self.shift_by))
            tmp = tmp.fillna(0)
            vector += tmp
        labelcol = target
        # Add vector to the df
        self.df.insert(loc=0, column=labelcol + 'tmp', value=vector)
        # Remove the rows with labelcol == 1.
        self.df = self.df.drop(self.df[self.df[labelcol] == 1].index)
        # Drop labelcol and rename the tmp col as labelcol
        self.df = self.df.drop(labelcol, axis=1)
        self.df = self.df.rename(columns={labelcol + 'tmp': labelcol})
        # Make the labelcol binary
        self.df.loc[self.df[labelcol] > 0, labelcol] = 1
        return self

    def show(self):
        return self.df