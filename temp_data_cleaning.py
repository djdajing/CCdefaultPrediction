import pandas as pd
import matplotlib.pyplot as plt
def add_ID_col(df):
    new_col =["ID"]
    df_col = df.columns[1:]
    for col in df_col:
        new_col.append(col)
    df.columns = new_col
    return df


def remove_dup(df):
    df2 = df[['X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21','X22', 'X23']]
    df_return = df.loc[~(df2==0).all(axis=1)]
    return df_return

def cat_to_onehot(df,column):
    a = pd.get_dummies(df[column],prefix=column)
    df = df.drop([column],axis =1)
    new_df = pd.concat([df,a],axis =1)
    return new_df


def replace(df, col, unwanted_vals, wanted_val):
    print(df[col].value_counts())
    for unwanted_val in unwanted_vals:
        fil = (df[col] == unwanted_val)
        df.loc[fil, col] = wanted_val
    print(df[col].value_counts())
    return df


def replace_cat_to_binary(df, col):
    for i in df.index:
        original = df.at[i,col]
        if original < 0:
            df.at[i,col] = 0
        else:
            df.at[i,col]=1
    return df


def check_col_y_relation(df,col):
    df_new = df.groupby([col,'Y']).size().unstack(1)# get table of y =Y x = Col,
    df_new.plot(kind='bar', stacked=True)
    df_new['percentage_default_1'] = (df_new[1] / (df_new[0] + df_new[1]))  # col index 0 is cases that did not default
    print(df_new)
    percent_df = df_new[['percentage_default_1']]
    print(percent_df)
    percent_df.plot(kind='bar', stacked=True)
    plt.show()

def check_y_col_relation(df,col):
    df_new = df.groupby(['Y',col]).size().unstack(1)# get table of y =Y x = Col,
    df_new.plot(kind='bar', stacked=True)
    df_new['percentage_default_1'] = (df_new[1] / (df_new[0] + df_new[1]))  # col index 0 is cases that did not default
    print(df_new)
    percent_df = df_new[['percentage_default_1']]
    print(percent_df)
    percent_df.plot(kind='bar', stacked=True)
    plt.show()

