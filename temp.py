import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample
from temp_svm import SVM2 as svm
from temp_svm import KNN as knn
from temp_svm import LR as lr
import temp_data_cleaning as data_cleaning
pd.set_option('display.max_columns', 500)
ROOT = r"C:\Users\huangdajing\Desktop\linus hw"
CSV_PATH= r"C:\Users\huangdajing\Desktop\linus hw\card.csv"
DESCRIPTION_PATH = r"C:\Users\huangdajing\Desktop\linus hw\description.csv"
HEATMAP_PATH = r"C:\Users\huangdajing\Desktop\linus hw\heatmap.png"
LABEL_PATH = r"C:\Users\huangdajing\Desktop\linus hw\label.png"

x_train_path = "./"+"X_train.npy"
x_test_path ="./"+"X_test.npy"
y_train_path ="./"+"y_train.npy"
y_test_path ="./"+"y_test.npy"

UPSAMPLING = False
DOWNSAMPLING = True

def TOP_get_save_description(df):
    print(df.describe())
    description = df.describe()
    description.to_csv(DESCRIPTION_PATH)


def print_corr(df):
    df = df[df.columns[1:-1]]
    corr = df.corr()
    ax = sns.heatmap(corr)
    fig = ax.get_figure()
    fig.savefig(HEATMAP_PATH)
    plt.show()


def print_categorial_plot(df, col_name, title, xlabel, ylabel ="Frequency"):
    save_path = os.path.join(ROOT,title.replace(" ","_").lower())
    y = df[col_name]
    labels = pd.value_counts(y,sort=True)
    ax = labels.plot(kind= "bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig = ax.get_figure()
    fig.savefig(save_path)
    plt.show()


def TOP_visualisation_features(df):
    print_categorial_plot(df, "Y", "Label distribution", "Class")
    print_categorial_plot(df, "X2", "Gender distribution", "Gender")
    print_categorial_plot(df, "X3", "Education Distribution", "Education")
    print_categorial_plot(df, "X4", "Marriage Distribution", "Marriage")
    for i in range(6,12):
        xn = "X"+str(i)
        print_categorial_plot(df, xn, xn+" Distribution", xn)


def TOP_df_init(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(df.index[0])  # drop header row
    df = df.astype(int)  # convert objects type to float for describtion
    return df


def sclae(df):
    # cols = df.columns[1:-2]
    cols =['X5','X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21',
       'X22', 'X23']
    for col in cols:
        transformed = StandardScaler().fit_transform(df[col].values.reshape(-1, 1))
        df = df.drop([col],axis =1)
        df[col]= transformed
    df["Y"].astype(int)
    return df


def do_down_sampling(df):
    # Lets shuffle the data before creating the subsamples
    df = df.sample(frac=1)
    label_1 = df[df['Y'] == 1]
    len_1=len(label_1)
    label_0 = df[df['Y'] == 0][:len_1]
    new_df = pd.concat([label_0, label_1])
    new_df = new_df.sample(frac=1, random_state=42)
    return new_df


def do_up_sampling(X_train, y_train):
    print (X_train.shape,y_train.shape)
    joined = pd.DataFrame(X_train)
    joined["Y"] = y_train

    # separate minority and majority classes
    not_defaulted = joined[joined.Y == 0]
    defaulted = joined[joined.Y == 1]

    # upsample minority
    fraud_upsampled = resample(defaulted,
                               replace=True,  # sample with replacement
                               n_samples=len(not_defaulted),  # match number in majority class
                               random_state=27)  # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([not_defaulted, fraud_upsampled])

    # check new class counts
    upsampled.Y.value_counts()
    X_train = df.drop(['Y'], axis=1)
    y_train = pd.DataFrame(df['Y'])

    return X_train,y_train


def pca2(df,continuous_var):
    print("Original df : ", df.shape)
    y_column = ['Y']
    # continuous_var = ['X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21','X22', 'X23']

    X_df = df[continuous_var]
    y_df = df['Y']

    df_no_label = df.drop(y_column,axis=1)
    cat_var_df = df_no_label.drop(continuous_var,axis=1)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X_df.values)
    principalDf = pd.DataFrame(data=principalComponents)
    finalDf = pd.concat([principalDf, cat_var_df, y_df], axis=1)
    finalDf = finalDf.dropna(how="any")
    print("After PCA df : ", finalDf.shape,"\n")
    return finalDf


def ndarray2df(ndarray,column_names):
    return  pd.DataFrame(ndarray, columns=column_names)


def data_splitting(df):

    # prepare the data
    features = df.drop(['Y','ID'], axis=1)
    col_name = features.columns
    labels = pd.DataFrame(df['Y'])

    feature_array = features.values
    label_array = labels.values

    X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.20)

    # convert ndarray to df
    X_train_df = ndarray2df(X_train, col_name)
    X_test_df = ndarray2df(X_test, col_name)
    y_train_df = ndarray2df(y_train, ["Y"])
    y_test_df = ndarray2df(y_test, ["Y"])

    print(X_train_df.shape,X_test_df.shape,y_train_df.shape,y_test_df.shape)

    return X_train_df, X_test_df, y_train_df, y_test_df


def data_split_post_processing(X_train_df, X_test_df, y_train_df, y_test_df):
    new_train_df = pd.concat([X_train_df, y_train_df], axis=1)
    new_test_df = pd.concat([X_test_df, y_test_df], axis=1)

    bill =['X12', 'X13', 'X14', 'X15', 'X16', 'X17']
    pay = ['X18', 'X19', 'X20', 'X21', 'X22', 'X23']

    pca_ed_train_df = pca2(new_train_df,bill)
    pca_ed_train_df = pca2(pca_ed_train_df, pay)

    pca_ed_test_df = pca2(new_test_df,bill)
    pca_ed_test_df = pca2(pca_ed_test_df, pay)

    # pca_ed_train_df=new_train_df
    # pca_ed_test_df=new_test_df
    y_train = pca_ed_train_df["Y"].values
    X_train = pca_ed_train_df.drop(["Y"],axis= 1).values

    y_test = pca_ed_test_df["Y"].values
    X_test = pca_ed_test_df.drop(["Y"],axis= 1).values

    """I DONT UNDERSTAND THIS PART"""
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    return X_train, X_test, y_train, y_test


def process_preprocessing():
    df = TOP_df_init(CSV_PATH)
    # df = data_cleaning.remove_dup(df)
    to_replace = []
    df = data_cleaning.add_ID_col(df)

    df = data_cleaning.replace_cat_to_binary(df, "X6")
    df = data_cleaning.replace_cat_to_binary(df, "X7")
    df = data_cleaning.replace_cat_to_binary(df, "X8")
    df = data_cleaning.replace_cat_to_binary(df, "X9")
    df = data_cleaning.replace_cat_to_binary(df, "X10")
    df = data_cleaning.replace_cat_to_binary(df, "X11")

    df = data_cleaning.replace(df,"X4",[0], 3) # replace marriage
    df = data_cleaning.replace(df,"X3",[0,5,6], 4) # replace education 0,5,6 to 4

    if DOWNSAMPLING:
        df = do_down_sampling(df)

    # data_cleaning.check_col_y_relation(df,"X2")
    # data_cleaning.check_col_y_relation(df,"X3")
    # data_cleaning.check_col_y_relation(df,"X4")
    # data_cleaning.check_col_y_relation(df,"X12")


    # df.to_csv("test.csv")
    df = data_cleaning.cat_to_onehot(df,"X2")
    df = data_cleaning.cat_to_onehot(df,"X3")
    df = data_cleaning.cat_to_onehot(df,"X4")

    X_train_df, X_test_df, y_train_df, y_test_df = data_splitting(df)

    X_train, X_test, y_train, y_test  = data_split_post_processing(X_train_df, X_test_df, y_train_df, y_test_df)


    np.save(x_train_path, X_train)
    np.save(x_test_path,X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)


def process_train():
    X_train = np.load(x_train_path)
    X_test= np.load(x_test_path)
    y_train= np.load(y_train_path)
    y_test= np.load(y_test_path)

    svm_class = svm()
    y_pred = svm_class.train(X_train, X_test, y_train)
    svm_class.varify(y_test,y_pred)


    lr_class = lr()
    y_pred = lr_class.train(X_train, X_test, y_train)
    lr_class.varify(y_test,y_pred)


# process_preprocessing()
process_train()
