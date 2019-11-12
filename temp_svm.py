from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
def feature_selection(X):
    from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
    print(X.shape)
    sel = VarianceThreshold(0.2)
    X_VT = sel.fit_transform(X)
    # X_new = SelectKBest(chi2, k=11).fit_transform(X_VT, y)
    # finalDf = pd.concat([X_new, y], axis=1)
    return X_VT

class SVM :
    def __init__(self):
        self.classifier = SVC(kernel='rbf')

    def train(self,X_train,X_test,y_train):
        # X_train = feature_selection(X_train)
        # X_test = feature_selection(X_test)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        return y_pred

    def varify(self,y_test,y_pred):
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        print(roc_auc_score(y_test, y_pred))

        plt.show()

class SVM2 :
    def __init__(self):
        pass

    def train(self,X_train,X_test,y_train):
        OR =False
        print("in train")
        steps = [('scaler', StandardScaler()), ('SVM', SVC())]

        if OR:
            pipeline = Pipeline(steps)  # define the pipeline object.
            parameteres = {'SVM__C': [0.3,0.31], 'SVM__gamma': [0.1]}
            grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5)
        else :
            parameters = {'kernel': ['rbf'], 'C': [10,15], 'gamma': [2,1,5]}
            svc = SVC()
            grid = GridSearchCV(svc, parameters, cv=10)

        print("fitting")
        grid.fit(X_train, y_train)
        print("fitted")
        print("Best param", grid.best_params_)
        y_pred = grid.predict(X_test)
        return y_pred

    def varify(self,y_test,y_pred):
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        print(roc_auc_score(y_test, y_pred))

        plt.show()

class KNN:
    def __init__(self):
        pass

    @staticmethod
    def train(X_train,y_train,X_test,y_test):
        neighbours = np.arange(1, 25)
        train_accuracy = np.empty(len(neighbours))
        test_accuracy = np.empty(len(neighbours))

        for i, k in enumerate(neighbours):
            # Setup a knn classifier with k neighbors
            knn = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)

            # Fit the model
            knn.fit(X_train, y_train.ravel())

            # Compute accuracy on the training set
            train_accuracy[i] = knn.score(X_train, y_train.ravel())

            # Compute accuracy on the test set
            test_accuracy[i] = knn.score(X_test, y_test.ravel())

        # Generate plot
        plt.title('k-NN Varying number of neighbors')
        plt.plot(neighbours, test_accuracy, label='Testing Accuracy')
        plt.plot(neighbours, train_accuracy, label='Training accuracy')
        plt.legend()
        plt.xlabel('Number of neighbors')
        plt.ylabel('Accuracy')
        plt.show()


class LR:
    def __init__(self):
        self.classifier = LogisticRegression()

    def train(self,X_train,X_test,y_train):

        self.classifier.fit(X_train, y_train)
        y_pred =self.classifier.predict(X_test)
        return y_pred

    def varify(self,y_test,y_pred):
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        print(roc_auc_score(y_test, y_pred))

        plt.show()