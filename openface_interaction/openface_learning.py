import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging
logging.getLogger().setLevel(logging.DEBUG)
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from sklearn.externals import joblib
from imblearn.under_sampling import InstanceHardnessThreshold, RandomUnderSampler, CondensedNearestNeighbour, \
    TomekLinks, NearMiss, ClusterCentroids, OneSidedSelection, NeighbourhoodCleaningRule


def get_dataset(path_to_csv):
    df = pd.read_csv(path_to_csv, header=0)
    df = df._get_numeric_data()
    y_x = None
    y_y = None
    try:
        y_x = df['expected_x']
        y_y = df['expected_y']
    except:
        print("WARNING: No expected_x in dataset : {0}".format(path_to_csv))
    try:
        df = df.drop(['expected_y', 'expected_x'], axis=1)
    except:
        print("WARNING: No expected_x, expected_y in dataset: {0}".format(path_to_csv))
    sorted_df = df.reindex_axis(sorted(df.columns), axis=1)
    # print("loaded db : {0}".format(sorted_df.describe()))

    return sorted_df, y_x, y_y
    # write the reverse_df to an excel spreadsheet

def coordinates_to_class(n_classes):
    pass

def transform_ys_to_bins(y_x, y_y):
    df = pd.DataFrame([y_x, y_y]).transpose()
    bins = (df // 960 * 540).round(1).stack().groupby(level=0).apply(tuple)
    return bins


def transform_ys_to_bins_v2(y_x, y_y):

    df = pd.DataFrame([y_x, y_y]).transpose()
    x_cut = pd.cut(df.expected_x, np.linspace(-1920, 1920, num=5), right=False)
    y_cut = pd.cut(df.expected_y, np.linspace(0, 1080, num=3), right=False)
    zibi = pd.DataFrame([x_cut, y_cut]).transpose()
    return zibi.apply(lambda x: (x['expected_x'].left,  x['expected_y'].left), axis=1)
    # return df.groupby([x_cut, y_cut]).apply(tuple)



if __name__=="__main__":

    X, y_x, y_y = get_dataset(r"D:\Programming\pyEye\train_eye\feature_extraction_openface\joint_csv.csv")

    ####
    Y = transform_ys_to_bins_v2(y_x, y_y)
    mlb = LabelEncoder()
    mlb.fit(Y)
    Y = mlb.transform(Y)
    joblib.dump(mlb, "models/label_encoder.pkl")




    # InstanceHardnessThreshold

    ###
    #In general it is a good idea to scale the data
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, "models/scalar.pkl")
    X=scaler.transform(X)

    logging.info("[V] Re-fitted Xs shape : {0} ".format(X.shape))
    logging.info("[V] y'ss shape : {0} ".format(y_x.shape))

    pca = PCA(n_components=20)
    pca.fit(X,Y)
    joblib.dump(pca, "models/pca.pkl")
    X = pca.transform(X)


    # under_sample
    X_orig = X
    Y_orig = Y

    smote_enn = SMOTEENN(random_state=0)
    X, Y = smote_enn.fit_sample(X, Y)
    # smote_tomek = SMOTETomek(random_state=0)
    # X, Y = smote_tomek.fit_sample(X, Y)
    print(X.shape)

    clf = RandomForestClassifier(random_state=0, n_estimators=50)
    logging.info("Training...")
    clf.fit(X,Y)
    # joblib.dump(clf, "models/clf_randomforestclassifier.pkl")

    # score = cross_val_score(clf, X, Y, cv=10, scoring='neg_log_loss')

    y_pred = cross_val_predict(clf, X, Y, cv=10)
    conf_mat = confusion_matrix(Y, y_pred)
    plt.imshow(conf_mat, cmap='hot', interpolation='nearest')
    plt.show()
    print(conf_mat )
    exit()


    ##### OLD ####
    clf_x = RandomForestRegressor(max_depth=3, random_state=0)
    clf_y = RandomForestRegressor(max_depth=3, random_state=0)
    clf_x.fit(x_new, y_x)
    clf_y.fit(x_new, y_y)
    joblib.dump(clf_x, "models/clf_randomforest_x.pkl")
    joblib.dump(clf_y, "models/clf_randomforest_y.pkl")
    print(clf_x.feature_importances_)
    predicted = clf_x.predict(x_new)
    # exit()
    # score = cross_val_score(clf, X, y, cv=10, scoring='neg_mean_absolute_error')
    # print(score)





    def myplot(score,coeff,labels=None):
        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]

        plt.scatter(xs ,ys, c = y) #without scaling
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
            if labels is None:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

    #Call the function.
    myplot(predicted)
    plt.show()