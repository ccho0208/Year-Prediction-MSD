import os, sys, time, random, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics

# reproducibility
seed = 1234567
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)


def main():
    start = time.time()

    # 0) load MSD dataset
    df = pd.read_csv('./YearPredictionMSD2.csv')

    #
    # 1) data pre-processing -------------------------------------------------------------------------------------------
    # a) group release years into decades
    df['label'] = df.label.apply(lambda year: year - (year % 10))
    # sns.countplot(y='label', data=df)
    # plt.show()

    # b) scale the dataset using min-max scaling, each feature is reduced to a range of 0 to 1
    df.iloc[:,1:] = (df.iloc[:,1:] - df.iloc[:,1:].min()) / (df.iloc[:,1:].max() - df.iloc[:,1:].min())

    # c) downsample the data: pick equal number of random samples for each category (release decade)
    df_t = df[df.label > 1940]
    min_samples = df_t.label.value_counts().min()
    decades = df_t.label.unique()
    df_sampled = pd.DataFrame(columns=df_t.columns)
    for decade in decades:
        df_sampled = df_sampled.append(df_t[df_t.label == decade].sample(min_samples))
    df_sampled.label = df_sampled.label.astype(int)
    # sns.countplot(x='label', data=df_sampled)
    # plt.show()

    # d) split the dataset
    df_sampled = shuffle(df_sampled)
    df_train, df_test = train_test_split(df_sampled, test_size=0.3)

    X_train = df_train.iloc[:,1:].values
    y_train = df_train.iloc[:,0].values
    X_test = df_test.iloc[:,1:].values
    y_test = df_test.iloc[:,0].values
    print('X_train: ', X_train.shape, ', y_train: ', y_train.shape)
    print('X_test: ', X_test.shape, ', y_test: ', y_test.shape)

    #
    # 2) train classifier: SVM -----------------------------------------------------------------------------------------
    clf = svm.SVC(kernel='rbf', C=10, gamma=5);
    clf.fit(X_train, y_train)

    #
    # 3) prediction ----------------------------------------------------------------------------------------------------
    pred = clf.predict(X_test)

    # evaluate model's performance
    print('Mean Absolute Error: %f' % np.mean(np.absolute(y_test - pred)))
    print('Mean Square Error: %f' % np.sqrt(np.mean(np.absolute(y_test - pred)^2)))

    # model: Mean Absolute Error: 9.947813, Mean Square Error: 3.225665

    #
    # 4) print out the prediction report -------------------------------------------------------------------------------
    print('classification report for classifier %s:\n%s\n' % (clf, metrics.classification_report(y_test, pred)))

    # plot confusion matrix
    conf_mat = metrics.confusion_matrix(y_test, pred)

    labels = sorted(df_test.label.unique())
    plot_confusion_matrix(conf_mat, classes=["{:02d}'s".format(label%100) for label in labels], fn='conf.png')

    elapsed = time.time() - start
    print('Elapsed {:.2f} minutes'.format(elapsed/60.0))
    return 0

def plot_confusion_matrix(conf_mat, classes, fn, normalize=True, title=None, cmap=plt.cm.Blues):
    """ plot the confusion matrix """
    if normalize:
        cm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt), horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
    plt.savefig(fn)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,IOError) as e:
        sys.exit(e)
