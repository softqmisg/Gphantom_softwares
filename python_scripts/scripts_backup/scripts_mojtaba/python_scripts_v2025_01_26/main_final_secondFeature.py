"""
Binary classification with final second
"""
import os
from sklearn.exceptions import UndefinedMetricWarning

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt

from sklearn.preprocessing import OneHotEncoder
import missingno as msno
import seaborn as sns
from sklearn import linear_model, model_selection

from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, RandomForestClassifier

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from six.moves import cPickle as pickle
def plot_permutation_importance(clf, X, y, ax):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X.columns[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")
    return ax


def save_dict(di_: dict, filename_: str):
    """
    Save dictionary
    :param di_: input dictionary
    :param filename_: name to save dictionary
    """
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di
def main():

    # print(df.head())
    # exit()
    to_pop_all = ['Intrinsic_Subtype', 'ERNegative_ERPositive', ' PRNegative_PRPositive', 'HER2Negative_HER2Positive', 'Ki67Negative_Ki67Positive', 'Pathology', 'Histological_Grade']

    # to_pop, target_names, class_iids = ['Histological_Grade'], ['class 1', 'class 2', 'class 3'], [1, 2, 3]
    to_pop, target_names, class_iids = ['ERNegative_ERPositive', ' PRNegative_PRPositive', 'HER2Negative_HER2Positive', 'Ki67Negative_Ki67Positive'], ['class 1', 'class 2'], [0, 1, 2]
    # to_pop, target_names, class_iids = ['Intrinsic_Subtype'], ['class 1', 'class 2', 'class 3', 'class 4'], [0, 1, 2, 3]
    # target_names = ['class 1', 'class 2', 'class 3', 'class 4']

    # class_iids = [0, 1, 2, 3]
    # class_iids = [1, 2, 3]
    test_steps = range(2, 12)
    for col in to_pop:
        rf_list_acc_vs_test_step = []
        lr_list_acc_vs_test_step = []
        for test_size in test_steps:
            df = pd.read_csv('data/datasheetMamoSono.csv')
            results = {}
            print("#" * 20)
            print("#" * 20)
            print(f"{col}, {test_size}")
            print("#" * 20)
            print("#" * 20)
            dir_to_save = os.path.join('./results', col, str(test_size))
            os.makedirs(dir_to_save, exist_ok=True)
            y = df[col]
            df = df.drop(to_pop_all, axis=1)

            df = df.drop(149)
            y = np.int8(y.drop(149))

            # print("Split data to train and test set {}%, {}%, respectively".format(int((1 - test_size) * 100), int((test_size) * 100)))
            X_train, X_test, y_train, y_test = model_selection.train_test_split(df, y, test_size=.3,
                                                                                random_state=42, stratify=y)

            """
            Lasso CV for feature selection
            """
            print()
            print()
            print("=" * 50)
            print("=" * 15 + "Lasso CV for feature selection" + "=" * 15)
            regressor = linear_model.LassoCV().fit(X_train, y_train)


            importance = np.abs(regressor.coef_)
            # idx_features = (-importance).argsort()[:6]
            idx_features = (-importance).argsort()[:test_size]
            name_features = np.array(df.columns)[idx_features]

            with open(os.path.join(dir_to_save, 'selected_features.txt'), "w") as f:
                for item in name_features:
                    # write each item on a new line
                    f.write("%s\n" % item)
            f.close()

            selected_X_train = X_train[name_features]
            selected_X_test = X_test[name_features]

            print(f"{selected_X_train.shape = }")

            clf_logistic = LogisticRegression(random_state=0).fit(selected_X_train, y_train)

            yhat_logistic = clf_logistic.predict(selected_X_test)
            yhat_logistic_probab = clf_logistic.predict_proba(selected_X_test)

            logistic_classification_results = metrics.classification_report(y_test,
                                                                            yhat_logistic,
                                                                            target_names=target_names,
                                                                            output_dict=True)
            results['logistic_regression'] = logistic_classification_results
            lr_list_acc_vs_test_step.append(logistic_classification_results['accuracy'])

            print()
            print()
            print("=" * 50)
            print("=" * 15 + "Random Forest with 3-fold cross validation" + "=" * 15)
            param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
            base_estimator = RandomForestClassifier(random_state=0)
            rfc = GridSearchCV(base_estimator, param_grid, cv=3).fit(selected_X_train, y_train)
            yhat_rf = rfc.predict(selected_X_test)
            yhat_rf_probab = rfc.predict_proba(selected_X_test)
            print(f"RF best parameters: {rfc.best_estimator_}")

            result_rf_classifier = metrics.classification_report(y_test, yhat_rf,
                                                                 target_names=target_names,
                                                                 output_dict=True)

            rf_list_acc_vs_test_step.append(result_rf_classifier['accuracy'])
            results['Random_forest_results'] = result_rf_classifier

            df_results = pd.DataFrame.from_dict(results)
            df_results.to_excel(os.path.join(dir_to_save, 'results.xlsx'))

            cm = 1 / 2.54  # centimeters in inches
            f = plt.figure(figsize=(8.4 * cm, 8.4 * cm))
            ax0 = f.add_subplot(111)
            lw = 1.2
            ls = ['-', '--', ':', '-.']
            # for idx_class, class_iid in enumerate(class_iids):
            # class_iid = 2
            # y_test_ = np.where(np.array(y_test) == class_iid, 1, 0)

            fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat_logistic_probab[:, 1])
            roc_auc = metrics.auc(fpr, tpr)
            ax0.plot(fpr, tpr, lw=lw, label = "LR:(AUC={:.2f})".format(roc_auc))

            fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(y_test, yhat_rf_probab[:, 1])
            roc_auc = metrics.auc(fpr_rf, tpr_rf)
            ax0.plot(fpr_rf, tpr_rf, lw=lw, label="RF:(AUC={:.2f})".format(roc_auc))
            # exi
            plt.xlim([-.01, 1.01])
            plt.ylim([-0.01, 1.05])
            plt.xlabel('False Positive Rate', fontsize=8)
            plt.ylabel('True Positive Rate', fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)


            ax0.plot([0, 1], ls="--", c="k", label="AUC=0.5")
            ax0.spines.right.set_visible(False)
            ax0.spines.top.set_visible(False)

            # Only show ticks on the left and bottom spines
            ax0.yaxis.set_ticks_position('left')
            ax0.xaxis.set_ticks_position('bottom')

            ax0.spines['left'].set_linewidth(1.5)
            ax0.spines['bottom'].set_linewidth(1.5)
            plt.legend(fontsize=6)
            # plt.show()
            # exit()
            # plt.savefig(f'test.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.savefig(os.path.join(dir_to_save, 'auc.jpg'), bbox_inches='tight', pad_inches=0.1, dpi=800)
            plt.savefig(os.path.join(dir_to_save, 'auc.png'), bbox_inches='tight', pad_inches=0.1, dpi=800)


            """
            Precision-Recall Curve
            """
            cm = 1 / 2.54  # centimeters in inches
            f = plt.figure(figsize=(8.4 * cm, 8.4 * cm))
            ax0 = f.add_subplot(111)
            lw = 1.2
            ls = ['-', '--', ':', '-.']
            # for idx_class, class_iid in enumerate(class_iids):
            #     y_test_ = np.where(np.array(y_test) == class_iid, 1, 0)
            _fpr, _tpr, _thresholds = metrics.roc_curve(
                y_test, yhat_logistic_probab[:, 1],
                # pos_label=pos_label,
                # sample_weight=sample_weight,
                # drop_intermediate=drop_intermediate,
            )
            precision, recall, thresholds = metrics.precision_recall_curve(y_test, yhat_logistic_probab[:, 1])
            ap = metrics.average_precision_score(y_test, yhat_logistic_probab[:, 1])
            ax0.plot(precision, recall, lw=lw, ls=ls[0], label="LR:(AP={:.2f})".format(ap))

            precision_rf, recall_rf, thresholds_rf = metrics.precision_recall_curve(y_test, yhat_rf_probab[:, 1])
            ap_rf = metrics.average_precision_score(y_test, yhat_logistic_probab[:, 1])
            ax0.plot(precision_rf, recall_rf, lw=lw + .2, ls=ls[1], label="RF:(AP={:.2f})".format(ap_rf))

            plt.xlim([-.01, 1.01])
            plt.ylim([-0.01, 1.05])
            plt.xlabel('Recall', fontsize=8)
            plt.ylabel('Precision', fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            ax0.plot([1, 0], ls="--", c="k", label="AP=0.5")
            ax0.spines.right.set_visible(False)
            ax0.spines.top.set_visible(False)

            # Only show ticks on the left and bottom spines
            ax0.yaxis.set_ticks_position('left')
            ax0.xaxis.set_ticks_position('bottom')

            ax0.spines['left'].set_linewidth(1.5)
            ax0.spines['bottom'].set_linewidth(1.5)
            plt.legend(fontsize=6)
            # plt.savefig(f'test.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.savefig(os.path.join(dir_to_save, 'pr.jpg'), bbox_inches='tight', pad_inches=0.1, dpi=800)
            plt.savefig(os.path.join(dir_to_save, 'pr.png'), bbox_inches='tight', pad_inches=0.1, dpi=800)
            plt.savefig(os.path.join(dir_to_save, 'pr.pdf'), bbox_inches='tight', pad_inches=0.1, dpi=800)


            mdi_importances = pd.Series(importance, index=X_train.columns)
            tree_importance_sorted_idx = np.argsort(importance)
            tree_indices = np.arange(0, len(importance)) + 0.5
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            mdi_importances.sort_values().plot.barh(ax=ax1)
            ax1.set_xlabel("Feature importance")
            plot_permutation_importance(regressor, X_train, y_train, ax2)
            ax2.set_xlabel("Decrease in accuracy score")
            # fig.suptitle(
            #     "Impurity-based vs. permutation importances on multicollinear features (train set)"
            # )
            _ = fig.tight_layout()

            plt.savefig(os.path.join(dir_to_save, f'featureImportance.jpg'), bbox_inches='tight', pad_inches=0.1, dpi=800)
            plt.savefig(os.path.join(dir_to_save, f'featureImportance.png'), bbox_inches='tight', pad_inches=0.1, dpi=800)
            plt.savefig(os.path.join(dir_to_save, f'featureImportance.pdf'), bbox_inches='tight', pad_inches=0.1, dpi=800)


        cm = 1 / 2.54  # centimeters in inches
        f = plt.figure(figsize=(8.4 * cm, 4.4 * cm))
        ax0 = f.add_subplot(111)
        lw = 1.2
        ls = ['-', '--', ':', '-.']
        ax0.plot(np.array(test_steps), np.array(lr_list_acc_vs_test_step) * 100, marker='^', lw=lw, ls=ls[0], label="LR")
        ax0.plot(np.array(test_steps), np.array(rf_list_acc_vs_test_step) * 100, marker='o', lw=lw, ls=ls[1], label="RF")
        plt.xlabel('Test size (%)', fontsize=8)
        plt.ylabel('Accuracy (%)', fontsize=8)
        plt.xticks(np.array(test_steps), np.array(test_steps), fontsize=8)
        plt.yticks(fontsize=8)

        # ax0.plot([1, 0], ls="--", c="k", label="AP=0.5")
        ax0.spines.right.set_visible(False)
        ax0.spines.top.set_visible(False)

        # Only show ticks on the left and bottom spines
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')

        ax0.spines['left'].set_linewidth(1.5)
        ax0.spines['bottom'].set_linewidth(1.5)
        plt.legend(fontsize=6)
        plt.savefig(os.path.join('./results', col, f'{col}_acc_vs_testSize.png'), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.savefig(os.path.join('./results', col, f'{col}_acc_vs_testSize.jpg'), bbox_inches='tight', pad_inches=0.1, dpi=800)
        plt.savefig(os.path.join('./results', col, f'{col}_acc_vs_testSize.pdf'), bbox_inches='tight', pad_inches=0.1, dpi=800)



if __name__ == '__main__':
    main()