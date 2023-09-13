

# from IPython.display import display 

# import pandas as pd
# import numpy as np
# import missingno as msno 
import seaborn as sns
import matplotlib.pyplot as plt 

#sklearn

# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import f1_score, fbeta_score
# from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, make_scorer
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import accuracy_score, balanced_accuracy_score

import numpy as np
# import matplotlib.pyplot as plt
# import operator
# # import to divide our data into train and test data
# from sklearn.model_selection import train_test_split
# # import to create polynomial features
# from sklearn.preprocessing import PolynomialFeatures
# # import of the linear regression model
# from sklearn.linear_model import LinearRegression
# # import of our evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

#RSEED = 12



from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, make_scorer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Calculate metric
def calculate_metrics(y_train, y_pred_train, y_test, y_pred_test):
    """Calculate and print out RMSE and R2 for train and test data

    Args:
        y_train (array): true values of y_train
        y_pred_train (array): predicted values of model for y_train
        y_test (array): true values of y_test
        y_pred_test (array): predicted values of model for y_test
    """

    print("Metrics on training data") 
    rmse = np.sqrt(mean_squared_error(y_train,y_pred_train))
    r2 = r2_score(y_train,y_pred_train)
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))
    print("---"*10)
    
    # Calculate metric
    print("Metrics on test data")  
    rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    # you can get the same result with this line:
    # rmse = np.sqrt(mean_squared_error(y_test,y_pred_test))

    r2 = r2_score(y_test,y_pred_test)
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))
    print("---"*10)


def error_analysis(y_test, y_pred_test):
    """Generated true vs. predicted values and residual scatter plot for models

    Args:
        y_test (array): true values for y_test
        y_pred_test (array): predicted values of model for y_test
    """     
    # Calculate residuals
    residuals = y_test - y_pred_test
    
    # Plot real vs. predicted values 
    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    plt.subplots_adjust(right=1)
    plt.suptitle('Error Analysis')
    
    ax[0].scatter(y_pred_test, y_test, color="#FF5A36", alpha=0.7)
    ax[0].plot([-400, 350], [-400, 350], color="#193251")
    ax[0].set_title("True vs. predicted values", fontsize=16)
    ax[0].set_xlabel("predicted values")
    ax[0].set_ylabel("true values")
    ax[0].set_xlim((y_pred_test.min()-10), (y_pred_test.max()+10))
    ax[0].set_ylim((y_test.min()-40), (y_test.max()+40))
    
    ax[1].scatter(y_pred_test, residuals, color="#FF5A36", alpha=0.7)
    ax[1].plot([-1000, 350], [0,0], color="#193251")
    ax[1].set_title("Residual Scatter Plot", fontsize=16)
    ax[1].set_xlabel("predicted values")
    ax[1].set_ylabel("residuals")
    ax[1].set_xlim((y_pred_test.min()-10), (y_pred_test.max()+10))
    ax[1].set_ylim((residuals.min()-10), (residuals.max()+10));




# write function displaying our model metrics
def our_metrics(y_true, y_pred, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm, cmap="YlGnBu", annot=True);
        print('Model Metrics and Normalized Confusion Matrix')
        print("_____________________")
        print("_____________________")
    else:
        print('Model Metrics and Confusion Matrix without Normalization')
        print("_____________________")
        print("_____________________")
        sns.heatmap(cm, cmap="YlGnBu", annot=True);
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("_____________________")
    print('F1-score:', round(f1_score(y_true, y_pred), 4))
    print("_____________________")
    print('Fbeta_score with beta=1.5:', round(fbeta_score(y_true, y_pred, beta=1.5), 4)) 
    print("_____________________")
    print('Fbeta_score with beta=2:', round(fbeta_score(y_true, y_pred, beta=2), 4)) 
    print("_____________________")
    print('Fbeta_score with beta=3:', round(fbeta_score(y_true, y_pred, beta=3), 4)) 
    print("_____________________")
    print('Recall', round(recall_score(y_true, y_pred), 4))
    print("_____________________")
    print('Specificity', round(recall_score(y_true, y_pred, pos_label=0), 4))

    
# make the Fbeta scorers needed for the grid search
def get_f15():
    f15_scorer = make_scorer(fbeta_score, beta=1.5)
    return f15_scorer

def get_f2():
    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    return ftwo_scorer

def get_f3():
    fthree_scorer = make_scorer(fbeta_score, beta=3)
    return fthree_scorer

# evaluation metrics : confusion matrix,, accuracy, balance accuracy, classification report
def eval_metrics(y_test, y_pred): 
    """
    Summary:
        Function to calculate the accuracy and balanced accuracy score for imbalanced data, get the confusion 
        matrix as well as the classification report of the ML 
        model based on the predictions and true target values for the test set.

    Args:
        y_test (numpy.ndarray): test target data
        y_pred (numpy.ndarray): predictions based on test data
    """    
    
    print("-----"*15)
    print(f'''Confusion Matrix: 
    {confusion_matrix(y_test, y_pred)} ''') 
    
    print("-----"*15)
    print (f''' Accuracy : 
    {(accuracy_score(y_test, y_pred).round(2)) * 100} ''')

    print("-----"*15)
    print (f''' Balanced Accuracy : 
    {(balanced_accuracy_score(y_test, y_pred).round(2)) * 100} ''')
    
    print("-----"*15)
    print(f'''Report :  
    {classification_report(y_test, y_pred)} ''') 


# # eval scoring metrics : recall, precision, f1_score, roc_auc_score, fpr, tpr
# def evaluate_model(predictions, probs, train_predictions, train_probs):
#     """Compare machine learning model to baseline performance.
#     Computes statistics and shows ROC curve."""
    
#     baseline = {}
    
#     baseline['recall'] = recall_score(y_test, [1 for _ in range(len(y_test))])
#     baseline['precision'] = precision_score(y_test, [1 for _ in range(len(y_test))])
#     baseline['f1_score'] = f1_score(y_test, [1 for _ in range(len(y_test))])
#     baseline['roc'] = 0.5
    
#     results = {}
    
#     results['recall'] = recall_score(y_test, predictions)
#     results['precision'] = precision_score(y_test, predictions)
#     results['f1_score'] = f1_score(y_test, predictions)
#     results['roc'] = roc_auc_score(y_test, probs)
    
#     # train_results = {}
#     # train_results['recall'] = recall_score(y_test, train_predictions)
#     # train_results['precision'] = precision_score(y_test, train_predictions)
#     # train_results['f1_score'] = f1_score(y_test, predictions)
#     # train_results['roc'] = roc_auc_score(y_test, train_probs)
    
#     for metric in ['recall', 'precision', 'f1_score', 'roc']:
#         #print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
#         print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} ')

#     # Calculate false positive rates and true positive rates
#     base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
#     model_fpr, model_tpr, _ = roc_curve(y_test, probs)

#     plt.figure(figsize = (8, 6))
#     plt.rcParams['font.size'] = 16
    
#     # Plot both curves
#     plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
#     plt.plot(model_fpr, model_tpr, 'r', label = 'model')
#     plt.legend()
#     plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves')

if (__name__ == "__main__"):
    print(get_f15())
    print(get_f2())
    print(get_f3())