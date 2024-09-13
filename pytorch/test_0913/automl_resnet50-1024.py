import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

test_result = pd.read_csv("./results/test_resnet50_latent-1024_size-256_batch-8.csv")
valid_result = pd.read_csv("./results/valid_resnet50_latent-1024_size-256_batch-8.csv")
train_result = pd.read_csv("./results/train_resnet50_latent-1024_size-256_batch-8.csv")

len(test_result), len(valid_result), len(train_result)

from sklearn.model_selection import train_test_split

df = pd.concat([train_result, valid_result, test_result], ignore_index=True)
train_df, test_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

from pycaret.classification import (setup, compare_models, 
                        plot_model, evaluate_model, predict_model)

s = setup(train_df, target="label", session_id=123)

best_model = compare_models()

import matplotlib as mpl

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, log_loss
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_confusion_matrix(target, prediction, labels=None):
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    cm = confusion_matrix(target, prediction, labels=labels)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, 
                    display_labels=["Normal", "Abnormal"])
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm/cm.sum(), 
                    display_labels=["Normal", "Abnormal"])

    disp1.plot(cmap='viridis', ax=ax1)
    disp2.plot(cmap='viridis', ax=ax2)

    for ax in (ax1, ax2):
        ax.grid(False)
        ax.tick_params(labelsize=10)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)

    fig.tight_layout()
    plt.show()

## Test Data
pred = predict_model(best_model, test_df)
show_confusion_matrix(pred["label"], pred["prediction_label"], labels=[1, 0])

## Train Data
pred = predict_model(best_model, train_df)
show_confusion_matrix(pred["label"], pred["prediction_label"], labels=[1, 0])
