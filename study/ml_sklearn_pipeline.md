### Sklearn Pipeline

- see: pjt_binary_clf_tuning.ipynb
- see: ml_cls-3_02_grid_cv.ipynb

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pathlib
import pickle

from collections import Counter

import warnings
warnings.filterwarnings(action='ignore')
```

### 함수 정의

```python
from time import time
from datetime import timedelta
from tqdm import tqdm
import sys

def get_scores(y_true, y_pred, name=None, tact=None):
    cm = multilabel_confusion_matrix(y_true, y_pred)
    (FN, FP), (TN, TP) = cm[-1]
    prec = (TP)/(FP + TP)
    recall = (TP) / (TN + TP)
    f1 = 2*(prec*recall)/(prec + recall)

    scores = {}
    scores['Model'] = [name] if name is not None else ['']
    scores['Accuracy'] = [accuracy_score(y_true, y_pred)]
    scores['Prec.'] = prec
    scores['Recall'] = recall
    scores['F1'] = f1

    df = pd.DataFrame(data=scores)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: round(x, 4))
    df['Tact'] = [tact] if tact is not None else ['']
    return df

def train(model, train_data, test_data, name=None, save_model=False, model_dir='./models'):
    x_train, y_train = train_data
    x_test, y_test = test_data

    start_time = time()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    tact = str(timedelta(seconds=time()-start_time)).split('.')[0]
    scores = get_scores(y_test, y_pred, name=name, tact=tact)
    f1 = scores['F1'].values
    
    if name is not None and save_model:
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(model_dir, "sklearn_F1-%.4f_%s.pkl" % (f1, name))
        pickle.dump(model, open(model_path, 'wb'))
    
    return scores

def train_models(models, train_data, test_data, save_model=True, sort_values=False, model_dir='./models'):
    results = pd.DataFrame()
    pbar = tqdm(models.items(), leave=False, file=sys.stdout, ascii=True)
    for i, (name, model) in enumerate(pbar):
        pbar.set_description("[%2d/%2d] %s" % (i+1, len(pbar), name))
        scores = train(model, train_data, test_data, name=name, 
                       save_model=save_model, model_dir=model_dir)
        results = pd.concat([results, scores])

    results.reset_index(drop=True, inplace=True)
    if sort_values:
        results.sort_values(by=["F1"], inplace=True, ascending=False)
    return results
```

### 데이터 불러오기

```python
from sklearn.model_selection import train_test_split

data = pd.read_csv(os.path.join("./common", "data.csv"))
data = data.iloc[:, :-1].values

target_map = {'np': 0, 'pn': 1, 'bundle': 2}
target = pd.read_csv(os.path.join("./common", "target.csv"))
target = target["target"].apply(lambda x: target_map[x]).values

x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, stratify=target, random_state=21)

print("x_train:", x_train.shape, x_train.dtype, x_train.min(), x_train.max())
print("y_train:", y_train.shape, y_train.dtype, y_train.min(), y_train.max(), Counter(y_train))
```

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

std = StandardScaler()
x_train1 = std.fit_transform(x_train)
print("x_train:", x_train1.shape, x_train1.dtype, x_train1.min(), x_train1.max())

minmax = MinMaxScaler()
x_train2 = minmax.fit_transform(x_train)
print("x_train:", x_train2.shape, x_train2.dtype, x_train2.min(), x_train2.max())
```

```python
## Preprocessing
from imblearn.pipeline import Pipeline as pipe
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA

pca = PCA(svd_solver='full', n_components=0.99, random_state=21)

smote = SMOTE(random_state=21, n_jobs=-1)
oversampler = RandomOverSampler(random_state=21)
adasyn = ADASYN(random_state=21, n_jobs=-1)
blsmote = BorderlineSMOTE(random_state=21, n_jobs=-1)
svmsomote = SVMSMOTE(random_state=21, n_jobs=-1)

undersampler = RandomUnderSampler(random_state=21)
```

### 기본 모델 평가

```python
## Sklearn basic models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

kwargs = dict(random_state=21, n_jobs=-1)

nb = GaussianNB()
ridge = RidgeClassifier(random_state=21)
dt = DecisionTreeClassifier(random_state=21)
knn = KNeighborsClassifier(n_jobs=-1)
svc = SVC(random_state=21)
lr = LogisticRegression(**kwargs)
rf = RandomForestClassifier(**kwargs)

lr_bal = LogisticRegression(**kwargs, class_weight='balanced')
rf_bal = RandomForestClassifier(**kwargs, class_weight='balanced')
```

```python
models = {}

models["nb"]    = pipe([('scaler', None), ('clf', nb)])
models["ridge"] = pipe([('scaler', None), ('clf', ridge)])
models["dt"]    = pipe([('scaler', None), ('clf', dt)])
models["knn"]   = pipe([('scaler', None), ('clf', knn)])
models["svc"]   = pipe([('scaler', None), ('clf', svc)])
models['lr']    = pipe([('scaler', None), ('clf', lr)])
models["rf"]    = pipe([('scaler', None), ('clf', rf)])

models["std_nb"]    = pipe([('scaler', std), ('clf', nb)])
models["std_ridge"] = pipe([('scaler', std), ('clf', ridge)])
models["std_dt"]    = pipe([('scaler', std), ('clf', dt)])
models["std_knn"]   = pipe([('scaler', std), ('clf', knn)])
models["std_svc"]   = pipe([('scaler', std), ('clf', svc)])
models['std_lr']    = pipe([('scaler', std), ('clf', lr)])
models["std_rf"]    = pipe([('scaler', std), ('clf', rf)])

models["minmax_nb"]     = pipe([('scaler', minmax), ('clf', nb)])
models["minmax_ridge"]  = pipe([('scaler', minmax), ('clf', ridge)])
models["minmax_dt"]     = pipe([('scaler', minmax), ('clf', dt)])
models["minmax_knn"]    = pipe([('scaler', minmax), ('clf', knn)])
models["minmax_svc"]    = pipe([('scaler', minmax), ('clf', svc)])
models['minmax_lr']     = pipe([('scaler', minmax), ('clf', lr)])
models["minmax_rf"]     = pipe([('scaler', minmax), ('clf', rf)])

train_models(models, (x_train, y_train), (x_test, y_test), save_model=True, sort_values=True)
```

```python
models = {}

models["std_pca_nb"]    = pipe([('scaler', std), ('pca', pca), ('clf', nb)])
models["std_pca_ridge"] = pipe([('scaler', std), ('pca', pca), ('clf', ridge)])
models["std_pca_dt"]    = pipe([('scaler', std), ('pca', pca), ('clf', dt)])
models["std_pca_knn"]   = pipe([('scaler', std), ('pca', pca), ('clf', knn)])
models["std_pca_svc"]   = pipe([('scaler', std), ('pca', pca), ('clf', svc)])
models['std_pca_lr']    = pipe([('scaler', std), ('pca', pca), ('clf', lr)])
models["std_pca_rf"]    = pipe([('scaler', std), ('pca', pca), ('clf', rf)])

models["minmax_pca_nb"]     = pipe([('scaler', minmax), ('pca', pca), ('clf', nb)])
models["minmax_pca_ridge"]  = pipe([('scaler', minmax), ('pca', pca), ('clf', ridge)])
models["minmax_pca_dt"]     = pipe([('scaler', minmax), ('pca', pca), ('clf', dt)])
models["minmax_pca_knn"]    = pipe([('scaler', minmax), ('pca', pca), ('clf', knn)])
models["minmax_pca_svc"]    = pipe([('scaler', minmax), ('pca', pca), ('clf', svc)])
models['minmax_pca_lr']     = pipe([('scaler', minmax), ('pca', pca), ('clf', lr)])
models["minmax_pca_rf"]     = pipe([('scaler', minmax), ('pca', pca), ('clf', rf)])

train_models(models, (x_train, y_train), (x_test, y_test), save_model=True, sort_values=True)
```

```python
models = {}

models["std_smote_nb"]    = pipe([('scaler', std), ('sampler', smote), ('clf', nb)])
models["std_smote_ridge"] = pipe([('scaler', std), ('sampler', smote), ('clf', ridge)])
models["std_smote_dt"]    = pipe([('scaler', std), ('sampler', smote), ('clf', dt)])
models["std_smote_knn"]   = pipe([('scaler', std), ('sampler', smote), ('clf', knn)])
models["std_smote_svc"]   = pipe([('scaler', std), ('sampler', smote), ('clf', svc)])
models['std_smote_lr']    = pipe([('scaler', std), ('sampler', smote), ('clf', lr)])
models["std_smote_rf"]    = pipe([('scaler', std), ('sampler', smote), ('clf', rf)])

models["minmax_smote_nb"]     = pipe([('scaler', minmax), ('sampler', smote), ('clf', nb)])
models["minmax_smote_ridge"]  = pipe([('scaler', minmax), ('sampler', smote), ('clf', ridge)])
models["minmax_smote_dt"]     = pipe([('scaler', minmax), ('sampler', smote), ('clf', dt)])
models["minmax_smote_knn"]    = pipe([('scaler', minmax), ('sampler', smote), ('clf', knn)])
models["minmax_smote_svc"]    = pipe([('scaler', minmax), ('sampler', smote), ('clf', svc)])
models['minmax_smote_lr']     = pipe([('scaler', minmax), ('sampler', smote), ('clf', lr)])
models["minmax_smote_rf"]     = pipe([('scaler', minmax), ('sampler', smote), ('clf', rf)])

train_models(models, (x_train, y_train), (x_test, y_test), save_model=True, sort_values=True)
```

```python
models = {}

models["std_over_pca_nb"]    = pipe([('scaler', std), ('sampler', oversampler), ('pca', pca), ('clf', nb)])
models["std_over_pca_ridge"] = pipe([('scaler', std), ('sampler', oversampler), ('pca', pca), ('clf', ridge)])
models["std_over_pca_dt"]    = pipe([('scaler', std), ('sampler', oversampler), ('pca', pca), ('clf', dt)])
models["std_over_pca_knn"]   = pipe([('scaler', std), ('sampler', oversampler), ('pca', pca), ('clf', knn)])
models["std_over_pca_svc"]   = pipe([('scaler', std), ('sampler', oversampler), ('pca', pca), ('clf', svc)])
models['std_over_pca_lr']    = pipe([('scaler', std), ('sampler', oversampler), ('pca', pca), ('clf', lr)])
models["std_over_pca_rf"]    = pipe([('scaler', std), ('sampler', oversampler), ('pca', pca), ('clf', rf)])

models["minmax_over_pca_nb"]     = pipe([('scaler', minmax), ('sampler', oversampler), ('pca', pca), ('clf', nb)])
models["minmax_over_pca_ridge"]  = pipe([('scaler', minmax), ('sampler', oversampler), ('pca', pca), ('clf', ridge)])
models["minmax_over_pca_dt"]     = pipe([('scaler', minmax), ('sampler', oversampler), ('pca', pca), ('clf', dt)])
models["minmax_over_pca_knn"]    = pipe([('scaler', minmax), ('sampler', oversampler), ('pca', pca), ('clf', knn)])
models["minmax_over_pca_svc"]    = pipe([('scaler', minmax), ('sampler', oversampler), ('pca', pca), ('clf', svc)])
models['minmax_over_pca_lr']     = pipe([('scaler', minmax), ('sampler', oversampler), ('pca', pca), ('clf', lr)])
models["minmax_over_pca_rf"]     = pipe([('scaler', minmax), ('sampler', oversampler), ('pca', pca), ('clf', rf)])

train_models(models, (x_train, y_train), (x_test, y_test), save_model=True, sort_values=True)
```

### EXtraTreesClassifier

```python
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(random_state=21, n_jobs=-1)
et_bal = ExtraTreesClassifier(random_state=21, n_jobs=-1, class_weight='balanced')

models = {}

models['et']            = pipe([('scaler', None),   ('sampler', None), ('pca', None), ('clf', et)])
models['minmax_et']     = pipe([('scaler', minmax), ('sampler', None), ('pca', None), ('clf', et)])
models['std_et']        = pipe([('scaler', std),    ('sampler', None), ('pca', None), ('clf', et)])

models['et-bal']            = pipe([('scaler', None),   ('sampler', None), ('pca', None), ('clf', et_bal)])
models['minmax_et-bal']     = pipe([('scaler', minmax), ('sampler', None), ('pca', None), ('clf', et_bal)])
models['std_et-bal']        = pipe([('scaler', std),    ('sampler', None), ('pca', None), ('clf', et_bal)])

models['pca_et-bal']            = pipe([('scaler', None),   ('sampler', None), ('pca', pca), ('clf', et_bal)])
models['minmax_pca_et-bal']     = pipe([('scaler', minmax), ('sampler', None), ('pca', pca), ('clf', et_bal)])
models['std_pca_et-bal']        = pipe([('scaler', std),    ('sampler', None), ('pca', pca), ('clf', et_bal)])

train_models(models, (x_train, y_train), (x_test, y_test), save_model=True, sort_values=True)
```

```python
models = {}

models['smote_et']              = pipe([('scaler', None),   ('sampler', smote), ('pca', None), ('clf', et)])
models['minmax_smote_et']       = pipe([('scaler', minmax), ('sampler', smote), ('pca', None), ('clf', et)])
models['std_smote_et']          = pipe([('scaler', std),    ('sampler', smote), ('pca', None), ('clf', et)])
models['smote_pca_et']          = pipe([('scaler', None),   ('sampler', smote), ('pca', pca),  ('clf', et)])
models['minmax_smote_pca_et']   = pipe([('scaler', minmax), ('sampler', smote), ('pca', pca),  ('clf', et)])
models['std_smote_pca_et']      = pipe([('scaler', std),    ('sampler', smote), ('pca', pca),  ('clf', et)])

models['adasyn_et']             = pipe([('scaler', None),   ('sampler', adasyn), ('pca', None), ('clf', et)])
models['minmax_adasyn_et']      = pipe([('scaler', minmax), ('sampler', adasyn), ('pca', None), ('clf', et)])
models['std_adasyn_et']         = pipe([('scaler', std),    ('sampler', adasyn), ('pca', None), ('clf', et)])
models['adasyn_pca_et']         = pipe([('scaler', None),   ('sampler', adasyn), ('pca', pca),  ('clf', et)])
models['minmax_adasyn_pca_et']  = pipe([('scaler', minmax), ('sampler', adasyn), ('pca', pca),  ('clf', et)])
models['std_adasyn_pca_et']     = pipe([('scaler', std),    ('sampler', adasyn), ('pca', pca),  ('clf', et)])

models['over_et']               = pipe([('scaler', None),   ('sampler', oversampler), ('pca', None), ('clf', et)])
models['minmax_over_et']        = pipe([('scaler', minmax), ('sampler', oversampler), ('pca', None), ('clf', et)])
models['std_over_et']           = pipe([('scaler', std),    ('sampler', oversampler), ('pca', None), ('clf', et)])
models['over_pca_et']           = pipe([('scaler', None),   ('sampler', oversampler), ('pca', pca),  ('clf', et)])
models['minmax_over_pca_et']    = pipe([('scaler', minmax), ('sampler', oversampler), ('pca', pca),  ('clf', et)])
models['std_over_pca_et']       = pipe([('scaler', std),    ('sampler', oversampler), ('pca', pca),  ('clf', et)])

models['blsmote_et']            = pipe([('scaler', None),   ('sampler', blsmote), ('pca', None), ('clf', et)])
models['minmax_blsmote_et']     = pipe([('scaler', minmax), ('sampler', blsmote), ('pca', None), ('clf', et)])
models['std_blsmote_et']        = pipe([('scaler', std),    ('sampler', blsmote), ('pca', None), ('clf', et)])
models['blsmote_pca_et']        = pipe([('scaler', None),   ('sampler', blsmote), ('pca', pca),  ('clf', et)])
models['minmax_blsmote_pca_et'] = pipe([('scaler', minmax), ('sampler', blsmote), ('pca', pca),  ('clf', et)])
models['std_blsmote_pca_et']    = pipe([('scaler', std),    ('sampler', blsmote), ('pca', pca),  ('clf', et)])

models['svmsmote_et']              = pipe([('scaler', None),   ('sampler', svmsomote), ('pca', None), ('clf', et)])
models['minmax_svmsmote_et']       = pipe([('scaler', minmax), ('sampler', svmsomote), ('pca', None), ('clf', et)])
models['std_svmsmote_et']          = pipe([('scaler', std),    ('sampler', svmsomote), ('pca', None), ('clf', et)])
models['svmsmote_pca_et']          = pipe([('scaler', None),   ('sampler', svmsomote), ('pca', pca),  ('clf', et)])
models['minmax_svmsmote_pca_et']   = pipe([('scaler', minmax), ('sampler', svmsomote), ('pca', pca),  ('clf', et)])
models['std_svmsmote_pca_et']      = pipe([('scaler', std),    ('sampler', svmsomote), ('pca', pca),  ('clf', et)])

train_models(models, (x_train, y_train), (x_test, y_test), save_model=True, sort_values=True)
```