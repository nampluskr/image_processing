## Classification

```python
import pandas as pd

# Data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Models
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Training
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Metrics
from sklearn.metrics import make_scorer, accuracy_score

# import warnings
# warnings.filterwarnings('ignore')
```

### [Level-1]

```python
cancer = load_breast_cancer()
x, y = cancer.data, cancer.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

svc = SVC()
svc.fit(x_train, y_train)

print(f"Train acc.: {svc.score(x_train, y_train):.4f}")
print(f"Test  acc.: {svc.score(x_test, y_test):.4f}")
```

### [Level-2]

```python
cancer = load_breast_cancer()
x, y = cancer.data, cancer.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

scaler = MinMaxScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

svc = SVC()
svc.fit(x_train_scaled, y_train)

print(f"Train acc.: {svc.score(x_train_scaled, y_train):.4f}")
print(f"Test  acc.: {svc.score(x_test_scaled, y_test):.4f}")
```

### [Level-3] GridSearchCV

```python
cancer = load_breast_cancer()
x, y = cancer.data, cancer.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

scaler = MinMaxScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

svc = SVC()
params = {"C": [0.1, 1, 10],
         "gamma": [0.1, 1, 10]}
trainer = GridSearchCV(svc, param_grid=params, cv=5, 
                    n_jobs=1, verbose=1)
trainer.fit(x_train_scaled, y_train)

print(f"Train acc.: {trainer.best_score_:.4f}")
print(f"Test  acc.: {trainer.best_estimator_.score(x_test_scaled, y_test):.4f}")
print(trainer.best_params_)
```

### [Level-4] Pipeline + GridSearchCV

```python
cancer = load_breast_cancer()
x, y = cancer.data, cancer.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

steps = Pipeline([("scalar", MinMaxScaler()),
                  ("svm", SVC())])
params = {"svm__C": [0.1, 1, 10],
         "svm__gamma": [0.1, 1, 10]}
trainer = GridSearchCV(steps, param_grid=params, cv=5, 
                    n_jobs=1, verbose=1)
trainer.fit(x_train, y_train)

print(f"Train acc.: {trainer.best_score_:.4f}")
print(f"Test  acc.: {trainer.best_estimator_.score(x_test, y_test):.4f}")
print(trainer.best_params_)
```

### [Level-5-1] Pipeline + GridSearchCV: Training Models

```python
cancer = load_breast_cancer()
x, y = cancer.data, cancer.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

steps = Pipeline([("pre", None),
                  ("clf", DummyClassifier())])
params_svc = {"pre": [StandardScaler()],
              "clf": [SVC()],
              "clf__C": [0.1, 1, 10],
              "clf__gamma": [0.1, 1, 10]
}
params_rfc = {"pre": [None],
              "clf": [RandomForestClassifier(n_estimators=100)],
              "clf__max_features": [1, 2, 3]
}
params_mlp = {"pre": [StandardScaler()],
              "clf":[MLPClassifier(max_iter=10000, random_state=42)],
              "clf__solver": ["lbfgs", "adam"],
              "clf__hidden_layer_sizes": [(10, 10), (100, 100)],
              "clf__activation": ["relu", "tanh"]
}
trainer = GridSearchCV(estimator=steps,
                       param_grid=[params_svc, params_rfc, params_mlp],
                       cv=5, n_jobs=1, verbose=1)
trainer.fit(x_train, y_train)

print(f"Train acc.: {trainer.best_score_:.4f}")
print(f"Test  acc.: {trainer.best_estimator_.score(x_test, y_test):.4f}")
print(trainer.best_params_)
```

### [Level-5-2] Pipeline + GridSearchCV: Training Models

```python
cancer = load_breast_cancer()
x, y = cancer.data, cancer.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

step = Pipeline([("pre", None), ("clf", DummyClassifier())])

models = {}
models["svc"] = {"pre": [MinMaxScaler()],
                 "clf": [SVC()],
                 "clf__C": [0.1, 1, 10],
                 "clf__gamma": [0.1, 1, 10],
}
models["rfc"] = {"pre": [None],
                 "clf":[RandomForestClassifier()],
                 "clf__n_estimators": [100],
                 "clf__max_features": [1, 2, 3],
}
models["mlp"] = {"pre": [StandardScaler()],
                 "clf": [MLPClassifier(max_iter=10000, random_state=42)],
                 "clf__solver": ["lbfgs", "adam"],
                 "clf__hidden_layer_sizes": [(10, 10), (100, 100)],
                 "clf__activation": ["relu", "tanh"]
}

for model, params in models.items():
    print(f"\n>> {model}:")
    trainer = GridSearchCV(steps, param_grid=params,
                           cv=5, n_jobs=1, verbose=1)
    trainer.fit(x_train, y_train)

    print(f"Train acc.: {trainer.best_score_:.4f}")
    print(f"Test  acc.: {trainer.best_estimator_.score(x_test, y_test):.4f}")
    print(trainer.best_params_)
```

### [Level-5-3] Pipeline + GridSearchCV: Training Models

```python
## Data
cancer = load_breast_cancer()
x, y = cancer.data, cancer.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)
```

```python
## Models
models = {}
models["svc"] = {
    "name": "Supprot Vector Machine",
    "steps": Pipeline([("pre", None),
                       ("clf", SVC())]),
    "params": {"pre": [MinMaxScaler(), StandardScaler()],
               "clf__C": [0.1, 1, 10],
               "clf__gamma": [0.1, 1, 10],}
}
models["rfc"] = {
    "name": "Random Forest",
    "steps": Pipeline([("pre", None),
                       ("clf", RandomForestClassifier())]),
    "params": {"pre": [None],
               "clf__n_estimators": [100],
               "clf__max_features": [1, 2, 3],}
}
models["mlp"] = {
    "name": "Multilayer Perception",
    "steps": Pipeline([("pre", None), 
                       ("clf", MLPClassifier(max_iter=10000, random_state=42))]),
    "params": {"pre": [StandardScaler()],
               "clf__solver": ["lbfgs", "adam"],
               "clf__hidden_layer_sizes": [(10, 10), (100, 100)],
               "clf__activation": ["relu", "tanh"],}
}
```

```python
# https://scikit-learn.org/stable/modules/model_evaluation.html

def train_model(model, x_train, y_train, score_fn, cv=5, verbose=1):
    print(f"\n>> {model['name']}:")
    trainer = GridSearchCV(estimator=model["steps"],
                        param_grid=model["params"],
                        scoring=make_scorer(score_fn),  # score_fn(y_ture, y_pred)
                        cv=cv, n_jobs=1, verbose=1)
    trainer.fit(x_train, y_train)

    results = pd.DataFrame(trainer.cv_results_)
    results = results.sort_values("rank_test_score", ascending=True)
    results = results.reset_index(drop=True)

    columns = ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    results = results[columns].head(5)
    results.insert(0, "name", model["name"])
    results.iloc[:, 2:4] = results.iloc[:, 2:4].apply(lambda x: round(x, 4))
    if verbose:
        display(results)
    return trainer.best_score_, trainer.best_estimator_

score, estimator = train_model(models["svc"], x_train, y_train, score_fn=accuracy_score)
```

```python
for model in models.values():
    best_score, best_estimator = 0, None
    score, estimator = train_model(model, x_train, y_train, score_fn=accuracy_score)
    if (score > best_score):
        best_score, best_estimator = score, estimator
        
estimator
```
