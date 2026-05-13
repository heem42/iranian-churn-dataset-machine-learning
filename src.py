from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, make_scorer
)
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# id=563 is our iranian churn dataset
dataset = fetch_ucirepo(id=563)
features = dataset.data.features
churn_labels = dataset.data.targets.values.ravel()


# PRE PROCESSING
# Features and labels are already extracted from the dataset
print(f'Data shape: {features.shape}')

# 2) Holdout preprocessing: 80/20 train/test split
train_features, test_features, train_labels, test_labels = train_test_split(
    features, churn_labels, test_size=0.2, random_state=42, stratify=churn_labels #random_state=42 is so that we get same results every time 
)


# 3)Oversampling for KNN model which increased our recall by almsot 6 percent.
#random_state=42 is so that we get same results every time 
smote_oversampler = SMOTE(random_state=42)
smote_train_features, smote_train_labels = smote_oversampler.fit_resample(train_features, train_labels)

# 4)Feature scaling
feature_scaler = StandardScaler()
scaled_train_features = feature_scaler.fit_transform(train_features)
scaled_test_features = feature_scaler.transform(test_features)
scaled_smote_train_features = feature_scaler.transform(smote_train_features)


# Model Training and Evaluation
def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]

    return {
        'Precision': precision_score(y_te, y_pred),
        'Recall': recall_score(y_te, y_pred),
        'F1-Score': f1_score(y_te, y_pred),
        'ROC-AUC': roc_auc_score(y_te, y_pred_proba)
    }

# Hyperparameter Tuning with
# We are using GridSearch algorithm. This will allow us to find best parameters for each model from our list.
model_params = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {'C': [0.01, 0.1, 1, 10]}
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [3, 5, 7, 9]}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth': [5, 10, 15, None]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
    }
}

tuned_models = {}
for model_name, config in model_params.items():
    # cv=10 means 10 fold cross validation is used to evaluate each parameter in grid search
    # scoring=f1 score so it uses both recall and precision to evaluate the model.
    # n_jobs=-1 will make the algorithm faster by using all CPU available.
    grid_search = GridSearchCV(config['model'], config['params'], cv=10, scoring='f1', n_jobs=-1)
    if model_name == 'K-Nearest Neighbors':
        grid_search.fit(scaled_smote_train_features, smote_train_labels)
    else:
        grid_search.fit(scaled_train_features, train_labels)
    # select the best model from grid search for evaluation
    tuned_models[model_name] = grid_search.best_estimator_
    print(f'{model_name}: {grid_search.best_params_}')

model_results = {}
for model_name, model in tuned_models.items():
    # We use this condition because KNN is using Oversampling as it gave us better recall.
    if model_name == 'K-Nearest Neighbors':
        model_results[model_name] = evaluate_model(
            model, scaled_smote_train_features, scaled_test_features, smote_train_labels, test_labels
        )
    # Other than KNN, we use normal training and test data
    else:
        model_results[model_name] = evaluate_model(
            model, scaled_train_features, scaled_test_features, train_labels, test_labels
        )

print('\n')
print('Evaluation Method 1) HOLDOUT METHOD (80/20 SPLIT)')
results_table = pd.DataFrame(model_results).T.round(4)
print(results_table.to_string())

print('\n')
print('Evaluation Method 2) 10-FOLD CROSS-VALIDATION (Mean)')

# n_splits=10 is used to divide the data into 10 folds
# shuffle=True is used to shuffle the data before splitting into fols for randomness
# random_state=42 is used to get same results every time we run the code.
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_results = {}

for model_name, model in tuned_models.items():
    if model_name == 'K-Nearest Neighbors':
        X_cv = scaled_smote_train_features
        y_cv = smote_train_labels
    else:
        X_cv = scaled_train_features
        y_cv = train_labels
    
    precision_scores = cross_val_score(model, X_cv, y_cv, cv=kfold, 
                                       scoring=make_scorer(precision_score, zero_division=0))
    recall_scores = cross_val_score(model, X_cv, y_cv, cv=kfold, 
                                    scoring=make_scorer(recall_score, zero_division=0))
    f1_scores = cross_val_score(model, X_cv, y_cv, cv=kfold, 
                                scoring=make_scorer(f1_score, zero_division=0))
    roc_auc_scores = cross_val_score(model, X_cv, y_cv, cv=kfold, scoring='roc_auc')
    
    # since we get 10 scores for each metric (one for each fold), we take the mean to get an overall estimate of performance.
    # this will give us 1 score so we can better compare with other models
    cv_results[model_name] = {
        'Precision': f'{precision_scores.mean():.4f}',
        'Recall': f'{recall_scores.mean():.4f}',
        'F1-Score': f'{f1_scores.mean():.4f}',
        'ROC-AUC': f'{roc_auc_scores.mean():.4f}'
    }

# Transpose the cv_results array for better visualization
cv_results_table = pd.DataFrame(cv_results).T
print(cv_results_table.to_string())

print("\n")
print('Evaluation Method 3) RANDOM SUBSAMPLING')
subsample_results = {}
for model_name, model in tuned_models.items():
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []
    for i in range(10):
        # random_state=42+i is used to get different random splits for each iteration, creating random subsamples
        # stratify=churn_labels is used to maintain the same class distribution in both train and test sets for accurate results
        X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(
            features, churn_labels, test_size=0.2, random_state=42+i, stratify=churn_labels
        )
        # scale the features of subsamples
        scaler_sub = StandardScaler()
        X_sub_train_scaled = scaler_sub.fit_transform(X_sub_train)
        X_sub_test_scaled = scaler_sub.transform(X_sub_test)
        if model_name == 'K-Nearest Neighbors':
            smote_sub = SMOTE(random_state=42)
            X_sub_train_scaled, y_sub_train = smote_sub.fit_resample(X_sub_train_scaled, y_sub_train)
        model.fit(X_sub_train_scaled, y_sub_train)
        y_sub_pred = model.predict(X_sub_test_scaled)
        y_sub_pred_proba = model.predict_proba(X_sub_test_scaled)[:, 1]
        precision_scores.append(precision_score(y_sub_test, y_sub_pred))
        recall_scores.append(recall_score(y_sub_test, y_sub_pred))
        f1_scores.append(f1_score(y_sub_test, y_sub_pred))
        roc_auc_scores.append(roc_auc_score(y_sub_test, y_sub_pred_proba))

    # calculate mean of scores so we have one value for each metric, easier for us to compare just like in CV mehtod
    subsample_results[model_name] = {
        'Precision': f'{pd.Series(precision_scores).mean():.4f}',
        'Recall': f'{pd.Series(recall_scores).mean():.4f}',
        'F1-Score': f'{pd.Series(f1_scores).mean():.4f}',
        'ROC-AUC': f'{pd.Series(roc_auc_scores).mean():.4f}'
    }

# Transpose the subsample_table array for better visualization
subsample_table = pd.DataFrame(subsample_results).T
print(subsample_table.to_string())

print("\n")
print('Feature Importance on Churn using Random Forest')
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(scaled_train_features, train_labels)
feature_scores = pd.Series(random_forest.feature_importances_, index=features.columns)
feature_scores = feature_scores.sort_values(ascending=False)
feature_percent = (feature_scores * 100).round(2)
feature_percent = feature_percent / feature_percent.sum() * 100
for i, (feature, percent) in enumerate(feature_percent.head(10).items(), 1):
    print(f'{i}. {feature}: {percent:.1f}%')
