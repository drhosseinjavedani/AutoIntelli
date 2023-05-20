import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from sklearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector


urldata = "https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data"
urlname = "https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.names"
# column names
col_names = [
    "class",
    "lymphatics",
    "block of affere",
    "bl. of lymph. c",
    "bl. of lymph. s",
    "by pass",
    "extravasates",
    "regeneration of",
    "early uptake in",
    "lym.nodes dimin",
    "lym.nodes enlar",
    "changes in lym.",
    "defect in node",
    "changes in node",
    "special forms",
    "dislocation of",
    "exclusion of no",
    "no. of nodes in",

]

data = pd.read_csv(urldata,names=col_names)
data.head()

data.loc[(data["class"] == 1) | (data["class"] == 2), "class"] = 0
data.loc[data["class"] == 3, "class"] = 1
data.loc[data["class"] == 4, "class"] = 2
data["class"] = data["class"].astype(int)

X = data.loc[:, data.columns != "class"]
y = data.loc[:, data.columns == "class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33,  random_state=42
)

shap_feature_selector_factory = (
    ShapFeatureSelector.shap_feature_selector_factory.set_model_params(
        X=X_train,
        y=y_train,
        verbose=0,
        random_state=0,
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params = {
            "sample_weight": None,
        },
        method="gridsearch",
        # if n_features=None only the threshold will be considered as a cut-off of features grades.
        # if threshold=None only n_features will be considered to select the top n features.
        # if both of them are set to some values, the threshold has the priority for selecting features.
        n_features=3,
        threshold = 0.4,
        list_of_obligatory_features_that_must_be_in_model=["defect in node"],
        list_of_features_to_drop_before_any_selection=["bl. of lymph. c"],
    )
    .set_shap_params(
        model_output="raw",
        feature_perturbation="interventional",
        algorithm="v2",
        shap_n_jobs=-1,
        memory_tolerance=-1,
        feature_names=None,
        approximate=False,
        shortcut=False,
    )
    .set_gridsearchcv_params(
        measure_of_accuracy=make_scorer(f1_score, greater_is_better=True, average='macro'),
        verbose=10,
        n_jobs=-1,
        cv=KFold(2),
    )
)

int_cols = X_train.select_dtypes(include=["int"]).columns.tolist()
print(int_cols)


pipeline = Pipeline(
    [
        # int missing values imputers
        (
            "intimputer",
            MeanMedianImputer(imputation_method="median", variables=int_cols),
        ),
        ("sfsf", shap_feature_selector_factory),
        # classification model
        ("logistic", LogisticRegression()),
    ]
)

pipeline.fit(X_train, y_train.values.ravel())
y_pred = pipeline.predict(X_test)


print("F1 score : ")
print(f1_score(y_test, y_pred,average='micro'))
print("Classification report : ")
print(classification_report(y_test, y_pred))
print("Confusion matrix : ")
print(confusion_matrix(y_test, y_pred))


print(ShapFeatureSelector.shap_feature_selector_factory.get_feature_selector_instance())
