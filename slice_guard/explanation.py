import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


def explain_clusters(features, feature_types, issue_df, df, prereduced_embeddings):
    """
    Method to generate explanations on why clusters are problematic or different from the rest of the data.

    :param features: The features to use for generating the explanations.
    :param feature_types: The previously inferred type of the features.
    :param issue_df: The dataframe containing the issues.
    :param: df: The dataframe containing the raw data.
    :param: prereduced_embeddings: Prereduced embeddings that can be used as input feature in case of unstructured data.

    """
    # Encode the data and keep track of conversions to keep interpretable
    feature_groups = []  # list of indices for grouped features
    current_feature_index = 0
    classification_data = np.zeros((len(df), 0))
    label_encoders = {}
    for col in features:
        feature_type = feature_types[col]
        if feature_type == "numerical":
            classification_data = np.concatenate(
                (classification_data, df[col].values.reshape(-1, 1)), axis=1
            )
            feature_groups.append([current_feature_index])
            current_feature_index += 1
        elif feature_type == "nominal" or feature_type == "ordinal":
            label_encoder = LabelEncoder()
            integer_encoded_data = label_encoder.fit_transform(
                df[col].values
            ).reshape(-1, 1)
            label_encoders[col] = label_encoder
            classification_data = np.concatenate(
                (classification_data, integer_encoded_data), axis=1
            )
            feature_groups.append([current_feature_index])
            current_feature_index += 1
        elif feature_type == "raw":
            reduced_embeddings = prereduced_embeddings[col]
            classification_data = np.concatenate(
                (classification_data, reduced_embeddings), axis=1
            )

            feature_groups.append(
                list(
                    range(
                        current_feature_index,
                        current_feature_index + reduced_embeddings.shape[1],
                    )
                )
            )
            current_feature_index += reduced_embeddings.shape[1]
        else:
            raise RuntimeError(
                "Met unexpected feature type while generating explanations."
            )

    # Fit tree to generate feature importances
    # TODO: Potentially replace with simpler univariate mechanism, see also spotlight relevance score
    # TODO: Probably try shap or something similar
    issue_df["issue_explanation"] = ""

    for issue in issue_df["issue"].unique():
        if issue == -1:  # Skip data points with no issues
            continue
        issue_indices_pandas = issue_df[issue_df["issue"] == issue].index
        issue_indices_list = np.where(issue_df["issue"] == issue)[0]
        y = np.zeros(len(issue_df))
        y[issue_indices_list] = 1
        clf = DecisionTreeClassifier(
            max_depth=3, max_features=4
        )  # keep the trees simple to not overfit
        clf.fit(classification_data, y)

        preds = clf.predict(classification_data)
        f1 = f1_score(y, preds)

        importances = clf.feature_importances_

        # aggregate importances of grouped features
        agg_importances = []
        for feature_group in feature_groups:
            if len(feature_group) > 1:
                agg_importances.append(importances[feature_group].sum())
            else:
                agg_importances.append(importances[feature_group[0]])
        importances = np.array(agg_importances)

        feature_order = np.argsort(importances)[::-1]
        ordered_importances = importances[feature_order]
        ordered_features = np.array(features)[feature_order]

        # if f1 > 0.7: # only add explanation if it is succicient to classify cluster?
        importance_strings = []
        for f, i in zip(ordered_features[:3], ordered_importances[:3]):
            importance_strings.append(f"{f}, ({i:.2f})")
        issue_df.loc[issue_indices_pandas, "issue_explanation"] = ", ".join(
            importance_strings
        )
    return issue_df