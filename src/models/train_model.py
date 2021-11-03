import json
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report 

def read_params(config_path):
    """
    read parameters from params.yalml file
    input: params.yaml location
    output: paramters as dictionary
    """

    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config 

def accuracymeasures(y_test, predictions, avg_method):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average = avg_method)
    target_names = ["0", "1"]
    print("Classification report")
    print("---------------------" , "\n")
    print(classification_report(y_test, predictions, target_names = target_names), "\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(y_test, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy, precision, recall, f1score

def get_feat_and_target(df, target):
    """
    Get features and target variables separately from given df and target
    input: data frame and target column
    output: two df for x and y
    """
    x = df.drop(target, axis = 1)
    y = df[target]
    return x, y

"""
processed_data_config:
  train_data_csv: data/processed/churn_train.csv
  test_data_csv: data/processed/churn_test.csv
"""

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    max_depth=config["random_forest"]["max_depth"]
    n_estimators=config["random_forest"]["n_estimators"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target) 

################# MLFLOW #########################
    """
    mlflow_config:
    artifacts_dir: artifacts
    experiment_name: model_iteration1
    run_name: random_forest
    registered_model_name: random_forest_model
    remote_server_url: http://localhost:5000
    """

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"] 
    #connects to a tracking URI
    mlflow.set_tracking_url = mlflow_config["remote_server_uri"] 
    #creates a new experiment and returns its ID
    mlflow.set_experiment(mlflow_config["experiment_name"])
    #Runs can be launched under the experiment by passing the experiment ID to mlflow.start_run
    
    with mlflow.start_run(run_name = mlflow_config["run_name"]) as mlops_run:
        model = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators)
        model.fit(train_x, train_y)
        joblib.dump(model, config["model_dir"])
        y_pred = model.predict(test_x)
        accuracy, precision, recall, f1score = accuracymeasures(test_y, y_pred, 'weighted')

        #logs a single key-value param in the currently active run. The key and value are both strings
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            #breakpoint()
            mlflow.sklearn.log_model(model,"models")
 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
