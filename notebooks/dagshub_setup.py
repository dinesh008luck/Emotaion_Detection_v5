import mlflow
import dagshub
mlflow.set_tracking_uri('https://dagshub.com/dinesh008luck/Emotaion_Detection_v5.mlflow')

dagshub.init(repo_owner='dinesh008luck', repo_name='Emotaion_Detection_v5', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)