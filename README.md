# House price prediction with predictive ML

## Project architecture

This project uses scikit-learn, XGBoost and Kubeflow pipelines to build a training pipeline that is registered to Vertex AI.
The idea is that this pipeline will run based on fresh data that is pointed to a GCP Storage bucket, and if the fresh data 
presents a higher RMSE value than the previous one, it will automatically register a new version for the model as an API on Vertex AI.

![Kubeflow Training Pipeline](./media/vertex_pipeline.png)

All of this is managed by GCP, with the added flexibility of being a Kubeflow pipeline, so apart from the model registry logic that currently uses `google.cloud.aiplatform`, it can be deployed to a cloud-agnostic Kubernetes cluster with a little effort.

## Machine Learning development
This model aims to predict the house prices in US dollars, and the exploration to find its first version is located at `analysis/eda.ipynb`
I have downloaded the `train.csv` file locally and placed it under `{PROJECT_ROOT}/data` so that we don't need to use the Google Drive mount point.

For a production use-case, this would likely be placed on a cloud storage solution, which is what is used for the Kubeflow training pipeline.
## Accessing the price prediction API endpoint

An example request to this endpoint is defined by the following steps: 

```bash
gcloud auth application-default login # to get your gcp credentials
```

And create these environment variables:
```
ENDPOINT_ID="3187414939690074112"
PROJECT_ID="954208583758"
INPUT_DATA_FILE="./data/input.json" 
```

With both the GCP access and the variables defined, you can make a request using curl, as the example shows: 

```bash
curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
"https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict" \
-d "@${INPUT_DATA_FILE}"
```

In the future, the idea is to make it publicly available as a more generic REST endpoint, which is out of the scope of this test case.

## Local development
In order to run this project on your machine you will need a Kubeflow pipelines component running locally.
We can do that by deploying it using Minikube, which emulates a real-world Kubernetes environment on the local machine,
and then use docker as the underlying engine to spin up the cluster.

### Deploying KF Pipelines with Minikube + docker

Local deployment of kubeflow following: https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/
Then port forwarding to 8080 we get

`kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80` 

>**NOTE**: I had an issue with Kubeflow's proxy agent since it was trying to resolve for a GKE deployment
>(probably for using Minikube), so I disabled it by scaling its pods to 0 replicas `kubectl scale deployment proxy-agent -n kubeflow --replicas=0`  

## Compile and submit
By running the following command
```bash
PYTHONPATH="$PYTHONPATH:$PWD" python3 src/run.py
```
the code will generate a local `pipeline.json` file that will also be submitted as a Vertex AI pipeline to your existing GCP account.
You can also import `src/run.py`'s `compile_pipeline` function in order to generate the compiled pipeline locally and then 
create your own Vertex AI pipeline manually through the console.

## Simplifications 
- I have decided to put the data under a GCS bucket, that will be the same bucket that is used for the pipeline to persist its artifacts
- For now, the REST API will be automatically deployed in Vertex -- instead of Terraform -- so for every new model we also get a new endpoint
- The pipeline is currently not active and listening to GCS - which would be a nice addition
- The model is not monitored 
- We don't have the Feature Weights to decide what parts of the raw data are the most important for inferencing

## Challenges faced
- As this was my first project with Vertex AI and Kubeflow, I had a lot to learn in a short period of time
- Authentication was especially hard, given that even with a Service Account, Vertex has its own ways to authenticate (through a Service Agent) with little to no useful error logs, so ramping up took most of the project time.
- Wish there was a couple more days to improve Monitoring and explainability, which are core ideas for a robust MLOps pipeline
- Found it hard to modularize Kubeflow components code, as each pipeline step is its own self-contained container execution, which needs to have all the logic
needed to run the code without imports. Could probably be solved by decoupling the model code into a python package, but it would add extra complexity out of the scope of this project case.
- Although I managed to deploy the Kubeflow Pipelines as a standalone app locally, the UI is pretty unstable and I could not get it 
to a seamless local development experience. Given a little more time and effort, that would **dramatically** speed up development without having to send anything to GCP at all.