# House price prediction with predictive ML

## Project architecture

## Machine Learning development

## Accessing the price prediction API endpoint

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

## Next steps