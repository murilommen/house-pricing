provider "google" {
  project = var.project_id
  region  = var.region
}

# Create the endpoint
resource "google_vertex_ai_endpoint" "prediction_endpoint" {
  display_name = "${var.model_name}-endpoint"
  location     = var.region
  project      = var.project_id
}

# Get the latest model version
data "google_vertex_ai_models" "existing_model" {
  display_name = var.model_name
  project      = var.project_id
  region       = var.region
}

# Deploy model to endpoint
resource "google_vertex_ai_model_deployment" "model_deployment" {
  endpoint = google_vertex_ai_endpoint.prediction_endpoint.id
  display_name = "deployment-${var.model_name}"

  deployed_model {
    model = data.google_vertex_ai_models.existing_model.models[0].id

    dedicated_resources {
      machine_spec {
        machine_type = var.machine_type
      }
      min_replica_count = var.min_replicas
      max_replica_count = var.max_replicas
    }
  }
}