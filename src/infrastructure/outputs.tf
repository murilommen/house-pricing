output "endpoint_id" {
  value = google_vertex_ai_endpoint.prediction_endpoint.id
  description = "The ID of the created endpoint"
}

output "endpoint_url" {
  value = google_vertex_ai_endpoint.prediction_endpoint.name
  description = "The URL of the endpoint"
}