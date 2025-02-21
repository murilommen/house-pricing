variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "Region for Vertex AI endpoint"
  type        = string
  default     = "us-central1"
}

variable "model_name" {
  description = "Name of the model in Vertex AI Model Registry"
  type        = string
}

variable "machine_type" {
  description = "Machine type for model deployment"
  type        = string
  default     = "n1-standard-4"
}

variable "min_replicas" {
  description = "Minimum number of replicas"
  type        = number
  default     = 1
}

variable "max_replicas" {
  description = "Maximum number of replicas"
  type        = number
  default     = 2
}