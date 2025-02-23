variable "project_id" {
  description = "The ID of the project where resources will be created"
  type        = string
}

variable "region" {
  description = "The region where resources will be created"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "The name of the Cloud Run service"
  type        = string
  default     = "fastapi-service"
}

variable "container_image" {
  description = "The container image to deploy"
  type        = string
}

variable "model_version" {
  description = "The version of the model to use"
  type        = string
}

variable "model_id" {
  description = "The ID of the model to use"
  type        = string
}

variable "container_port" {
  description = "The port that will be exposed on Cloud Run"
  type        = string
}