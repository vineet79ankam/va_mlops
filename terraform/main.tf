terraform {
  required_providers {
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
}

provider "local" {}

resource "local_file" "hello_tf" {
  content  = "Terraform is working!"
  filename = "${path.module}/hello.txt"
}
