"""Create a reviewable Terraform configuration for one rightsized AWS VM."""

from __future__ import annotations

import re
from typing import Any

SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")
SAFE_AMI = re.compile(r"^ami-[A-Za-z0-9]+$")
SAFE_SUBNET = re.compile(r"^subnet-[A-Za-z0-9]+$")


def generate_vm_terraform(
    resource_id: str,
    plan: str,
    pricing: list[dict[str, Any]],
    region: str,
    ami_id: str,
    subnet_id: str,
) -> dict[str, str]:
    if not SAFE_TOKEN.fullmatch(resource_id):
        raise ValueError("resource_id may only contain letters, digits, '.', '_' or '-'.")
    if not SAFE_TOKEN.fullmatch(plan):
        raise ValueError("plan may only contain letters, digits, '.', '_' or '-'.")
    if not SAFE_AMI.fullmatch(ami_id):
        raise ValueError("ami_id must have the AWS AMI identifier form ami-...")
    if not SAFE_SUBNET.fullmatch(subnet_id):
        raise ValueError("subnet_id must have the AWS subnet identifier form subnet-...")
    if not SAFE_TOKEN.fullmatch(region):
        raise ValueError("region may only contain letters, digits, '.', '_' or '-'.")

    catalog_entry = next(
        (item for item in pricing if item.get("plan") == plan and item.get("type") == "VM"),
        None,
    )
    if catalog_entry is None:
        raise ValueError(f"VM plan {plan!r} was not found in the pricing catalog.")
    resource_name = resource_id.replace("-", "_").replace(".", "_")
    specs = (
        f"# Plan: {plan}\n"
        f"# Capacity from catalog: {int(catalog_entry['vcpu'])} vCPU, "
        f"{int(catalog_entry['ram_gb'])} GiB RAM\n"
    )
    config = f'''{specs}terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = var.aws_region
}}

variable "aws_region" {{
  type        = string
  description = "AWS region for the replacement instance."
  default     = "{region}"
}}

variable "ami_id" {{
  type        = string
  description = "Approved AMI for this workload and account."
  default     = "{ami_id}"
}}

variable "subnet_id" {{
  type        = string
  description = "Target subnet in the selected AWS region."
  default     = "{subnet_id}"
}}

resource "aws_instance" "rightsized_{resource_name}" {{
  ami           = var.ami_id
  instance_type = "{plan}"
  subnet_id     = var.subnet_id

  tags = {{
    Name       = "{resource_id}"
    ManagedBy  = "Terraform"
    Rightsized = "true"
  }}
}}

output "instance_id" {{
  value       = aws_instance.rightsized_{resource_name}.id
  description = "ID of the rightsized EC2 instance."
}}
'''
    return {
        "resource_id": resource_id,
        "plan": plan,
        "region": region,
        "terraform": config,
        "warning": (
            "Review AMI, subnet, security groups, storage, IAM, user data, and "
            "state/import strategy before applying. This configuration creates a "
            "new instance; it does not migrate or destroy the existing VM."
        ),
    }
