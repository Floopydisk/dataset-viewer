#!/bin/bash
# EC2 User Data Script - Run on instance launch
# This installs Docker, pulls the repo, and starts the full backend

set -e

# Update system
apt-get update
apt-get install -y git curl

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone the repo
cd /home/ubuntu
git clone https://github.com/huggingface/dataset-viewer.git
cd dataset-viewer

# Create directories for persistent storage
mkdir -p /mnt/data/{storage,parquet_metadata,duckdb_index,stats_cache,datasets_cache,modules_cache,numba_cache}
chmod -R 777 /mnt/data

# Copy and configure env file from template (populate before launching)
# NOTE: You MUST edit .env.production with your actual values before launching
cp .env .env.production

# Start the backend with docker-compose
# Note: Change log-driver to awslogs if you want CloudWatch logs
docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d

# Print status
echo "Backend services starting up..."
docker-compose ps
