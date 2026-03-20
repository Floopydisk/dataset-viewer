# EC2 Deployment Checklist

Use this checklist to deploy the backend on AWS EC2 with Supabase S3 + MongoDB Atlas.

## Pre-Launch Checklist

### Accounts & Setup (Do these BEFORE launching EC2)

- [ ] **AWS Account** ready and logged in
- [ ] **MongoDB Atlas** account created, cluster running
  - [ ] Database user created
  - [ ] Connection string copied
  - [ ] IP allowlist includes `0.0.0.0/0` (or your EC2 IP after launch)
- [ ] **Supabase Account** created, S3 credentials enabled
  - [ ] S3 access key + secret saved
  - [ ] Project reference noted
  - [ ] S3 bucket created (e.g., `dataset-viewer`)
- [ ] **EC2 Key Pair** downloaded and saved locally
- [ ] `.env.production` file prepared locally
- [ ] `aws-ec2-launch.sh` permissions set: `chmod +x aws-ec2-launch.sh`

## Launch Checklist

### AWS EC2 Launch

- [ ] **Instance Type**: t3.small or t2.small
- [ ] **OS**: Ubuntu 22.04 LTS
- [ ] **Storage**: 30GB GP2
- [ ] **Security Group** configured:
  - [ ] SSH (22) from your IP
  - [ ] HTTP (80) from 0.0.0.0/0
  - [ ] Outbound all traffic allowed
- [ ] **User Data**: `aws-ec2-launch.sh` pasted in Advanced Details
- [ ] **Key Pair**: Selected and downloaded
- [ ] **Instance Launched** - note public IP

### Wait for Instance
- [ ] Instance running (check AWS console)
- [ ] Status checks passed (2/2)
- [ ] Wait 3-5 minutes for Docker/git to finish setup

## Configuration Checklist

### Fill in `.env.production`

MongoDB section:
- [ ] `CACHE_MONGO_URL` = full Atlas connection string
- [ ] `QUEUE_MONGO_URL` = same as above

Supabase S3 section:
- [ ] `S3_ACCESS_KEY_ID` = from Supabase Storage
- [ ] `S3_SECRET_ACCESS_KEY` = from Supabase Storage
- [ ] `S3_ENDPOINT_URL` = https://PROJECT_REF.storage.supabase.co/storage/v1/s3
- [ ] `ASSETS_STORAGE_ROOT` = bucket-name/assets
- [ ] `CACHED_ASSETS_STORAGE_ROOT` = bucket-name/cached-assets

Public URLs:
- [ ] `ASSETS_BASE_URL` = http://EC2_PUBLIC_IP/assets
- [ ] `CACHED_ASSETS_BASE_URL` = http://EC2_PUBLIC_IP/cached-assets

Optional auth:
- [ ] `COMMON_HF_TOKEN` = (leave empty if not using gated datasets)
- [ ] `COMMITTER_HF_TOKEN` = (leave empty if not pushing parquet to Hub)

## Deployment Checklist

### SSH & Upload

- [ ] SSH into instance: `ssh -i key.pem ubuntu@EC2_IP`
- [ ] Verify you're in: `~/dataset-viewer/`
- [ ] Upload .env.production: `scp -i key.pem .env.production ubuntu@EC2_IP:~/dataset-viewer/`
- [ ] Verify file uploaded: `ls -la .env.production`

### Start Services

- [ ] Start compose: `docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d`
- [ ] Wait 2-3 minutes
- [ ] Check status: `docker-compose ps`
- [ ] All services show "Up" status (green)

### Validate Deployment

- [ ] Run validation script: `bash validate-deployment.sh`
- [ ] Check API responds: `curl http://localhost/healthcheck`
- [ ] Check worker logs: `docker-compose logs worker | head -20`
- [ ] No errors in logs

## Post-Launch Checklist

### Testing

- [ ] Test /splits endpoint: `curl http://EC2_IP/splits?dataset=ibm/duorc`
- [ ] Verify S3 connection in worker logs
- [ ] Monitor disk usage: `df -h /mnt/data`
- [ ] Check MongoDB connection: `docker-compose logs api | grep "cache database"`

### Frontend Connection

- [ ] Update frontend API_BASE_URL to `http://EC2_IP`
- [ ] Test a dataset viewer request from frontend
- [ ] Verify assets load (images/audio if dataset has them)

### Monitoring Setup (Optional)

- [ ] Set up CloudWatch monitoring in AWS console
- [ ] Save SSH command for quick access
- [ ] Bookmark EC2 instance page

## Emergency Commands

If something breaks, use these:

```bash
# View all logs
docker-compose logs -f

# Restart specific service
docker-compose restart api

# Restart everything
docker-compose restart

# Check resource usage
docker stats

# SSH backup (if lost key)
# → Create new key pair in AWS console, attach new instance
```

## Done! ✨

Your dataset viewer backend is live at:
- **Public API**: `http://EC2_PUBLIC_IP`
- **Healthcheck**: `http://EC2_PUBLIC_IP/healthcheck`

Rough monthly costs:
- EC2 t3.small: $7
- Storage: $2
- Data transfer: $5-10
- MongoDB Atlas (optional paid tier): $0-9
- Supabase S3: $5-20 (usage-based)
- **Total: ~$20-48/mo**

---

**Questions?** Check the full guide: [AWS-EC2-DEPLOYMENT.md](AWS-EC2-DEPLOYMENT.md)
