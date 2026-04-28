# EC2 Deployment Checklist

Use this checklist to deploy the backend on AWS EC2 with Cloudflare R2 + MongoDB Atlas.

## Pre-Launch Checklist

### Accounts & Setup (Do these BEFORE launching EC2)

- [ ] **AWS Account** ready and logged in
- [ ] **MongoDB Atlas** account created, cluster running
  - [ ] Database user created
  - [ ] Connection string copied
  - [ ] IP allowlist includes `0.0.0.0/0` (or your EC2 IP after launch)
- [ ] **Cloudflare Account** created, R2 bucket and S3 credentials enabled
  - [ ] S3 access key + secret saved
  - [ ] Account endpoint noted
  - [ ] R2 bucket created (e.g., `models`)
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

Cloudflare R2 S3 section:

- [ ] `S3_ACCESS_KEY_ID` = from Cloudflare R2
- [ ] `S3_SECRET_ACCESS_KEY` = from Cloudflare R2
- [ ] `S3_REGION_NAME` = auto
- [ ] `S3_ENDPOINT_URL` = https://<account-id>.r2.cloudflarestorage.com
- [ ] `ASSETS_STORAGE_ROOT` = models/assets
- [ ] `CACHED_ASSETS_STORAGE_ROOT` = models/cached-assets
- [ ] `LOCAL_DATASETS_STORAGE_ROOT` = models/local-datasets

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
- [ ] Check training capabilities: `curl http://localhost/train/capabilities`
- [ ] Check training alias: `curl http://localhost/api/train/capabilities`
- [ ] Check worker logs: `docker-compose logs worker | head -20`
- [ ] No errors in logs

If training capabilities return 404, rebuild API + reverse proxy:

```bash
docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d --build api reverse-proxy
bash validate-deployment.sh
```

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
- Cloudflare R2: $5-20 (usage-based)
- **Total: ~$20-48/mo**

---

**Questions?** Check the full guide: [AWS-EC2-DEPLOYMENT.md](AWS-EC2-DEPLOYMENT.md)
