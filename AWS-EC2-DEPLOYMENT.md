# EC2 Deployment Guide for Dataset Viewer Backend

Deploy the full 8-service backend on a single AWS EC2 instance for ~$7/mo + storage costs.

## Prerequisites

1. **AWS Account** with EC2 access
2. **MongoDB Atlas** account (free tier 512MB, or $9/mo tier for production)
3. **Supabase Account** with S3 bucket and credentials
4. **EC2 Key Pair** (for SSH access)
5. **Security Group** allowing:
   - Inbound HTTP (80) and HTTPS (443) - for reverse proxy
   - Inbound SSH (22) - for you only
   - All outbound (default)

## Step 1: Launch EC2 Instance

1. Go to **AWS EC2 Dashboard** → **Instances** → **Launch Instance**
2. Select **Ubuntu 22.04 LTS** (free tier eligible, t2.micro or t3.small)
3. **Instance Type**: `t3.small` (1 vCPU, 2GB RAM) - ~$7/mo
4. **Storage**: 30 GB GP2 (default) - ~$2/mo
5. **Security Group**: Create one that allows:
   - SSH from your IP
   - HTTP (80) from 0.0.0.0/0
6. **Advanced Details** → **User Data**: Paste the contents of `aws-ec2-launch.sh` from this repo
7. **Review & Launch** → Accept the key pair and launch

**Wait 3-5 minutes for instance to boot and initialize.**

## Step 2: Configure Environment Variables

While the instance is launching, prepare your `.env.production` file:

1. Copy `.env.production.template` to `.env.production`
2. Fill in the following:

### MongoDB Atlas
1. Create free cluster at mongodb.com
2. Create database user + get connection string
3. Set `CACHE_MONGO_URL` and `QUEUE_MONGO_URL`
4. Example: `mongodb+srv://user:password@cluster.mongodb.net/dataset_viewer_cache?retryWrites=true&w=majority`

### Supabase S3
1. Go to your Supabase project → **Settings** → **Storage**
2. Create S3 credentials (or use existing)
3. Find your project reference (in URL: `https://PROJECT_REF.supabase.co`)
4. Fill in:
   - `S3_ACCESS_KEY_ID`
   - `S3_SECRET_ACCESS_KEY`
   - `S3_ENDPOINT_URL=https://PROJECT_REF.storage.supabase.co/storage/v1/s3`
5. Create a bucket (e.g., `dataset-viewer`) and set `ASSETS_STORAGE_ROOT=dataset-viewer/assets`

### Public URLs
Once your EC2 instance is running, get its public IP from the AWS console, then set:
- `ASSETS_BASE_URL=http://<EC2_PUBLIC_IP>/assets`
- `CACHED_ASSETS_BASE_URL=http://<EC2_PUBLIC_IP>/cached-assets`

## Step 3: Upload and Start Services

1. **SSH into your instance:**
   ```bash
   ssh -i /path/to/key.pem ubuntu@<ec2-public-ip>
   ```

2. **Upload your `.env.production` file:**
   ```bash
   scp -i /path/to/key.pem .env.production ubuntu@<ec2-public-ip>:~/dataset-viewer/
   ```

3. **Navigate to repo and start:**
   ```bash
   cd ~/dataset-viewer
   docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d
   ```

4. **Check status:**
   ```bash
   docker-compose ps
   docker-compose logs -f reverse-proxy
   ```

Wait 2-3 minutes for all services to start (especially Search, which downloads DuckDB extensions).

## Step 4: Validate Deployment

Once all services are healthy (green in `docker-compose ps`):

1. **Test reverse proxy:**
   ```bash
   curl http://<ec2-public-ip>/healthcheck
   ```

2. **Verify training capabilities routes:**
   ```bash
   curl http://<ec2-public-ip>/train/capabilities
   curl http://<ec2-public-ip>/api/train/capabilities
   ```

   If either returns 404, your API/reverse-proxy containers are likely stale. Rebuild:
   ```bash
   docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d --build api reverse-proxy
   ```

3. **Test API:**
   ```bash
   curl http://<ec2-public-ip>/splits?dataset=ibm/duorc
   ```

4. **Check worker processing:**
   ```bash
   docker-compose logs worker | grep "Processing job"
   ```

5. **Verify S3 writes:**
   ```bash
   docker-compose logs worker | grep "storage"
   ```

## Step 5: Connect Your Frontend

Point your frontend to:
```
API_BASE_URL = http://<ec2-public-ip>
```

Endpoints available:
- `/splits`
- `/first-rows`
- `/rows`
- `/search`
- `/parquet`
- `/statistics`

## Monitoring & Maintenance

### View logs:
```bash
docker-compose logs <service-name>
docker-compose logs -f  # Follow all
```

### Restart a service:
```bash
docker-compose restart api
```

### Restart everything:
```bash
docker-compose restart
```

### Stop everything:
```bash
docker-compose down
```

### Free up disk space:
```bash
docker system prune -a
```

## Costs Breakdown

| Component | Cost/Month |
|-----------|-----------|
| t3.small EC2 | $7.00 |
| 30GB storage | $2.00 |
| Data transfer out (est.) | $5-10.00 |
| MongoDB Atlas free OR paid | $0-9.00 |
| Supabase S3 (usage-based) | $5-20.00 |
| **Total** | **$19-48/mo** |

*(Assuming low traffic; will scale with usage)*

## Troubleshooting

### Services won't start
```bash
# Check Docker is running
docker ps

# View detailed logs
docker-compose logs <service-name>

# Common: MongoDB connection failed
# → Check CACHE_MONGO_URL is correct
# → Check Atlas IP whitelist includes EC2 IP
```

### Storage errors
```bash
# Check S3 connectivity
docker-compose exec api curl $S3_ENDPOINT_URL

# Check local disk
df -h /mnt/data
```

### High CPU/Memory
```bash
docker stats
# Reduce WORKER_UVICORN_NUM_WORKERS to 1 in .env.production and restart
```

## Next Steps

- **Custom domain**: Point DNS to EC2 public IP or attach Elastic IP
- **HTTPS**: Use Let's Encrypt certbot in nginx reverse proxy
- **Scaling**: Move to ECS Fargate if traffic grows beyond single instance capacity
- **Backups**: Set up EBS snapshots in AWS console

---

Questions? Check service logs first:
```bash
docker-compose logs -f <service>
```
