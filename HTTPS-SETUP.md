# HTTPS & Authentication Setup Guide

This guide covers setting up HTTPS with Let's Encrypt and enabling HuggingFace JWT authentication for your EC2 deployment.

## Prerequisites

- EC2 instance with dataset-viewer running
- Valid domain name pointing to your EC2 public IP (recommended, or use IP-based cert)
- All services healthy and running
- MongoDB Atlas connectivity confirmed

## Part 1: HTTPS Setup with Let's Encrypt & Certbot

### Step 1: SSH into EC2

```bash
ssh -i data-viewer.pem ec2-user@<EC2-PUBLIC-IP>
cd ~/dataset-viewer
```

### Step 2: Create Certificate Storage Directory

```bash
sudo mkdir -p /mnt/data/certs
sudo chmod 755 /mnt/data/certs
```

### Step 3: Install Certbot (on EC2 host, not in Docker)

```bash
# Update package manager
sudo yum update -y

# Install certbot
sudo yum install -y certbot python3-certbot-dns-route53

# Verify installation
certbot --version
```

### Step 4: Get SSL Certificate from Let's Encrypt

**Option A: Using Domain Name (Recommended)**

If you have a domain (e.g., `datasets.example.com`):

```bash
# Stop services temporarily (certbot needs port 80/443 free)
docker-compose -f docker-compose.ec2.yml --env-file .env.production down

# Request certificate
sudo certbot certonly --standalone \
  --email your-email@example.com \
  --agree-tos \
  -d your-domain.com

# Restart services
docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d
```

**Option B: Using EC2 Public IP (Self-Signed, for testing)**

```bash
# For testing without domain, generate self-signed cert
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /mnt/data/certs/privkey.pem \
  -out /mnt/data/certs/fullchain.pem \
  -subj "/CN=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
```

### Step 5: Copy Certificates to Storage Directory

If certbot was used (Option A):

```bash
# Certbot stores certs in /etc/letsencrypt/
# Copy them to your persistent storage
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem /mnt/data/certs/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem /mnt/data/certs/

# Fix permissions so Docker can read
sudo chmod 644 /mnt/data/certs/fullchain.pem
sudo chmod 644 /mnt/data/certs/privkey.pem
sudo chown ec2-user:ec2-user /mnt/data/certs/*
```

### Step 6: Update .env.production with HTTPS URLs

Edit `.env.production` and replace URLs:

**Before:**
```bash
ASSETS_BASE_URL=http://<YOUR-DOMAIN-OR-IP>/assets
CACHED_ASSETS_BASE_URL=http://<YOUR-DOMAIN-OR-IP>/cached-assets
```

**After:**
```bash
ASSETS_BASE_URL=https://your-domain.com/assets
# OR if using IP:
ASSETS_BASE_URL=https://54.83.64.219/assets

CACHED_ASSETS_BASE_URL=https://your-domain.com/cached-assets
# OR if using IP:
CACHED_ASSETS_BASE_URL=https://54.83.64.219/cached-assets
```

### Step 7: Restart Services with HTTPS Config

```bash
# Upload updated .env.production (from local machine)
scp -i data-viewer.pem .env.production ec2-user@<EC2-IP>:~/dataset-viewer/

# Back on EC2: restart services
cd ~/dataset-viewer
docker-compose -f docker-compose.ec2.yml --env-file .env.production down
docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d

# Wait for services to stabilize
sleep 5
docker-compose -f docker-compose.ec2.yml --env-file .env.production ps
```

### Step 8: Test HTTPS

```bash
# Test from local machine
curl -k https://54.83.64.219/healthcheck
curl -k https://your-domain.com/healthcheck

# Test with actual API call
curl -k https://54.83.64.219/splits?dataset=ibm/duorc
```

### Step 9: Set Up Auto-Renewal (Optional, Recommended for Prod)

For domain-based certificates, set up automatic renewal:

```bash
# Create renewal script
cat > ~/renew-certs.sh << 'EOF'
#!/bin/bash
cd ~/dataset-viewer

# Stop docker services
docker-compose -f docker-compose.ec2.yml --env-file .env.production down

# Renew certificate
sudo certbot renew --quiet

# Copy renewed certs
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem /mnt/data/certs/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem /mnt/data/certs/
sudo chmod 644 /mnt/data/certs/*
sudo chown ec2-user:ec2-user /mnt/data/certs/*

# Restart services
docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d
EOF

chmod +x ~/renew-certs.sh

# Add to crontab (runs daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * ~/renew-certs.sh >> /var/log/cert-renewal.log 2>&1") | crontab -
```

## Part 2: Enable HuggingFace JWT Authentication

### Step 1: Verify HF API Endpoint Accessibility

From EC2, test if you can reach HuggingFace:

```bash
curl -v https://huggingface.co/api/keys/jwt | head -20
```

If this fails, your EC2 security group may be blocking outbound HTTPS. Fix it:
- Go to AWS Console → EC2 → Security Groups
- Add outbound rule: HTTPS (443) to 0.0.0.0/0

### Step 2: Update .env.production (Already Done)

The authentication settings are already in place:

```bash
API_HF_JWT_PUBLIC_KEY_URL=https://huggingface.co/api/keys/jwt
API_HF_JWT_ADDITIONAL_PUBLIC_KEYS=
API_HF_JWT_ALGORITHM=EdDSA
```

### Step 3: Verify Services Loaded JWT Keys

Restart services and check logs:

```bash
# First time: may see JWT key fetching
docker-compose -f docker-compose.ec2.yml --env-file .env.production logs api rows search | grep -i "jwt\|public"

# Should see successful key loading, no errors about 404
```

### Step 4: Test Authenticated Request

Create a test script to verify JWT auth works:

```bash
# Get a HF token (create one at https://huggingface.co/settings/tokens)
export HF_TOKEN=hf_xxxxxxx

# Make authenticated request
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://54.83.64.219/splits?dataset=ibm/duorc
```

## Troubleshooting

### Certificate File Permissions Error

```
Error: Permission denied reading /etc/nginx/certs/privkey.pem
```

Fix:
```bash
sudo chmod 644 /mnt/data/certs/*
sudo chown ec2-user:ec2-user /mnt/data/certs/*
```

### SSL Certificate Expired

```bash
# Check cert expiration
sudo certbot certificates

# Renew manually
sudo certbot renew --force-renewal
```

### Can't Fetch JWT Keys (404 Error)

Check EC2 security group and connectivity:

```bash
# Test from EC2
curl -v https://huggingface.co/api/keys/jwt

# If fails: check outbound HTTPS in security group
# AWS Console → Security Groups → Outbound Rules → Add HTTPS to 0.0.0.0/0
```

### Services Still Using HTTP

Verify nginx is loading the new HTTPS config:

```bash
# Check which config is loaded
docker-compose -f docker-compose.ec2.yml --env-file .env.production exec reverse-proxy \
  cat /etc/nginx/conf.d/default.conf | head -30

# Should show SSL certificate directives
```

## Verification Checklist

- [ ] EC2 port 443 is open in security group
- [ ] Certificates exist at `/mnt/data/certs/fullchain.pem` and `/privkey.pem`
- [ ] `.env.production` has HTTPS URLs set
- [ ] Services are healthy: `docker-compose ps`
- [ ] HTTPS healthcheck works: `curl -k https://<IP>/healthcheck`
- [ ] JWT keys loaded without errors in logs
- [ ] Domain DNS (if used) points to EC2 IP

## Production Recommendations

1. **Enable HSTS**: Uncomment the HSTS header in nginx config
2. **Use DNS**: Domain names are more stable than IPs for certificate renewal
3. **Monitor Cert Expiry**: Set calendar reminder 30 days before expiration
4. **Rate Limiting**: Consider adding nginx rate limiting for public endpoints
5. **WAF**: Consider AWS WAF for additional protection
