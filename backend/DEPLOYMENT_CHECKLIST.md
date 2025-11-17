# Backend Deployment Checklist & Quick Reference

## 📋 Pre-Deployment Checklist

### Code Quality
- [ ] All code is PEP 8 compliant
- [ ] No hardcoded secrets in code
- [ ] Error handling on all endpoints
- [ ] Logging implemented for debugging
- [ ] Type hints on all functions
- [ ] Docstrings on all modules/functions

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] API endpoints tested with Postman/curl
- [ ] Load tested with concurrent requests
- [ ] Audio file validation tested
- [ ] Error responses tested

### Configuration
- [ ] `.env.example` provided with all required vars
- [ ] All environment variables documented
- [ ] Settings validated on startup
- [ ] Secrets stored in environment, not code
- [ ] Different configs for dev/staging/prod

### Security
- [ ] CORS origins restricted (not `*`)
- [ ] Input validation on all endpoints
- [ ] File upload size limits set
- [ ] SQL injection prevention (if DB used)
- [ ] HTTPS enabled in production
- [ ] API authentication implemented
- [ ] Rate limiting configured

### Documentation
- [ ] README.md complete
- [ ] DEPLOYMENT.md complete
- [ ] ARCHITECTURE.md complete
- [ ] API endpoints documented
- [ ] Error codes documented
- [ ] Database schema documented (if applicable)

### Performance
- [ ] Response times acceptable
- [ ] CPU/Memory usage monitored
- [ ] Database queries optimized
- [ ] Caching implemented where needed
- [ ] Async/await properly used
- [ ] No N+1 query problems

### Monitoring & Logging
- [ ] Logging configured
- [ ] Log rotation configured
- [ ] Error alerting set up
- [ ] Uptime monitoring configured
- [ ] Performance metrics collected
- [ ] Access logs recorded

---

## 🚀 Deployment Steps

### Local Development
```bash
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1              # Windows
source .venv/bin/activate                # Linux/Mac

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Create .env file
cp backend/.env.example backend/.env

# 4. Run backend
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### Docker Deployment (Local)
```bash
cd backend
docker-compose up --build
```

### Docker Deployment (Production)
```bash
# Build image
docker build -t voice-backend:1.0.0 -f backend/Dockerfile .

# Push to registry
docker tag voice-backend:1.0.0 your-registry/voice-backend:1.0.0
docker push your-registry/voice-backend:1.0.0

# Deploy with docker-compose
docker-compose -f backend/docker-compose.yml up -d
```

### Cloud Deployment (AWS)

#### Using AWS App Runner
```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name voice-backend

# 2. Build and push image
docker build -t voice-backend:latest backend/
docker tag voice-backend:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/voice-backend:latest
aws ecr get-login-password | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/voice-backend:latest

# 3. Create App Runner service from AWS Console
```

#### Using AWS Lambda (with Container Image)
```bash
# 1. Create lambda.py
# 2. Use Mangum for ASGI->Lambda adapter
# 3. Deploy via console or CLI
```

### Kubernetes Deployment

```yaml
# backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voice-backend
  template:
    metadata:
      labels:
        app: voice-backend
    spec:
      containers:
      - name: backend
        image: voice-backend:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: production
        - name: LOG_LEVEL
          value: INFO
        healthCheck:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

```bash
kubectl apply -f backend-deployment.yaml
kubectl apply -f backend-service.yaml
```

---

## 📊 Monitoring Dashboard

### Key Metrics to Monitor

1. **Health**
   ```bash
   curl http://127.0.0.1:8000/health
   ```

2. **Response Times**
   - `/detect_wakeword`: Should be < 500ms
   - `/detect_command`: Should be < 500ms
   - `/verify_voice`: Should be < 1000ms

3. **Resource Usage**
   - CPU: Should be < 70% on production
   - Memory: Should be < 50% of available
   - Disk: Audio samples should be managed/archived

4. **Error Rates**
   - Should be < 1% for normal operation
   - Check logs for patterns

5. **Request Volume**
   - Requests per second
   - Peak usage times
   - Trend analysis

---

## 🔧 Maintenance Tasks

### Daily
- [ ] Check error logs for issues
- [ ] Monitor disk space for audio files
- [ ] Verify API is responding

### Weekly
- [ ] Review performance metrics
- [ ] Archive old audio samples
- [ ] Update dependencies (security patches)
- [ ] Run backup procedures

### Monthly
- [ ] Full system health check
- [ ] Database maintenance (if applicable)
- [ ] Performance optimization review
- [ ] Security audit
- [ ] Update documentation

### Quarterly
- [ ] Model performance review
- [ ] Feature analysis from logs
- [ ] Capacity planning
- [ ] Disaster recovery drill

---

## 🐛 Troubleshooting Guide

### Backend Won't Start
```bash
# Check logs
tail -f backend.log

# Verify Python version
python --version

# Check dependencies
pip list | grep -E "fastapi|uvicorn"

# Try restarting from scratch
rm -rf .venv
python -m venv .venv
pip install -r requirements.txt
```

### High Memory Usage
```bash
# Profile memory
python -m memory_profiler backend/app/services/inference_service.py

# Check for memory leaks
python -m tracemalloc

# Consider using async generators for large files
```

### Slow Requests
```bash
# Profile with cProfile
python -m cProfile -s cumtime -m uvicorn app:app

# Check audio file sizes
du -sh backend/data/*

# Archive old files
find backend/data -type f -mtime +30 -delete
```

### Connection Timeouts
```bash
# Check server is running
ps aux | grep uvicorn

# Check port is open
netstat -tuln | grep 8000

# Check firewall rules
sudo iptables -L -n | grep 8000
```

---

## 📝 API Response Codes

| Code | Status | Meaning |
|------|--------|---------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid input (check error message) |
| 413 | Payload Too Large | Audio file too large |
| 500 | Server Error | Check backend logs |
| 503 | Service Unavailable | Backend down or overloaded |

---

## 🔐 Security Checklist

- [ ] All endpoints require authentication
- [ ] CORS only allows known origins
- [ ] File uploads validated (size, type, content)
- [ ] No sensitive data in logs
- [ ] API rate limiting enabled
- [ ] HTTPS enforced
- [ ] Security headers configured
- [ ] SQL injection prevention (if DB used)
- [ ] CSRF tokens implemented
- [ ] Secrets rotated regularly
- [ ] Security updates applied promptly
- [ ] Penetration testing completed

---

## 📚 Additional Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Docker Docs**: https://docs.docker.com/
- **PyTorch Docs**: https://pytorch.org/docs/
- **Uvicorn Docs**: https://www.uvicorn.org/
- **AWS Docs**: https://docs.aws.amazon.com/

---

## ✅ Final Verification

Before going live:

1. **Test all endpoints**
   ```bash
   python test_client_comprehensive.py
   ```

2. **Load test**
   ```bash
   # Using Apache Bench
   ab -n 100 -c 10 http://127.0.0.1:8000/health
   ```

3. **Check logs for errors**
   ```bash
   tail -100 backend.log
   ```

4. **Verify data persistence**
   ```bash
   ls -la backend/data/*/
   ```

5. **Test recovery**
   - Stop backend
   - Verify data is still there
   - Restart backend
   - Verify all works

---

**Last Updated**: 2025-11-17  
**Version**: 1.0.0  
**Status**: ✅ Ready for Production
