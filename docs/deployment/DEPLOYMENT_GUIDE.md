# Eidolon AI Personal Assistant - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Eidolon AI Personal Assistant in production environments. It covers installation, configuration, security, monitoring, and maintenance procedures.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Installation Checklist](#pre-installation-checklist)
3. [Installation Process](#installation-process)
4. [Configuration](#configuration)
5. [Security Setup](#security-setup)
6. [Service Management](#service-management)
7. [Monitoring & Health Checks](#monitoring--health-checks)
8. [Backup & Recovery](#backup--recovery)
9. [Updates & Maintenance](#updates--maintenance)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10/11 (via WSL2)
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 100GB free space (SSD recommended)
- **Network**: Stable internet connection for cloud AI services
- **Python**: 3.9+ with pip and venv

### Recommended Production Specs
- **CPU**: 8 cores, 3.0GHz+
- **RAM**: 16GB+
- **Storage**: 500GB SSD
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, for local AI acceleration)
- **Network**: 100Mbps+ with low latency

### Dependencies
- **System**: tesseract-ocr, ffmpeg, build tools
- **Python**: See `requirements.txt` for Python dependencies
- **Optional**: Docker for containerized deployment

## Pre-Installation Checklist

### 1. Environment Preparation
- [ ] Verify system meets minimum requirements
- [ ] Ensure user has appropriate permissions (non-root user with sudo access)
- [ ] Confirm network connectivity and firewall settings
- [ ] Plan data storage locations and backup strategy
- [ ] Obtain necessary API keys for cloud AI services

### 2. Security Considerations
- [ ] Review privacy and compliance requirements
- [ ] Plan encryption key management
- [ ] Configure network security (firewalls, VPN access)
- [ ] Set up monitoring and alerting
- [ ] Plan user access and authentication

### 3. Infrastructure Planning
- [ ] Determine service management approach (systemd, Docker, etc.)
- [ ] Plan backup and disaster recovery procedures
- [ ] Set up monitoring infrastructure
- [ ] Configure log management and retention
- [ ] Plan for updates and maintenance windows

## Installation Process

### Method 1: Automated Installation (Recommended)

```bash
# Download and run the installation script
curl -fsSL https://raw.githubusercontent.com/eidolon-ai/eidolon/main/scripts/install.sh | bash

# For enterprise features
curl -fsSL https://raw.githubusercontent.com/eidolon-ai/eidolon/main/scripts/install.sh | bash -s -- --enterprise
```

### Method 2: Manual Installation

```bash
# 1. Create user and directories
sudo useradd -m -s /bin/bash eidolon
sudo mkdir -p /opt/eidolon
sudo chown eidolon:eidolon /opt/eidolon

# 2. Switch to eidolon user
sudo su - eidolon

# 3. Set up Python environment
python3 -m venv /opt/eidolon/venv
source /opt/eidolon/venv/bin/activate

# 4. Install Eidolon
pip install --upgrade pip wheel
pip install eidolon[enterprise]

# 5. Initialize configuration
eidolon init --production
```

### Method 3: Docker Deployment

```bash
# Pull the official image
docker pull eidolon/eidolon:latest

# Run with docker-compose
curl -O https://raw.githubusercontent.com/eidolon-ai/eidolon/main/docker-compose.prod.yml
docker-compose -f docker-compose.prod.yml up -d
```

## Configuration

### 1. Production Configuration

Copy the production configuration template:
```bash
cp /opt/eidolon/config/production-settings.yaml /opt/eidolon/config/settings.yaml
```

### 2. Environment Variables

Create and configure the environment file:
```bash
cat > /opt/eidolon/.env <<EOF
# Eidolon Production Environment

# API Keys (REQUIRED)
GEMINI_API_KEY_PROD=your_gemini_production_key
CLAUDE_API_KEY_PROD=your_claude_production_key
OPENAI_API_KEY_PROD=your_openai_production_key

# Database Configuration
DATABASE_URL=sqlite:///opt/eidolon/data/eidolon.db

# Security
ENCRYPTION_KEY=$(openssl rand -hex 32)
API_SECRET_KEY=$(openssl rand -hex 32)

# Monitoring
ENABLE_MONITORING=true
METRICS_RETENTION_DAYS=30

# Backup
BACKUP_ENABLED=true
BACKUP_LOCATION=/opt/eidolon/backup

# Logging
LOG_LEVEL=INFO
LOG_RETENTION_DAYS=30
EOF

chmod 600 /opt/eidolon/.env
```

### 3. Key Configuration Areas

#### Privacy & Security
```yaml
privacy:
  encrypt_at_rest: true
  auto_redaction: true
  gdpr_compliance: true
  audit_logging: true
```

#### Performance Tuning
```yaml
observer:
  capture_interval: 15
  max_cpu_percent: 15.0
  max_memory_mb: 4096

analysis:
  performance:
    max_model_memory_gb: 2
    batch_processing: true
```

#### Monitoring
```yaml
monitoring:
  enabled: true
  metrics_collection_interval: 30
  dashboard_enabled: true
  alert_thresholds:
    cpu_percent: 80.0
    memory_mb: 6144
```

## Security Setup

### 1. Encryption Configuration

Generate encryption keys:
```bash
# Generate master encryption key
openssl rand -hex 32 > /opt/eidolon/data/.master_key
chmod 600 /opt/eidolon/data/.master_key

# Generate API authentication key
openssl rand -hex 32 > /opt/eidolon/data/.api_key
chmod 600 /opt/eidolon/data/.api_key
```

### 2. File Permissions

Set appropriate file permissions:
```bash
# Set directory permissions
chmod 700 /opt/eidolon
chmod 755 /opt/eidolon/data
chmod 755 /opt/eidolon/logs
chmod 700 /opt/eidolon/config

# Set file permissions
find /opt/eidolon/config -type f -exec chmod 600 {} \;
find /opt/eidolon/data -name "*.key" -exec chmod 600 {} \;
```

### 3. Network Security

Configure firewall rules:
```bash
# Ubuntu/Debian
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8080/tcp  # Monitoring dashboard (restrict as needed)
sudo ufw enable

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=22/tcp
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

### 4. SSL/TLS Configuration

For web interfaces, configure SSL certificates:
```bash
# Generate self-signed certificate (for testing)
openssl req -x509 -newkey rsa:4096 -keyout /opt/eidolon/ssl/key.pem -out /opt/eidolon/ssl/cert.pem -days 365 -nodes

# Or use Let's Encrypt for production
sudo certbot certonly --standalone -d your-domain.com
```

## Service Management

### 1. Install Service

```bash
# Run the service setup script
/opt/eidolon/scripts/service/setup-service.sh

# Enable and start service
sudo systemctl enable eidolon
sudo systemctl start eidolon
```

### 2. Service Management Commands

```bash
# Check service status
sudo systemctl status eidolon

# View logs
sudo journalctl -u eidolon -f

# Restart service
sudo systemctl restart eidolon

# Stop service
sudo systemctl stop eidolon
```

### 3. Service Configuration

Edit service file if needed:
```bash
sudo systemctl edit eidolon
```

Add custom configuration:
```ini
[Service]
Environment=EIDOLON_ENV=production
LimitNOFILE=65536
Restart=always
RestartSec=10
```

## Monitoring & Health Checks

### 1. Enable Monitoring Dashboard

Access the monitoring dashboard:
```bash
# Start dashboard
eidolon dashboard start

# Access at http://localhost:8080
```

### 2. Health Check Commands

```bash
# System health check
eidolon status

# Detailed health report
eidolon health-check --detailed

# Performance metrics
eidolon metrics --hours 24
```

### 3. Log Monitoring

Configure log monitoring:
```bash
# Real-time log monitoring
tail -f /opt/eidolon/logs/eidolon.log

# Error log monitoring
grep ERROR /opt/eidolon/logs/eidolon.log

# Performance log analysis
eidolon analyze-logs --performance
```

### 4. External Monitoring Integration

Example Prometheus configuration:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'eidolon'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

## Backup & Recovery

### 1. Configure Automated Backups

```bash
# Set up daily backups
/opt/eidolon/scripts/backup/backup-manager.sh schedule data daily

# Set up weekly full backups
/opt/eidolon/scripts/backup/backup-manager.sh schedule full weekly
```

### 2. Manual Backup Commands

```bash
# Create full backup
/opt/eidolon/scripts/backup/backup-manager.sh backup full

# Create data-only backup
/opt/eidolon/scripts/backup/backup-manager.sh backup data

# List backups
/opt/eidolon/scripts/backup/backup-manager.sh list
```

### 3. Restore Procedures

```bash
# List available backups
/opt/eidolon/scripts/backup/backup-manager.sh list

# Restore from backup
/opt/eidolon/scripts/backup/backup-manager.sh restore backup_id_here

# Test restore (to temporary location)
/opt/eidolon/scripts/backup/backup-manager.sh restore backup_id_here /tmp/restore_test
```

### 4. Disaster Recovery Plan

1. **Immediate Response**
   - Stop Eidolon service
   - Assess data integrity
   - Identify most recent valid backup

2. **Recovery Process**
   - Restore from latest backup
   - Verify data integrity
   - Test system functionality
   - Resume normal operations

3. **Post-Recovery**
   - Analyze failure cause
   - Update procedures if needed
   - Document incident

## Updates & Maintenance

### 1. Update Management

```bash
# Check for updates
eidolon update check

# Download update (manual approval)
eidolon update download

# Install update with automatic backup
eidolon update install --backup

# Rollback if needed
eidolon update rollback
```

### 2. Maintenance Tasks

#### Weekly Maintenance
```bash
# Clean up old logs
eidolon cleanup logs --days 30

# Optimize database
eidolon database optimize

# Check system health
eidolon health-check --detailed
```

#### Monthly Maintenance
```bash
# Clean up old backups
/opt/eidolon/scripts/backup/backup-manager.sh cleanup 90 20

# Rotate encryption keys
eidolon security rotate-keys

# Review performance metrics
eidolon metrics --hours 720 --export /tmp/monthly_metrics.json
```

### 3. Maintenance Windows

Schedule maintenance during low-usage periods:
```bash
# Schedule maintenance window
eidolon maintenance schedule --start "02:00" --duration "2h" --day "sunday"

# Run maintenance tasks
eidolon maintenance run --tasks "cleanup,optimize,health-check"
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check service status
sudo systemctl status eidolon

# Check logs
sudo journalctl -u eidolon --since "1 hour ago"

# Check configuration
eidolon config validate

# Check permissions
ls -la /opt/eidolon/
```

#### 2. High Memory Usage
```bash
# Check memory usage
eidolon metrics memory

# Restart service to clear memory
sudo systemctl restart eidolon

# Adjust memory limits in configuration
```

#### 3. Database Issues
```bash
# Check database integrity
eidolon database check

# Repair database if needed
eidolon database repair

# Rebuild database from backups if necessary
eidolon database restore
```

#### 4. API Connection Issues
```bash
# Test API connectivity
eidolon test api-connections

# Check API key configuration
eidolon config show api-keys

# Review network connectivity
eidolon test network
```

### Log Analysis

Common log patterns to monitor:
```bash
# Error patterns
grep -E "(ERROR|CRITICAL)" /opt/eidolon/logs/eidolon.log

# Performance issues
grep -E "(slow|timeout|memory)" /opt/eidolon/logs/eidolon.log

# Security events
grep -E "(auth|login|access)" /opt/eidolon/logs/eidolon.log
```

## Best Practices

### 1. Security Best Practices

- **Use strong, unique API keys** for all cloud services
- **Enable encryption** for data at rest and in transit
- **Regularly rotate** encryption keys and API keys
- **Monitor access logs** for suspicious activity
- **Keep software updated** with latest security patches
- **Use least privilege principle** for user accounts
- **Implement network segmentation** where possible

### 2. Performance Best Practices

- **Monitor resource usage** regularly
- **Optimize capture intervals** based on usage patterns
- **Use local AI models** when possible to reduce API costs
- **Implement proper caching** for frequently accessed data
- **Clean up old data** regularly to maintain performance
- **Use SSDs** for data storage
- **Monitor and optimize** database queries

### 3. Operational Best Practices

- **Implement automated backups** with regular testing
- **Set up comprehensive monitoring** and alerting
- **Document all configuration changes**
- **Maintain an incident response plan**
- **Regular health checks** and system maintenance
- **Capacity planning** for growth
- **Regular staff training** on procedures

### 4. Compliance Best Practices

- **Implement data retention policies** according to regulations
- **Enable audit logging** for all system activities
- **Regular compliance audits** and assessments
- **Data minimization** - collect only necessary data
- **User consent management** for data processing
- **Privacy impact assessments** for changes
- **Regular staff training** on privacy and compliance

## Support and Resources

### Documentation
- [API Reference](../api/README.md)
- [User Guide](../user-guide/README.md)
- [Configuration Reference](../CONFIG_GUIDE.md)

### Community
- GitHub Issues: https://github.com/eidolon-ai/eidolon/issues
- Discord: [Eidolon Community](https://discord.gg/eidolon)
- Forum: https://forum.eidolon.ai

### Professional Support
- Enterprise Support: support@eidolon.ai
- Consulting Services: consulting@eidolon.ai
- Training: training@eidolon.ai

## Appendix

### A. Configuration Templates
- [Production Settings](../config/production-settings.yaml)
- [Docker Compose](../../docker-compose.prod.yml)
- [Environment Variables](.env.example)

### B. Security Checklist
- [ ] Encryption keys generated and secured
- [ ] File permissions set correctly
- [ ] Network security configured
- [ ] API keys secured
- [ ] Audit logging enabled
- [ ] Backup encryption enabled

### C. Performance Tuning Checklist
- [ ] Resource limits configured
- [ ] Monitoring enabled
- [ ] Database optimized
- [ ] Cache settings tuned
- [ ] Log rotation configured
- [ ] Cleanup procedures scheduled

### D. Compliance Checklist
- [ ] Data retention policies implemented
- [ ] Privacy controls enabled
- [ ] Audit logging configured
- [ ] User consent mechanisms in place
- [ ] Data export capabilities tested
- [ ] Incident response plan documented