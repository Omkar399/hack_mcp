# Screenshot Monitoring

Eidolon's intelligent screenshot monitoring is the foundation of its digital memory system. This feature continuously captures and analyzes your screen activity to build a comprehensive knowledge base of your digital life.

## 🎯 How It Works

### Intelligent Capture
Eidolon doesn't just take screenshots at fixed intervals. Instead, it uses smart algorithms to:

- **Detect Activity Changes**: Only captures when meaningful changes occur on screen
- **Identify Important Content**: Focuses on productive work vs. idle time
- **Minimize Resource Usage**: Optimizes capture frequency based on system load
- **Respect Privacy**: Automatically excludes sensitive applications and content

### Content Analysis Pipeline

1. **Screenshot Capture**: High-quality screen captures using MSS library
2. **Change Detection**: Computer vision algorithms identify meaningful changes
3. **OCR Extraction**: Tesseract and EasyOCR extract all visible text
4. **UI Element Detection**: Florence-2 AI identifies buttons, forms, and interface elements
5. **Content Classification**: CLIP model categorizes content type and importance
6. **Semantic Understanding**: Cloud AI provides deep content analysis (optional)

## ⚙️ Configuration Options

### Basic Settings

```bash
# Set capture frequency (seconds between potential captures)
python -m eidolon config set observer.capture_interval 10

# Adjust sensitivity (0.01 = very sensitive, 0.1 = less sensitive)
python -m eidolon config set observer.activity_threshold 0.05

# Set storage limits (GB)
python -m eidolon config set observer.max_storage_gb 50
```

### Advanced Settings

```yaml
# ~/.eidolon/config/settings.yaml
observer:
  capture_interval: 10
  activity_threshold: 0.05
  storage_path: "./data/screenshots"
  max_storage_gb: 50
  
  # Quality settings
  image_quality: 85
  compression_level: 6
  
  # Activity detection
  motion_threshold: 0.02
  text_change_threshold: 0.1
  ui_change_threshold: 0.05
  
  # Timing controls
  active_hours: "8-20"  # Only capture 8 AM - 8 PM
  active_days: ["monday", "tuesday", "wednesday", "thursday", "friday"]
  
  # Multi-monitor support
  monitor_selection: "all"  # "all", "primary", or specific monitor index
  
  # Performance optimization
  max_fps: 2  # Maximum captures per second
  batch_processing: true
  async_analysis: true
```

## 🖥️ Multi-Monitor Support

### Configuration

```bash
# Capture all monitors
python -m eidolon config set observer.monitor_selection "all"

# Capture only primary monitor
python -m eidolon config set observer.monitor_selection "primary"

# Capture specific monitor (0-indexed)
python -m eidolon config set observer.monitor_selection 1
```

### Monitor Information

```bash
# List available monitors
python -m eidolon monitors list

# Get monitor details
python -m eidolon monitors info

# Test capture on specific monitor
python -m eidolon capture --monitor 1 --test
```

## 🔍 Content Detection Features

### Text Recognition (OCR)

Eidolon uses multiple OCR engines for maximum accuracy:

- **Tesseract**: Primary OCR engine with multiple language support
- **EasyOCR**: Backup OCR for challenging text recognition
- **Cloud OCR**: Optional cloud-based OCR for highest accuracy

```bash
# Configure OCR settings
python -m eidolon config set analysis.ocr.primary_engine "tesseract"
python -m eidolon config set analysis.ocr.languages "['eng', 'spa', 'fra']"
python -m eidolon config set analysis.ocr.confidence_threshold 0.6
```

### UI Element Detection

Florence-2 AI model identifies interface elements:

- **Buttons and Controls**: Clickable interface elements
- **Text Fields**: Input areas and forms
- **Navigation Elements**: Menus, tabs, and navigation bars
- **Content Areas**: Main content regions and panels

### Content Classification

CLIP model automatically categorizes content:

- **Work vs. Personal**: Distinguishes professional from personal activities
- **Content Types**: Documents, websites, applications, media
- **Importance Scoring**: Rates content relevance and significance
- **Topic Classification**: Identifies subject matter and themes

## 📊 Monitoring Statistics

### Real-time Status

```bash
# Check current monitoring status
python -m eidolon status

# View capture statistics
python -m eidolon stats capture

# Monitor performance impact
python -m eidolon stats performance
```

### Historical Analysis

```bash
# Daily capture summary
python -m eidolon stats daily

# Weekly productivity insights
python -m eidolon insights --period "week"

# Storage usage breakdown
python -m eidolon stats storage --detailed
```

## 🔧 Performance Optimization

### Resource Management

```bash
# Check current resource usage
python -m eidolon stats resources

# Optimize for low-end systems
python -m eidolon config set observer.performance_mode "conservative"

# Optimize for high-performance systems
python -m eidolon config set observer.performance_mode "aggressive"
```

### Performance Modes

#### Conservative Mode
- Lower capture frequency
- Reduced image quality
- Minimal background processing
- Ideal for older systems or laptops

#### Balanced Mode (Default)
- Adaptive capture frequency
- Standard image quality
- Background processing during idle
- Good for most systems

#### Aggressive Mode
- High capture frequency
- Maximum image quality
- Real-time processing
- Best for powerful workstations

### Custom Performance Tuning

```yaml
observer:
  performance_mode: "custom"
  
  # Capture settings
  capture_interval: 5
  max_captures_per_minute: 12
  
  # Processing settings
  async_processing: true
  batch_size: 10
  max_concurrent_jobs: 4
  
  # Quality vs. performance trade-offs
  fast_analysis: true
  skip_redundant_captures: true
  intelligent_scheduling: true
```

## 🛡️ Privacy and Security

### Application Exclusions

```bash
# Exclude sensitive applications
python -m eidolon config add privacy.excluded_apps "1Password"
python -m eidolon config add privacy.excluded_apps "Banking App"

# View current exclusions
python -m eidolon config get privacy.excluded_apps
```

### Content Filtering

```bash
# Add sensitive patterns to auto-redact
python -m eidolon config add privacy.sensitive_patterns "password"
python -m eidolon config add privacy.sensitive_patterns "social security"

# Configure redaction behavior
python -m eidolon config set privacy.redaction_mode "blur"  # or "block", "skip"
```

### Secure Storage

```bash
# Enable encryption for stored screenshots
python -m eidolon config set security.encrypt_storage true

# Set encryption key (or use default derived key)
python -m eidolon config set security.encryption_key "your-secure-key"
```

## 🚨 Troubleshooting

### Common Issues

#### No Screenshots Being Captured

```bash
# Check permissions
python -m eidolon doctor permissions

# Test screenshot capability
python -m eidolon test capture

# Check if monitoring is running
python -m eidolon status
```

#### Poor OCR Quality

```bash
# Test OCR with sample image
python -m eidolon test ocr

# Increase image quality
python -m eidolon config set observer.image_quality 95

# Switch OCR engine
python -m eidolon config set analysis.ocr.primary_engine "easyocr"
```

#### High Resource Usage

```bash
# Check resource consumption
python -m eidolon stats resources

# Switch to conservative mode
python -m eidolon config set observer.performance_mode "conservative"

# Reduce capture frequency
python -m eidolon config set observer.capture_interval 30
```

### Performance Issues

#### Slow Analysis

```bash
# Enable fast analysis mode
python -m eidolon config set analysis.fast_mode true

# Reduce batch size
python -m eidolon config set observer.batch_size 5

# Disable cloud analysis
python -m eidolon config set analysis.cloud_enabled false
```

#### Storage Issues

```bash
# Check storage usage
python -m eidolon stats storage

# Clean old data
python -m eidolon cleanup --older-than "30 days"

# Compress existing data
python -m eidolon compress --all
```

## 🔄 Monitoring Management

### Start/Stop Monitoring

```bash
# Start monitoring in background
python -m eidolon capture --daemon

# Stop monitoring
python -m eidolon stop

# Restart monitoring
python -m eidolon restart

# Monitor in foreground (for debugging)
python -m eidolon capture --verbose
```

### Scheduled Monitoring

```bash
# Set up automatic start at system boot
python -m eidolon service install

# Configure scheduled monitoring windows
python -m eidolon schedule --weekdays "9:00-17:00"
python -m eidolon schedule --weekends "10:00-14:00"
```

## 📈 Advanced Features

### AI-Enhanced Monitoring

```bash
# Enable intelligent content prioritization
python -m eidolon config set analysis.smart_prioritization true

# Configure importance thresholds
python -m eidolon config set analysis.importance_threshold 0.7

# Enable predictive capture
python -m eidolon config set observer.predictive_capture true
```

### Custom Triggers

```bash
# Capture on specific window titles
python -m eidolon config add observer.trigger_keywords "Important Meeting"

# Capture on application switches
python -m eidolon config set observer.capture_on_app_switch true

# Capture on clipboard changes
python -m eidolon config set observer.capture_on_clipboard true
```

## 🎯 Best Practices

### Optimal Configuration

1. **Start Conservative**: Begin with default settings and adjust based on usage
2. **Monitor Resources**: Regularly check system impact
3. **Review Privacy**: Ensure sensitive content is properly excluded
4. **Regular Maintenance**: Clean up old data and optimize storage

### Privacy Guidelines

1. **Application Exclusions**: Block password managers, banking apps, private browsing
2. **Sensitive Patterns**: Configure automatic redaction for personal information
3. **Review Captures**: Periodically check what's being captured
4. **Secure Storage**: Enable encryption for sensitive environments

### Performance Tips

1. **Match System Capabilities**: Adjust performance mode to your hardware
2. **Optimize Storage**: Use compression and regular cleanup
3. **Network Awareness**: Disable cloud features on metered connections
4. **Battery Optimization**: Reduce frequency on laptops running on battery

---

The monitoring system is the heart of Eidolon's capabilities. With proper configuration, it provides comprehensive digital memory while respecting your privacy and system resources.