#!/bin/bash

# Eidolon Data Backup Manager
# Comprehensive backup and data management system

set -euo pipefail

# Configuration
EIDOLON_HOME="${HOME}/.eidolon"
BACKUP_DIR="${EIDOLON_HOME}/backup"
DATA_DIR="${EIDOLON_HOME}/data"
CONFIG_DIR="${EIDOLON_HOME}/config"
LOGS_DIR="${EIDOLON_HOME}/logs"

# Backup settings
DEFAULT_RETENTION_DAYS=30
DEFAULT_MAX_BACKUPS=10
COMPRESSION_LEVEL=6

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
BACKUP_LOG="${LOGS_DIR}/backup.log"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "${BACKUP_LOG}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "${BACKUP_LOG}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "${BACKUP_LOG}"
}

# Function to create full backup
create_full_backup() {
    local backup_name="full_$(date +'%Y%m%d_%H%M%S')"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    local description="${1:-Manual full backup}"
    
    log "Creating full backup: ${backup_name}"
    
    # Create backup directory
    mkdir -p "${backup_path}"
    
    # Create manifest file
    cat > "${backup_path}/manifest.json" <<EOF
{
    "backup_id": "${backup_name}",
    "backup_type": "full",
    "description": "${description}",
    "created_at": "$(date -Iseconds)",
    "eidolon_version": "$(get_eidolon_version)",
    "includes": ["data", "config", "logs"],
    "compression": "gzip"
}
EOF
    
    # Backup data directory
    if [[ -d "$DATA_DIR" ]]; then
        log "Backing up data directory..."
        tar -czf "${backup_path}/data.tar.gz" -C "${EIDOLON_HOME}" data/
        log "Data backup completed: $(du -h "${backup_path}/data.tar.gz" | cut -f1)"
    fi
    
    # Backup configuration
    if [[ -d "$CONFIG_DIR" ]]; then
        log "Backing up configuration..."
        tar -czf "${backup_path}/config.tar.gz" -C "${EIDOLON_HOME}" config/
        log "Config backup completed: $(du -h "${backup_path}/config.tar.gz" | cut -f1)"
    fi
    
    # Backup recent logs (last 7 days)
    if [[ -d "$LOGS_DIR" ]]; then
        log "Backing up recent logs..."
        find "${LOGS_DIR}" -name "*.log" -mtime -7 -print0 | tar -czf "${backup_path}/logs.tar.gz" --null -T -
        log "Logs backup completed: $(du -h "${backup_path}/logs.tar.gz" | cut -f1)"
    fi
    
    # Create backup metadata
    create_backup_metadata "${backup_path}"
    
    # Calculate total size
    local total_size=$(du -sh "${backup_path}" | cut -f1)
    log "Full backup completed: ${backup_name} (${total_size})"
    
    echo "${backup_name}"
}

# Function to create data-only backup
create_data_backup() {
    local backup_name="data_$(date +'%Y%m%d_%H%M%S')"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    local description="${1:-Data-only backup}"
    
    log "Creating data backup: ${backup_name}"
    
    # Create backup directory
    mkdir -p "${backup_path}"
    
    # Create manifest file
    cat > "${backup_path}/manifest.json" <<EOF
{
    "backup_id": "${backup_name}",
    "backup_type": "data",
    "description": "${description}",
    "created_at": "$(date -Iseconds)",
    "eidolon_version": "$(get_eidolon_version)",
    "includes": ["data"],
    "compression": "gzip"
}
EOF
    
    # Backup data directory
    if [[ -d "$DATA_DIR" ]]; then
        log "Backing up data directory..."
        tar -czf "${backup_path}/data.tar.gz" -C "${EIDOLON_HOME}" data/
        log "Data backup completed: $(du -h "${backup_path}/data.tar.gz" | cut -f1)"
    else
        error "Data directory not found: ${DATA_DIR}"
        return 1
    fi
    
    # Create backup metadata
    create_backup_metadata "${backup_path}"
    
    local total_size=$(du -sh "${backup_path}" | cut -f1)
    log "Data backup completed: ${backup_name} (${total_size})"
    
    echo "${backup_name}"
}

# Function to create incremental backup
create_incremental_backup() {
    local backup_name="incremental_$(date +'%Y%m%d_%H%M%S')"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    local since_hours="${1:-24}"
    local description="Incremental backup (last ${since_hours} hours)"
    
    log "Creating incremental backup: ${backup_name}"
    
    # Create backup directory
    mkdir -p "${backup_path}"
    
    # Create manifest file
    cat > "${backup_path}/manifest.json" <<EOF
{
    "backup_id": "${backup_name}",
    "backup_type": "incremental",
    "description": "${description}",
    "created_at": "$(date -Iseconds)",
    "eidolon_version": "$(get_eidolon_version)",
    "includes": ["changed_files"],
    "since_hours": ${since_hours},
    "compression": "gzip"
}
EOF
    
    # Find changed files in the last N hours
    local changed_files=$(mktemp)
    
    if [[ -d "$DATA_DIR" ]]; then
        find "${DATA_DIR}" -type f -mtime "-${since_hours}h" > "${changed_files}"
    fi
    
    if [[ -s "$changed_files" ]]; then
        log "Backing up $(wc -l < "${changed_files}") changed files..."
        tar -czf "${backup_path}/changed_files.tar.gz" -T "${changed_files}"
        
        # Create file list
        cp "${changed_files}" "${backup_path}/file_list.txt"
        
        log "Incremental backup completed: $(du -h "${backup_path}/changed_files.tar.gz" | cut -f1)"
    else
        log "No changed files found in the last ${since_hours} hours"
        echo "No changes" > "${backup_path}/no_changes.txt"
    fi
    
    rm -f "${changed_files}"
    
    # Create backup metadata
    create_backup_metadata "${backup_path}"
    
    local total_size=$(du -sh "${backup_path}" | cut -f1)
    log "Incremental backup completed: ${backup_name} (${total_size})"
    
    echo "${backup_name}"
}

# Function to restore backup
restore_backup() {
    local backup_id="$1"
    local restore_target="${2:-${EIDOLON_HOME}}"
    local backup_path="${BACKUP_DIR}/${backup_id}"
    
    if [[ ! -d "$backup_path" ]]; then
        error "Backup not found: ${backup_id}"
        return 1
    fi
    
    log "Restoring backup: ${backup_id}"
    
    # Check manifest
    if [[ ! -f "${backup_path}/manifest.json" ]]; then
        error "Backup manifest not found"
        return 1
    fi
    
    local backup_type=$(jq -r '.backup_type' "${backup_path}/manifest.json" 2>/dev/null || echo "unknown")
    log "Backup type: ${backup_type}"
    
    # Stop Eidolon service
    log "Stopping Eidolon service..."
    "${EIDOLON_HOME}/manage-service.sh" stop 2>/dev/null || true
    
    # Create restore point
    local restore_point="pre_restore_$(date +'%Y%m%d_%H%M%S')"
    log "Creating restore point: ${restore_point}"
    create_data_backup "Pre-restore backup - ${restore_point}" > /dev/null
    
    # Restore files based on backup type
    case "$backup_type" in
        "full")
            restore_full_backup "$backup_path" "$restore_target"
            ;;
        "data")
            restore_data_backup "$backup_path" "$restore_target"
            ;;
        "incremental")
            restore_incremental_backup "$backup_path" "$restore_target"
            ;;
        *)
            error "Unknown backup type: ${backup_type}"
            return 1
            ;;
    esac
    
    # Start Eidolon service
    log "Starting Eidolon service..."
    "${EIDOLON_HOME}/manage-service.sh" start 2>/dev/null || true
    
    log "Restore completed successfully"
}

# Function to restore full backup
restore_full_backup() {
    local backup_path="$1"
    local restore_target="$2"
    
    log "Restoring full backup..."
    
    # Restore data
    if [[ -f "${backup_path}/data.tar.gz" ]]; then
        log "Restoring data..."
        tar -xzf "${backup_path}/data.tar.gz" -C "${restore_target}"
    fi
    
    # Restore config
    if [[ -f "${backup_path}/config.tar.gz" ]]; then
        log "Restoring configuration..."
        tar -xzf "${backup_path}/config.tar.gz" -C "${restore_target}"
    fi
    
    # Restore logs (optional)
    if [[ -f "${backup_path}/logs.tar.gz" ]]; then
        log "Restoring logs..."
        tar -xzf "${backup_path}/logs.tar.gz" -C "${restore_target}"
    fi
}

# Function to restore data backup
restore_data_backup() {
    local backup_path="$1"
    local restore_target="$2"
    
    log "Restoring data backup..."
    
    if [[ -f "${backup_path}/data.tar.gz" ]]; then
        tar -xzf "${backup_path}/data.tar.gz" -C "${restore_target}"
    else
        error "Data archive not found in backup"
        return 1
    fi
}

# Function to restore incremental backup
restore_incremental_backup() {
    local backup_path="$1"
    local restore_target="$2"
    
    log "Restoring incremental backup..."
    
    if [[ -f "${backup_path}/changed_files.tar.gz" ]]; then
        tar -xzf "${backup_path}/changed_files.tar.gz" -C "${restore_target}"
    elif [[ -f "${backup_path}/no_changes.txt" ]]; then
        log "No files to restore (no changes in backup)"
    else
        error "No files found in incremental backup"
        return 1
    fi
}

# Function to list backups
list_backups() {
    local backup_type="${1:-all}"
    
    echo -e "${BLUE}Eidolon Backup List${NC}"
    echo "=================="
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        echo "No backups found"
        return 0
    fi
    
    local found_backups=false
    
    for backup_path in "${BACKUP_DIR}"/*; do
        if [[ -d "$backup_path" && -f "${backup_path}/manifest.json" ]]; then
            local backup_id=$(basename "$backup_path")
            local manifest="${backup_path}/manifest.json"
            
            # Parse manifest
            local type=$(jq -r '.backup_type' "$manifest" 2>/dev/null || echo "unknown")
            local description=$(jq -r '.description' "$manifest" 2>/dev/null || echo "No description")
            local created_at=$(jq -r '.created_at' "$manifest" 2>/dev/null || echo "Unknown")
            local size=$(du -sh "$backup_path" | cut -f1)
            
            # Filter by type if specified
            if [[ "$backup_type" != "all" && "$type" != "$backup_type" ]]; then
                continue
            fi
            
            found_backups=true
            
            echo ""
            echo "Backup ID: $backup_id"
            echo "Type: $type"
            echo "Description: $description"
            echo "Created: $created_at"
            echo "Size: $size"
        fi
    done
    
    if [[ "$found_backups" == false ]]; then
        echo "No backups found"
    fi
}

# Function to delete backup
delete_backup() {
    local backup_id="$1"
    local backup_path="${BACKUP_DIR}/${backup_id}"
    
    if [[ ! -d "$backup_path" ]]; then
        error "Backup not found: ${backup_id}"
        return 1
    fi
    
    log "Deleting backup: ${backup_id}"
    
    # Get backup info
    if [[ -f "${backup_path}/manifest.json" ]]; then
        local size=$(du -sh "$backup_path" | cut -f1)
        log "Backup size: ${size}"
    fi
    
    # Confirm deletion
    if [[ "${FORCE_DELETE:-false}" != "true" ]]; then
        echo -n "Are you sure you want to delete backup ${backup_id}? (y/N): "
        read -r confirmation
        if [[ "$confirmation" != "y" && "$confirmation" != "Y" ]]; then
            log "Deletion cancelled"
            return 0
        fi
    fi
    
    # Delete backup
    rm -rf "$backup_path"
    log "Backup deleted: ${backup_id}"
}

# Function to cleanup old backups
cleanup_old_backups() {
    local retention_days="${1:-$DEFAULT_RETENTION_DAYS}"
    local max_backups="${2:-$DEFAULT_MAX_BACKUPS}"
    
    log "Cleaning up old backups (retention: ${retention_days} days, max: ${max_backups})"
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log "No backup directory found"
        return 0
    fi
    
    local deleted_count=0
    
    # Delete backups older than retention period
    while IFS= read -r -d '' backup_path; do
        local backup_id=$(basename "$backup_path")
        local age_days=$((($(date +%s) - $(stat -c %Y "$backup_path")) / 86400))
        
        if [[ $age_days -gt $retention_days ]]; then
            log "Deleting old backup: ${backup_id} (${age_days} days old)"
            rm -rf "$backup_path"
            ((deleted_count++))
        fi
    done < <(find "$BACKUP_DIR" -maxdepth 1 -type d -name "*_*" -print0)
    
    # Keep only the most recent backups if we exceed max count
    local backup_count=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "*_*" | wc -l)
    
    if [[ $backup_count -gt $max_backups ]]; then
        local excess_count=$((backup_count - max_backups))
        log "Too many backups ($backup_count > $max_backups), removing $excess_count oldest"
        
        # Get oldest backups
        while IFS= read -r backup_path; do
            local backup_id=$(basename "$backup_path")
            log "Deleting excess backup: ${backup_id}"
            rm -rf "$backup_path"
            ((deleted_count++))
            ((excess_count--))
            
            if [[ $excess_count -le 0 ]]; then
                break
            fi
        done < <(find "$BACKUP_DIR" -maxdepth 1 -type d -name "*_*" -printf '%T@ %p\n' | sort -n | cut -d' ' -f2-)
    fi
    
    log "Cleanup completed: ${deleted_count} backups deleted"
}

# Function to get backup status
get_backup_status() {
    echo -e "${BLUE}Eidolon Backup Status${NC}"
    echo "===================="
    
    # Check backup directory
    if [[ -d "$BACKUP_DIR" ]]; then
        local backup_count=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "*_*" | wc -l)
        local total_size=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1 || echo "0")
        
        echo "Backup directory: $BACKUP_DIR"
        echo "Total backups: $backup_count"
        echo "Total size: $total_size"
        
        # Most recent backup
        local latest_backup=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "*_*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
        if [[ -n "$latest_backup" ]]; then
            local latest_id=$(basename "$latest_backup")
            local latest_date=$(stat -c %y "$latest_backup" | cut -d'.' -f1)
            echo "Latest backup: $latest_id ($latest_date)"
        fi
    else
        echo "No backup directory found"
    fi
    
    echo ""
    
    # Data directory size
    if [[ -d "$DATA_DIR" ]]; then
        local data_size=$(du -sh "$DATA_DIR" | cut -f1)
        echo "Data directory size: $data_size"
    fi
    
    # Available disk space
    local available_space=$(df -h "$EIDOLON_HOME" | awk 'NR==2 {print $4}')
    echo "Available disk space: $available_space"
}

# Function to create backup metadata
create_backup_metadata() {
    local backup_path="$1"
    
    # Create checksums for verification
    find "$backup_path" -name "*.tar.gz" -exec sha256sum {} \; > "${backup_path}/checksums.txt"
    
    # Create backup info
    cat > "${backup_path}/backup_info.txt" <<EOF
Eidolon Backup Information
=========================

Backup ID: $(basename "$backup_path")
Created: $(date)
Host: $(hostname)
User: $(whoami)
Eidolon Version: $(get_eidolon_version)
Backup Script Version: 1.0
Python Version: $(python3 --version 2>/dev/null || echo "Unknown")
OS: $(uname -s -r)

Files:
$(ls -la "$backup_path")

Total Size: $(du -sh "$backup_path" | cut -f1)
EOF
}

# Function to get Eidolon version
get_eidolon_version() {
    if [[ -f "${EIDOLON_HOME}/venv/bin/python" ]]; then
        "${EIDOLON_HOME}/venv/bin/python" -c "import eidolon; print(eidolon.__version__)" 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

# Function to schedule automatic backups
schedule_backup() {
    local backup_type="${1:-data}"
    local frequency="${2:-daily}"
    
    log "Setting up automatic backup schedule"
    
    # Create cron job
    local cron_schedule
    case "$frequency" in
        "hourly") cron_schedule="0 * * * *" ;;
        "daily") cron_schedule="0 2 * * *" ;;
        "weekly") cron_schedule="0 2 * * 0" ;;
        *) 
            error "Invalid frequency: $frequency (use hourly, daily, or weekly)"
            return 1
            ;;
    esac
    
    local cron_command="${EIDOLON_HOME}/scripts/backup/backup-manager.sh backup ${backup_type}"
    local cron_entry="${cron_schedule} ${cron_command} > ${LOGS_DIR}/auto-backup.log 2>&1"
    
    # Add to crontab
    (crontab -l 2>/dev/null | grep -v "backup-manager.sh"; echo "$cron_entry") | crontab -
    
    log "Scheduled ${frequency} ${backup_type} backups"
}

# Function to show help
show_help() {
    echo "Eidolon Backup Manager"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  backup <type>           - Create backup (full, data, incremental)"
    echo "  restore <backup_id>     - Restore from backup"
    echo "  list [type]             - List backups (all, full, data, incremental)"
    echo "  delete <backup_id>      - Delete a backup"
    echo "  cleanup [days] [count]  - Clean up old backups"
    echo "  status                  - Show backup status"
    echo "  schedule <type> <freq>  - Schedule automatic backups"
    echo "  help                    - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 backup full"
    echo "  $0 backup data"
    echo "  $0 backup incremental 12"
    echo "  $0 restore full_20240720_143000"
    echo "  $0 list data"
    echo "  $0 cleanup 7 5"
    echo "  $0 schedule data daily"
    echo ""
    echo "Environment variables:"
    echo "  FORCE_DELETE=true       - Skip confirmation for deletions"
    echo ""
}

# Main execution
main() {
    # Ensure directories exist
    mkdir -p "$BACKUP_DIR" "$LOGS_DIR"
    
    # Check if Eidolon is installed
    if [[ ! -d "$EIDOLON_HOME" ]]; then
        error "Eidolon not found at $EIDOLON_HOME"
        exit 1
    fi
    
    local command="${1:-help}"
    
    case "$command" in
        "backup")
            local backup_type="${2:-full}"
            case "$backup_type" in
                "full")
                    create_full_backup "${3:-Manual full backup}"
                    ;;
                "data")
                    create_data_backup "${3:-Manual data backup}"
                    ;;
                "incremental")
                    create_incremental_backup "${3:-24}"
                    ;;
                *)
                    error "Invalid backup type: $backup_type"
                    show_help
                    exit 1
                    ;;
            esac
            ;;
        "restore")
            if [[ -z "${2:-}" ]]; then
                error "Backup ID required for restore"
                exit 1
            fi
            restore_backup "$2" "${3:-}"
            ;;
        "list")
            list_backups "${2:-all}"
            ;;
        "delete")
            if [[ -z "${2:-}" ]]; then
                error "Backup ID required for delete"
                exit 1
            fi
            delete_backup "$2"
            ;;
        "cleanup")
            cleanup_old_backups "${2:-$DEFAULT_RETENTION_DAYS}" "${3:-$DEFAULT_MAX_BACKUPS}"
            ;;
        "status")
            get_backup_status
            ;;
        "schedule")
            schedule_backup "${2:-data}" "${3:-daily}"
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Check dependencies
if ! command -v jq &> /dev/null; then
    warn "jq not found - some features may not work properly"
fi

# Run main function
main "$@"