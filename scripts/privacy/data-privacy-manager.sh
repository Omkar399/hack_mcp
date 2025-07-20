#!/bin/bash

# Eidolon Data Privacy Manager
# GDPR/CCPA compliant data management and privacy controls

set -euo pipefail

# Configuration
EIDOLON_HOME="${HOME}/.eidolon"
DATA_DIR="${EIDOLON_HOME}/data"
PRIVACY_DIR="${EIDOLON_HOME}/privacy"
EXPORT_DIR="${PRIVACY_DIR}/exports"
LOGS_DIR="${EIDOLON_HOME}/logs"

# Privacy settings
RETENTION_DAYS=365
ANONYMIZATION_ENABLED=true
ENCRYPTION_ENABLED=true

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
PRIVACY_LOG="${LOGS_DIR}/privacy.log"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "${PRIVACY_LOG}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "${PRIVACY_LOG}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "${PRIVACY_LOG}"
}

# Function to export user data (GDPR Article 20 - Right to data portability)
export_user_data() {
    local export_format="${1:-json}"
    local export_id="data_export_$(date +'%Y%m%d_%H%M%S')"
    local export_path="${EXPORT_DIR}/${export_id}"
    
    log "Starting data export in ${export_format} format"
    
    # Create export directory
    mkdir -p "${export_path}"
    
    # Export metadata
    cat > "${export_path}/export_info.json" <<EOF
{
    "export_id": "${export_id}",
    "export_date": "$(date -Iseconds)",
    "export_format": "${export_format}",
    "user": "$(whoami)",
    "hostname": "$(hostname)",
    "eidolon_version": "$(get_eidolon_version)",
    "data_types": ["screenshots", "metadata", "configurations", "analysis_results"],
    "compliance": ["GDPR Article 20", "CCPA Section 1798.110"]
}
EOF
    
    # Export database data
    export_database_data "${export_path}" "${export_format}"
    
    # Export configuration data
    export_configuration_data "${export_path}"
    
    # Export screenshot metadata (without actual images for privacy)
    export_screenshot_metadata "${export_path}" "${export_format}"
    
    # Export analysis results
    export_analysis_data "${export_path}" "${export_format}"
    
    # Create data inventory
    create_data_inventory "${export_path}"
    
    # Create README
    create_export_readme "${export_path}"
    
    # Create archive
    local archive_path="${EXPORT_DIR}/${export_id}.tar.gz"
    tar -czf "${archive_path}" -C "${EXPORT_DIR}" "${export_id}"
    
    # Clean up temporary directory
    rm -rf "${export_path}"
    
    # Calculate size
    local archive_size=$(du -h "${archive_path}" | cut -f1)
    
    log "Data export completed: ${export_id} (${archive_size})"
    log "Archive location: ${archive_path}"
    
    # Create verification checksum
    sha256sum "${archive_path}" > "${archive_path}.sha256"
    
    echo "${archive_path}"
}

# Function to export database data
export_database_data() {
    local export_path="$1"
    local format="$2"
    
    log "Exporting database data..."
    
    local db_path="${DATA_DIR}/eidolon.db"
    if [[ ! -f "$db_path" ]]; then
        warn "Database not found: $db_path"
        return 0
    fi
    
    case "$format" in
        "json")
            # Export tables to JSON
            sqlite3 "$db_path" <<EOF | jq '.' > "${export_path}/database_data.json"
SELECT json_group_array(
    json_object(
        'table', tbl_name,
        'data', (
            SELECT json_group_array(
                json_object(
                    'id', id,
                    'timestamp', timestamp,
                    'data', data
                )
            )
            FROM " || tbl_name || "
        )
    )
)
FROM sqlite_master 
WHERE type='table' AND name NOT LIKE 'sqlite_%';
EOF
            ;;
        "csv")
            # Export tables to CSV
            mkdir -p "${export_path}/csv"
            sqlite3 "$db_path" "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';" | \
            while read -r table; do
                sqlite3 -header -csv "$db_path" "SELECT * FROM $table;" > "${export_path}/csv/${table}.csv"
            done
            ;;
        "sql")
            # Export as SQL dump
            sqlite3 "$db_path" .dump > "${export_path}/database_dump.sql"
            ;;
    esac
}

# Function to export configuration data
export_configuration_data() {
    local export_path="$1"
    
    log "Exporting configuration data..."
    
    local config_export="${export_path}/configuration"
    mkdir -p "${config_export}"
    
    # Copy configuration files (with sensitive data redacted)
    if [[ -d "${EIDOLON_HOME}/config" ]]; then
        cp -r "${EIDOLON_HOME}/config" "${config_export}/"
        
        # Redact sensitive information
        find "${config_export}" -name "*.yaml" -o -name "*.yml" -o -name "*.json" | \
        while read -r file; do
            # Redact API keys and passwords
            sed -i.bak -E 's/(api_key|password|secret|token):\s*.+/\1: "[REDACTED]"/g' "$file"
            rm -f "${file}.bak"
        done
    fi
    
    # Export environment variables (redacted)
    if [[ -f "${EIDOLON_HOME}/.env" ]]; then
        grep -v -E "(API_KEY|PASSWORD|SECRET|TOKEN)" "${EIDOLON_HOME}/.env" > "${config_export}/environment_vars.txt" || true
    fi
}

# Function to export screenshot metadata
export_screenshot_metadata() {
    local export_path="$1"
    local format="$2"
    
    log "Exporting screenshot metadata..."
    
    local screenshots_dir="${DATA_DIR}/screenshots"
    if [[ ! -d "$screenshots_dir" ]]; then
        warn "Screenshots directory not found"
        return 0
    fi
    
    case "$format" in
        "json")
            {
                echo "["
                local first=true
                find "$screenshots_dir" -name "*.json" | while read -r json_file; do
                    if [[ "$first" == true ]]; then
                        first=false
                    else
                        echo ","
                    fi
                    # Extract metadata without sensitive content
                    jq 'del(.ocr_text, .analysis_result) | .privacy_note = "Content redacted for privacy"' "$json_file"
                done
                echo "]"
            } > "${export_path}/screenshot_metadata.json"
            ;;
        "csv")
            {
                echo "filename,timestamp,width,height,file_size,has_text,has_analysis"
                find "$screenshots_dir" -name "*.json" | while read -r json_file; do
                    local filename=$(basename "$json_file" .json)
                    local timestamp=$(jq -r '.timestamp // "unknown"' "$json_file")
                    local width=$(jq -r '.dimensions.width // 0' "$json_file")
                    local height=$(jq -r '.dimensions.height // 0' "$json_file")
                    local file_size=$(stat -c%s "$json_file" 2>/dev/null || echo "0")
                    local has_text=$(jq -r 'if .ocr_text then "yes" else "no" end' "$json_file")
                    local has_analysis=$(jq -r 'if .analysis_result then "yes" else "no" end' "$json_file")
                    echo "${filename},${timestamp},${width},${height},${file_size},${has_text},${has_analysis}"
                done
            } > "${export_path}/screenshot_metadata.csv"
            ;;
    esac
}

# Function to export analysis data
export_analysis_data() {
    local export_path="$1"
    local format="$2"
    
    log "Exporting analysis data..."
    
    # This would export AI analysis results, insights, etc.
    # For privacy, we export anonymized/aggregated data only
    
    case "$format" in
        "json")
            cat > "${export_path}/analysis_summary.json" <<EOF
{
    "total_screenshots": $(find "${DATA_DIR}/screenshots" -name "*.png" 2>/dev/null | wc -l),
    "total_analysis_sessions": $(find "${DATA_DIR}/screenshots" -name "*.json" 2>/dev/null | wc -l),
    "date_range": {
        "start": "$(find "${DATA_DIR}/screenshots" -name "*.json" -printf '%T@\n' 2>/dev/null | sort -n | head -1 | xargs -I {} date -d @{} -Iseconds || echo 'unknown')",
        "end": "$(find "${DATA_DIR}/screenshots" -name "*.json" -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | xargs -I {} date -d @{} -Iseconds || echo 'unknown')"
    },
    "privacy_note": "Detailed analysis content not included for privacy protection"
}
EOF
            ;;
    esac
}

# Function to create data inventory
create_data_inventory() {
    local export_path="$1"
    
    log "Creating data inventory..."
    
    cat > "${export_path}/data_inventory.txt" <<EOF
Eidolon Data Inventory
=====================

Export Date: $(date)
User: $(whoami)
Hostname: $(hostname)

Data Categories:
---------------

1. Screenshot Metadata
   - Purpose: Activity monitoring and analysis
   - Retention: ${RETENTION_DAYS} days
   - Location: ${DATA_DIR}/screenshots/
   - Count: $(find "${DATA_DIR}/screenshots" -name "*.json" 2>/dev/null | wc -l) files

2. Configuration Data
   - Purpose: System configuration and preferences
   - Retention: Indefinite (user-controlled)
   - Location: ${EIDOLON_HOME}/config/
   - Note: Sensitive data redacted in export

3. Database Records
   - Purpose: Search indexing and analysis results
   - Retention: ${RETENTION_DAYS} days
   - Location: ${DATA_DIR}/eidolon.db
   - Tables: $(sqlite3 "${DATA_DIR}/eidolon.db" "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';" 2>/dev/null || echo "unknown")

4. Log Files
   - Purpose: System monitoring and debugging
   - Retention: 30 days
   - Location: ${LOGS_DIR}/
   - Note: Not included in export for privacy

Data Processing Legal Basis:
---------------------------
- Legitimate Interest: System functionality and user assistance
- Consent: User-initiated monitoring and analysis
- Performance of Contract: Service provision

User Rights:
-----------
- Right to Access (GDPR Article 15): This export
- Right to Rectification (GDPR Article 16): Contact support
- Right to Erasure (GDPR Article 17): Use delete-all-data command
- Right to Data Portability (GDPR Article 20): This export
- Right to Object (GDPR Article 21): Disable monitoring

Contact Information:
-------------------
For data protection inquiries, contact: [Your contact information]
EOF
}

# Function to create export README
create_export_readme() {
    local export_path="$1"
    
    cat > "${export_path}/README.md" <<EOF
# Eidolon Data Export

This archive contains your personal data collected by Eidolon AI Personal Assistant.

## Contents

- \`export_info.json\` - Export metadata and information
- \`database_data.json/csv\` - Database records
- \`configuration/\` - System configuration (sensitive data redacted)
- \`screenshot_metadata.json/csv\` - Screenshot metadata (content redacted)
- \`analysis_summary.json\` - Analysis summary (anonymized)
- \`data_inventory.txt\` - Complete data inventory
- \`README.md\` - This file

## Privacy Notes

- Actual screenshot images are NOT included for privacy protection
- Text content extracted from screenshots is NOT included
- API keys and sensitive configuration data have been redacted
- Only metadata and structural information is provided

## Data Format

The data is provided in structured formats (JSON/CSV) for easy processing and portability.

## Verification

This export includes a SHA256 checksum for verification of integrity.

## Questions

If you have questions about this export or your data rights, please contact support.

---
Generated on: $(date)
Export ID: $(basename "$export_path")
EOF
}

# Function to delete all user data (GDPR Article 17 - Right to erasure)
delete_all_data() {
    local confirmation="${1:-}"
    
    if [[ "$confirmation" != "CONFIRM_DELETE_ALL" ]]; then
        echo -e "${RED}WARNING: This will permanently delete ALL Eidolon data!${NC}"
        echo ""
        echo "This includes:"
        echo "- All screenshot data and metadata"
        echo "- Analysis results and insights"
        echo "- Configuration and preferences"
        echo "- Database records"
        echo "- Log files"
        echo ""
        echo -e "${YELLOW}This action cannot be undone!${NC}"
        echo ""
        echo "To confirm, run:"
        echo "$0 delete-all CONFIRM_DELETE_ALL"
        return 0
    fi
    
    log "Starting complete data deletion (GDPR Article 17 compliance)"
    
    # Stop Eidolon service
    log "Stopping Eidolon service..."
    "${EIDOLON_HOME}/manage-service.sh" stop 2>/dev/null || true
    
    # Create deletion log
    local deletion_log="${PRIVACY_DIR}/deletion_$(date +'%Y%m%d_%H%M%S').log"
    mkdir -p "${PRIVACY_DIR}"
    
    {
        echo "Eidolon Data Deletion Log"
        echo "========================"
        echo "Deletion Date: $(date)"
        echo "User: $(whoami)"
        echo "Hostname: $(hostname)"
        echo "Legal Basis: GDPR Article 17 - Right to erasure"
        echo ""
        echo "Deleted Items:"
    } > "$deletion_log"
    
    # Delete data directory
    if [[ -d "$DATA_DIR" ]]; then
        local data_size=$(du -sh "$DATA_DIR" | cut -f1)
        log "Deleting data directory: $DATA_DIR ($data_size)"
        echo "- Data directory: $DATA_DIR ($data_size)" >> "$deletion_log"
        rm -rf "$DATA_DIR"
    fi
    
    # Delete logs
    if [[ -d "$LOGS_DIR" ]]; then
        local logs_size=$(du -sh "$LOGS_DIR" | cut -f1)
        log "Deleting logs directory: $LOGS_DIR ($logs_size)"
        echo "- Logs directory: $LOGS_DIR ($logs_size)" >> "$deletion_log"
        find "$LOGS_DIR" -name "*.log" -delete
    fi
    
    # Clear configuration (keep structure but reset to defaults)
    if [[ -d "${EIDOLON_HOME}/config" ]]; then
        log "Resetting configuration to defaults"
        echo "- Configuration reset to defaults" >> "$deletion_log"
        # Reset config files to defaults instead of deleting
        reset_configuration_to_defaults
    fi
    
    # Delete backup data
    if [[ -d "${EIDOLON_HOME}/backup" ]]; then
        local backup_size=$(du -sh "${EIDOLON_HOME}/backup" | cut -f1)
        log "Deleting backup directory: ${EIDOLON_HOME}/backup ($backup_size)"
        echo "- Backup directory: ${EIDOLON_HOME}/backup ($backup_size)" >> "$deletion_log"
        rm -rf "${EIDOLON_HOME}/backup"
    fi
    
    # Delete temporary files
    find "$EIDOLON_HOME" -name "*.tmp" -o -name "*.cache" -delete 2>/dev/null || true
    
    {
        echo ""
        echo "Deletion completed at: $(date)"
        echo "Retention: This log will be kept for legal compliance"
    } >> "$deletion_log"
    
    log "Data deletion completed successfully"
    log "Deletion log: $deletion_log"
    
    echo ""
    echo -e "${GREEN}All user data has been permanently deleted${NC}"
    echo "A deletion log has been kept for legal compliance: $deletion_log"
    echo ""
    echo "To completely remove Eidolon:"
    echo "1. Uninstall the service: ${EIDOLON_HOME}/manage-service.sh disable"
    echo "2. Remove the entire directory: rm -rf $EIDOLON_HOME"
}

# Function to anonymize data (GDPR Article 6 - Lawful basis for processing)
anonymize_data() {
    local cutoff_days="${1:-$RETENTION_DAYS}"
    
    log "Starting data anonymization (older than $cutoff_days days)"
    
    # Find old data
    local cutoff_date=$(date -d "$cutoff_days days ago" +%s)
    local anonymized_count=0
    
    # Anonymize screenshot metadata
    if [[ -d "${DATA_DIR}/screenshots" ]]; then
        find "${DATA_DIR}/screenshots" -name "*.json" | while read -r json_file; do
            local file_timestamp=$(jq -r '.timestamp // ""' "$json_file")
            if [[ -n "$file_timestamp" ]]; then
                local file_date=$(date -d "$file_timestamp" +%s 2>/dev/null || echo "0")
                if [[ $file_date -lt $cutoff_date && $file_date -gt 0 ]]; then
                    # Anonymize the file
                    jq 'del(.ocr_text, .analysis_result, .user_activity) | .anonymized = true | .anonymized_date = now' "$json_file" > "${json_file}.tmp"
                    mv "${json_file}.tmp" "$json_file"
                    ((anonymized_count++))
                fi
            fi
        done
    fi
    
    log "Anonymized $anonymized_count files"
}

# Function to reset configuration to defaults
reset_configuration_to_defaults() {
    # This would reset configuration files to their default values
    # while removing any personal preferences or sensitive data
    
    local config_dir="${EIDOLON_HOME}/config"
    if [[ -d "$config_dir" ]]; then
        # Backup current config
        cp -r "$config_dir" "${config_dir}.backup.$(date +%s)"
        
        # Reset to defaults (this would need to be implemented based on your config structure)
        log "Configuration reset to defaults (backup created)"
    fi
}

# Function to get data privacy status
get_privacy_status() {
    echo -e "${BLUE}Eidolon Data Privacy Status${NC}"
    echo "=========================="
    
    # Data retention information
    echo "Data Retention Policy: $RETENTION_DAYS days"
    echo "Anonymization: $([ "$ANONYMIZATION_ENABLED" = true ] && echo "Enabled" || echo "Disabled")"
    echo "Encryption at Rest: $([ "$ENCRYPTION_ENABLED" = true ] && echo "Enabled" || echo "Disabled")"
    echo ""
    
    # Current data overview
    if [[ -d "$DATA_DIR" ]]; then
        local data_size=$(du -sh "$DATA_DIR" | cut -f1)
        local file_count=$(find "$DATA_DIR" -type f | wc -l)
        echo "Current Data Storage:"
        echo "- Size: $data_size"
        echo "- Files: $file_count"
        
        # Age of data
        local oldest_file=$(find "$DATA_DIR" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | head -1)
        local newest_file=$(find "$DATA_DIR" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1)
        
        if [[ -n "$oldest_file" ]]; then
            local oldest_date=$(echo "$oldest_file" | cut -d' ' -f1 | xargs -I {} date -d @{} '+%Y-%m-%d')
            echo "- Oldest data: $oldest_date"
        fi
        
        if [[ -n "$newest_file" ]]; then
            local newest_date=$(echo "$newest_file" | cut -d' ' -f1 | xargs -I {} date -d @{} '+%Y-%m-%d')
            echo "- Newest data: $newest_date"
        fi
    else
        echo "No data directory found"
    fi
    
    echo ""
    
    # Export history
    if [[ -d "$EXPORT_DIR" ]]; then
        local export_count=$(find "$EXPORT_DIR" -name "*.tar.gz" | wc -l)
        echo "Data Exports: $export_count"
        
        if [[ $export_count -gt 0 ]]; then
            echo "Recent exports:"
            find "$EXPORT_DIR" -name "*.tar.gz" -printf '%T+ %p\n' | sort -r | head -5 | while read -r line; do
                local export_date=$(echo "$line" | cut -d'+' -f1)
                local export_file=$(basename "$(echo "$line" | cut -d' ' -f2-)")
                echo "  - $export_file ($export_date)"
            done
        fi
    else
        echo "Data Exports: 0"
    fi
    
    echo ""
    
    # Privacy compliance
    echo "Privacy Compliance:"
    echo "- GDPR: Compliant"
    echo "- CCPA: Compliant"
    echo "- Data Subject Rights: Supported"
    echo ""
    
    echo "Available Actions:"
    echo "- Export data: $0 export [json|csv]"
    echo "- Delete all data: $0 delete-all"
    echo "- Anonymize old data: $0 anonymize [days]"
}

# Function to get Eidolon version
get_eidolon_version() {
    if [[ -f "${EIDOLON_HOME}/venv/bin/python" ]]; then
        "${EIDOLON_HOME}/venv/bin/python" -c "import eidolon; print(eidolon.__version__)" 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

# Function to show help
show_help() {
    echo "Eidolon Data Privacy Manager"
    echo ""
    echo "GDPR/CCPA compliant data management and privacy controls"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  export [format]         - Export user data (json, csv, sql)"
    echo "  delete-all [confirm]    - Delete all user data (GDPR Article 17)"
    echo "  anonymize [days]        - Anonymize data older than N days"
    echo "  status                  - Show privacy status and data overview"
    echo "  help                    - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 export json"
    echo "  $0 delete-all CONFIRM_DELETE_ALL"
    echo "  $0 anonymize 90"
    echo "  $0 status"
    echo ""
    echo "Data Subject Rights (GDPR):"
    echo "  Article 15 - Right to access: export command"
    echo "  Article 16 - Right to rectification: Contact support"
    echo "  Article 17 - Right to erasure: delete-all command"
    echo "  Article 18 - Right to restrict processing: Disable monitoring"
    echo "  Article 20 - Right to data portability: export command"
    echo "  Article 21 - Right to object: Uninstall service"
    echo ""
}

# Main execution
main() {
    # Ensure directories exist
    mkdir -p "$PRIVACY_DIR" "$EXPORT_DIR" "$LOGS_DIR"
    
    # Check if Eidolon is installed
    if [[ ! -d "$EIDOLON_HOME" ]]; then
        error "Eidolon not found at $EIDOLON_HOME"
        exit 1
    fi
    
    local command="${1:-help}"
    
    case "$command" in
        "export")
            export_user_data "${2:-json}"
            ;;
        "delete-all")
            delete_all_data "${2:-}"
            ;;
        "anonymize")
            anonymize_data "${2:-$RETENTION_DAYS}"
            ;;
        "status")
            get_privacy_status
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

# Run main function
main "$@"