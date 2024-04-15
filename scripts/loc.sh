#!/bin/bash

py_file_count=$(find . -type f -name '*.py' -not -path '*/venv/*' -not -path '*/venv_pred_server/*' -not -path '*/node_modules/*' -not -path '*/build/*' -not -path "*/.mypy_cache/*" -not -path "*/binaries/*" | wc -l)
ts_file_count=$(find . -type f \( -name '*.ts' -o -name '*.tsx' \) -not -path '*/node_modules/*' -not -path '*/thirdparty/*' -not -path '*/build/*' -not -path "*/binaries/*" | wc -l)
go_file_count=$(find . -type f -name '*.go' -not -path '*/venv/*' -not -path '*/thirdparty/*' -not -path '*/node_modules/*' -not -path '*/.mypy_cache/*' -not -path '*/binaries/*' -not -path '*/build/*' | wc -l) 
total_file_count=$((py_file_count + ts_file_count + go_file_count))

py_lines=$(find . -type f -name '*.py' -not -path '*/venv/*' -not -path '*/venv_pred_server/*' -not -path '*/node_modules/*' -not -path '*/build/*' -not -path "*/.mypy_cache/*" -not -path "*/binaries/*" | xargs cat | wc -l)
ts_lines=$(find . -type f \( -name '*.ts' -o -name '*.tsx' \) -not -path '*/node_modules/*' -not -path '*/thirdparty/*' -not -path '*/build/*' -not -path "*/binaries/*" | xargs cat | wc -l)
go_lines=$(find . -type f -name '*.go' -not -path '*/venv/*' -not -path '*/node_modules/*' -not -path '*/thirdparty/*' -not -path '*/.mypy_cache/*' -not -path '*/binaries/*' -not -path '*/build/*' | xargs cat | wc -l) 
total_line_count=$((py_lines + ts_lines + go_lines))

echo "Line Counts:"
echo "------------"
echo "Python line count: $py_lines"
echo "TypeScript line count: $ts_lines"
echo "Go line count: $go_lines"
echo "Total line count: $total_line_count"

echo ""
echo "File Counts:"
echo "------------"
echo "Python file count: $py_file_count"
echo "TypeScript file count: $ts_file_count"
echo "Go file count: $go_file_count"
echo "Total file count: $total_file_count"


# Prepare the message
message="Line Counts:\n------------\nPython line count: $py_lines\nTypeScript line count: $ts_lines\nGo line count: $go_lines\nTotal line count: $total_line_count\n\nFile Counts:\n------------\nPython file count: $py_file_count\nTypeScript file count: $ts_file_count\nGo file count: $go_file_count\nTotal file count: $total_file_count"

# Your Slack webhook URL
webhook_url='https://hooks.slack.com/services/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'

# Use curl to send the message as a JSON payload to the Slack webhook
curl -X POST -H 'Content-Type: application/json' --data "{\"text\": \"${message//\"/\\\"}\"}" $webhook_url


