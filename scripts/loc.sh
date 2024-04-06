#!/bin/bash

py_file_count=$(find . -type f -name '*.py' -not -path '*/venv/*' -not -path '*/venv_pred_server/*' -not -path '*/node_modules/*' -not -path '*/build/*' -not -path "*/.mypy_cache/*" -not -path "*/binaries/*" | wc -l)
ts_file_count=$(find . -type f \( -name '*.ts' -o -name '*.tsx' \) -not -path '*/node_modules/*' -not -path '*/build/*' -not -path "*/binaries/*" | wc -l)
go_file_count=$(find . -type f -name '*.go' -not -path '*/venv/*' -not -path '*/thirdparty/*' -not -path '*/node_modules/*' -not -path '*/.mypy_cache/*' -not -path '*/binaries/*' -not -path '*/build/*' | wc -l) 
total_file_count=$((py_file_count + ts_file_count + go_file_count))

py_lines=$(find . -type f -name '*.py' -not -path '*/venv/*' -not -path '*/venv_pred_server/*' -not -path '*/node_modules/*' -not -path '*/build/*' -not -path "*/.mypy_cache/*" -not -path "*/binaries/*" | xargs cat | wc -l)
ts_lines=$(find . -type f \( -name '*.ts' -o -name '*.tsx' \) -not -path '*/node_modules/*' -not -path '*/build/*' -not -path "*/binaries/*" | xargs cat | wc -l)
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



