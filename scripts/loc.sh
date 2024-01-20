#!/bin/bash
#fun script for counting lines of .py, .rs, .ts and .tsx
find . -type f \( -name '*.py' -o -name '*.rs' -o -name '*.ts' -o -name '*.tsx' \) -not -path '*/venv/*' -not -path '*/node_modules/*' -not -path '*/build/*' | xargs wc -l
