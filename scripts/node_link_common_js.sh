#!/bin/bash


link_common_js() {
    if [ -d "$1" ]; then
        echo "Changing directory to $1"
        cd "$1" || exit
        echo "Current directory: $(pwd)"
        npm link common_js
        cd - > /dev/null || exit
    else
        echo "Directory $1 does not exist. Skipping."
    fi
}

cd projects/common_js
npm run build
npm link
cd - > /dev/null

link_common_js "projects//backtest_platform/client"
link_common_js "projects/analytics_www"



