
#!/bin/bash


IP_TO_WHITELIST="$1"

if [ -z "$IP_TO_WHITELIST" ]; then
    echo "Usage: $0 <IP_TO_WHITELIST>"
    exit 1
fi

DATABASE_URI="postgresql://username:password@localhost/pred_server_integration_tests"

SQL_COMMAND="INSERT INTO whitelisted_ip (ip) VALUES ('$IP_TO_WHITELIST') ON CONFLICT (ip) DO NOTHING;"

# Execute SQL Command
PGPASSWORD=${DATABASE_URI##*:} psql -h localhost -U ${DATABASE_URI##*//} -d ${DATABASE_URI##*@} -c "$SQL_COMMAND"
