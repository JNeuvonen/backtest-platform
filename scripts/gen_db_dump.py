import subprocess
import sys
import os
from datetime import datetime


def dump_database(connection_string, output_file=None):
    try:
        db_name = connection_string.split("/")[-1]

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_file = f"{db_name}_backup_{timestamp}.sql"

        os.environ["PGPASSWORD"] = connection_string.split(":")[2].split("@")[0]

        dump_command = [
            "pg_dump",
            "-d",
            connection_string,
            "-F",
            "c",  # Custom format
            "-b",  # Include large objects
            "-v",  # Verbose mode
            "-f",
            output_file,  # Output file
        ]

        subprocess.run(dump_command, check=True)

        print(f"Database dump successful. Backup saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dump_database.py <connection_string> [<output_file>]")
        sys.exit(1)

    connection_string = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    dump_database(connection_string, output_file)
