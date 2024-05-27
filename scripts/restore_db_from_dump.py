import subprocess
import sys
import os


def restore_database(dump_file, connection_string):
    try:
        # Extract the database name from the connection string
        db_name = connection_string.split("/")[-1]

        # Extract connection parameters
        user = connection_string.split("//")[1].split(":")[0]
        password = connection_string.split(":")[2].split("@")[0]
        host = connection_string.split("@")[1].split(":")[0]
        port = connection_string.split(":")[3].split("/")[0]

        # Set the environment variable for the password
        os.environ["PGPASSWORD"] = password

        # Determine the file format (custom, tar, plain)
        if dump_file.endswith(".sql"):
            # Plain text format, use psql to restore
            restore_command = [
                "psql",
                "-h",
                host,
                "-U",
                user,
                "-p",
                port,
                "-d",
                db_name,
                "-f",
                dump_file,
            ]
        else:
            # Custom or tar format, use pg_restore
            restore_command = [
                "pg_restore",
                "-h",
                host,
                "-U",
                user,
                "-p",
                port,
                "-d",
                db_name,
                "-v",  # Verbose mode
                dump_file,
            ]

        # Execute the restore command
        subprocess.run(restore_command, check=True)

        print(f"Database restoration successful from {dump_file} to {db_name}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python restore_database.py <connection_string> <dump_file_path>")
        sys.exit(1)

    connection_string = sys.argv[1]
    dump_file = sys.argv[2]

    restore_database(dump_file, connection_string)
