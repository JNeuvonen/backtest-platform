import subprocess
import sys
import os


def run_command(command, env=None):
    """Helper function to run a shell command."""
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Command '{' '.join(command)}' failed with error: {e}")
        sys.exit(1)


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
        env = os.environ.copy()
        env["PGPASSWORD"] = password

        # Terminate all connections to the database
        terminate_connections_command = [
            "psql",
            "-h",
            host,
            "-U",
            user,
            "-p",
            port,
            "-c",
            f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{db_name}'
              AND pid <> pg_backend_pid();
            """,
        ]
        run_command(terminate_connections_command, env)

        # Drop the existing database
        drop_command = [
            "psql",
            "-h",
            host,
            "-U",
            user,
            "-p",
            port,
            "-c",
            f"DROP DATABASE IF EXISTS {db_name};",
        ]
        run_command(drop_command, env)

        # Create a new database
        create_command = [
            "psql",
            "-h",
            host,
            "-U",
            user,
            "-p",
            port,
            "-c",
            f"CREATE DATABASE {db_name};",
        ]
        run_command(create_command, env)

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
        run_command(restore_command, env)

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
