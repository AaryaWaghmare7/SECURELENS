"""Export account metadata and saved-analysis activity, never password hashes."""
import argparse
from datetime import date, datetime
from getpass import getpass
import json
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


def export_activity(connection, output):
    # This connection is administrative, but the entire export is read-only.
    with connection.transaction():
        connection.execute("SET TRANSACTION READ ONLY")
        users = connection.execute("""
            SELECT u.id, u.name, u.email, u.created_at,
                   COUNT(a.id) AS saved_analyses, MAX(a.created_at) AS last_analysis_at
            FROM users u LEFT JOIN analyses a ON a.user_id = u.id
            GROUP BY u.id, u.name, u.email, u.created_at
            ORDER BY u.created_at DESC
        """).fetchall()
        analyses = connection.execute("""
            SELECT id, user_id, filename, analysis_type, status, created_at
            FROM analyses ORDER BY created_at DESC
        """).fetchall()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"users": users, "analyses": analyses,
                   "note": "Login events are not recorded by the current application."},
                  handle, indent=2, default=serialize)
        handle.write("\n")
    return len(users), len(analyses)


def serialize(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError("Unsupported export value")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Prompting avoids putting database credentials in shell history or arguments.
    url = getpass("Render External Database URL (hidden): ").strip()
    url = url.replace("postgresql+psycopg://", "postgresql://", 1)
    try:
        with psycopg.connect(url, connect_timeout=10, row_factory=dict_row,
                             sslmode="require",
                             options="-c default_transaction_read_only=on -c statement_timeout=10000") as connection:
            users, analyses = export_activity(connection, args.output)
    except Exception as error:
        # Connection exceptions may contain credentials; only expose their type.
        print(f"Export failed ({type(error).__name__}). No credentials were displayed.")
        return 1
    print(f"Private export saved: {args.output} ({users} users, {analyses} saved analyses).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
