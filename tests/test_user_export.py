from contextlib import nullcontext
from datetime import datetime, timezone
import json
import stat

import pytest

from tools.export_user_activity import export_activity


class ReadOnlyConnection:
    def __init__(self):
        self.queries = []

    def transaction(self):
        return nullcontext()

    def execute(self, query):
        self.queries.append(query)
        return self

    def fetchall(self):
        if "FROM users" in self.queries[-1]:
            return [{"id": "test-user", "name": "Test", "email": "test@example.com",
                     "created_at": datetime(2026, 10, 10, tzinfo=timezone.utc),
                     "saved_analyses": 1, "last_analysis_at": None}]
        return [{"id": "test-analysis", "user_id": "test-user", "filename": "fixture.png"}]


def test_export_is_read_only_private_and_excludes_password_columns(tmp_path):
    connection = ReadOnlyConnection()
    output = tmp_path / "private" / "activity.json"
    assert export_activity(connection, output) == (1, 1)
    assert connection.queries[0] == "SET TRANSACTION READ ONLY"
    assert all("password" not in query.lower() for query in connection.queries)
    payload = json.loads(output.read_text())
    assert payload["users"][0]["created_at"] == "2026-10-10T00:00:00+00:00"
    assert "Login events are not recorded" in payload["note"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_export_does_not_overwrite_existing_data(tmp_path):
    output = tmp_path / "activity.json"
    output.write_text("preserve me")
    with pytest.raises(FileExistsError):
        export_activity(ReadOnlyConnection(), output)
    assert output.read_text() == "preserve me"
