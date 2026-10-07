from copy import deepcopy
from datetime import datetime, timedelta, timezone
import base64
import json
from uuid import uuid4

from sqlalchemy import select

from ..analysis.engine import DISCLAIMER
from ..models import Analysis


def cleanup_expired(session, storage, user_id=None):
    query = select(Analysis).where(Analysis.images_expire_at < datetime.now(timezone.utc))
    if user_id:
        query = query.where(Analysis.user_id == user_id)
    for row in session.scalars(query):
        for key in row.image_keys:
            storage.delete(key)
        result = deepcopy(row.result)
        for item in result["items"]:
            item["visualizations"] = {}
        row.result = result
        row.image_keys = []
        row.images_expire_at = None
    session.commit()


def report_payload(record):
    result = deepcopy(record.result)
    for item in result["items"]:
        item.pop("visualizations", None)
    return {"title": "SecureLens Analysis Report", "id": record.id,
            "analysis_type": record.analysis_type, "filename": record.filename,
            "timestamp": record.created_at.isoformat(), "result": result, "limitations": DISCLAIMER}


def save_record(db, storage, user, kind, result, retain_images, settings):
    identifier = str(uuid4())
    stored = deepcopy(result)
    keys = []
    report_key = f"reports/{identifier}.json"
    failures = sum(item["status"] == "failed" for item in stored["items"])
    status = "failed" if failures == len(stored["items"]) else "partial" if failures else "complete"
    try:
        for index, item in enumerate(stored["items"]):
            views = {}
            if retain_images:
                for view, data in item.get("visualizations", {}).items():
                    key = f"images/{identifier}/{index}-{view}.png"
                    storage.write(key, base64.b64decode(data.split(",", 1)[1]))
                    keys.append(key)
                    views[view] = f"/api/history/{identifier}/images/{index}/{view}"
            item["visualizations"] = views
        row = Analysis(id=identifier, user_id=user.id, analysis_type=kind, filename=stored["items"][0]["filename"],
                       result=stored, report_key=report_key, image_keys=keys,
                       created_at=datetime.now(timezone.utc),
                       status=status,
                       images_expire_at=datetime.now(timezone.utc) + timedelta(days=settings.image_retention_days) if keys else None)
        storage.write(report_key, json.dumps(report_payload(row), indent=2, allow_nan=False).encode())
        db.add(row)
        db.commit()
        return row
    except Exception:
        db.rollback()
        for key in keys + [report_key]:
            storage.delete(key)
        raise


def response_record(row):
    return {"id": row.id, "analysis_type": row.analysis_type, "filename": row.filename,
            "created_at": row.created_at.isoformat(), "saved": True, "status": row.status,
            "result": row.result, "images_expire_at": row.images_expire_at.isoformat() if row.images_expire_at else None}
