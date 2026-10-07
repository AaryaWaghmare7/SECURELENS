from datetime import datetime, timezone
from pathlib import PurePath
import re

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
import json
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from PIL import Image
from io import BytesIO

from ..analysis.engine import run_analysis, compare_results
from ..auth.security import current_user
from ..database.session import get_db
from ..models import User, UserPreference
from ..services.records import save_record, response_record
from src.image_utils import ImageAnalysisError

router = APIRouter(prefix="/api/analyze", tags=["analysis"])
ALLOWED_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


async def validated_upload(upload, settings):
    try:
        name = (upload.filename or "image").replace("\\", "/").split("/")[-1]
        suffix = PurePath(name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS or upload.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, "Upload a JPG, JPEG or PNG. WEBP is not yet supported by the shared engine.")
        data = await upload.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "Each image must be smaller than 16 MB.")
        if not data:
            raise HTTPException(422, "The image is empty.")
        try:
            with Image.open(BytesIO(data)) as image:
                if image.format != ALLOWED_TYPES[upload.content_type] or (suffix == ".png") != (image.format == "PNG"):
                    raise HTTPException(415, "The file contents do not match its extension and MIME type.")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(422, "This image is corrupt or could not be decoded.")
        name = re.sub(r"[^\w. -]", "_", name, flags=re.UNICODE)[:180] or "image" + suffix
        return data, name
    finally:
        await upload.close()


def persist_or_ephemeral(request, db, user, kind, result, save, retain_images):
    if save:
        row = save_record(db, request.app.state.storage, user, kind, result, retain_images, request.app.state.settings)
        response = response_record(row)
        # Return in-memory views for the current screen even when retention is off.
        response["result"] = result
        return response
    return {"id": None, "analysis_type": kind, "filename": result["items"][0]["filename"],
            "created_at": datetime.now(timezone.utc).isoformat(), "saved": False, "result": result}


async def compute(request, data, name):
    try:
        async with request.app.state.analysis_slots:
            return await run_in_threadpool(run_analysis, data, name)
    except ImageAnalysisError as error:
        raise HTTPException(422, str(error))


@router.post("")
async def analyze(request: Request, image: UploadFile = File(...), save: bool = Form(False),
                  retain_images: bool = Form(False), capture_source: str = Form("upload"),
                  db: Session = Depends(get_db), user: User = Depends(current_user)):
    if capture_source not in ("upload", "webcam"):
        raise HTTPException(422, "Unknown capture source.")
    data, name = await validated_upload(image, request.app.state.settings)
    item = await compute(request, data, name)
    item["capture_source"] = capture_source
    # Camera origin does not establish authenticity or force a REAL result.
    kind = "live" if capture_source == "webcam" else "single"
    return persist_or_ephemeral(request, db, user, kind, {"items": [item]}, save, retain_images)


@router.post("/batch")
async def batch(request: Request, images: list[UploadFile] = File(...), save: bool = Form(False),
                retain_images: bool = Form(False), stream: bool = Query(False),
                db: Session = Depends(get_db), user: User = Depends(current_user)):
    if not 1 <= len(images) <= request.app.state.settings.max_batch_images:
        for image in images:
            await image.close()
        raise HTTPException(422, "Choose between 1 and 10 images per batch.")
    async def events():
        items = []
        try:
            for image in images:
                name = re.sub(r"[^\w. -]", "_", (image.filename or "image").replace("\\", "/").split("/")[-1])[:180]
                try:
                    data, name = await validated_upload(image, request.app.state.settings)
                    item = await compute(request, data, name)
                except HTTPException as error:
                    item = {"filename": name, "status": "failed", "error": error.detail}
                items.append(item)
                yield {"completed": len(items), "total": len(images), "item": item}
            yield {"record": persist_or_ephemeral(request, db, user, "batch", {"items": items}, save, retain_images)}
        finally:
            for image in images:
                await image.close()
    if stream:
        async def stream_events():
            try:
                async for event in events():
                    yield "data: " + json.dumps(event, allow_nan=False) + "\n\n"
            except Exception:
                yield 'data: {"error":"Batch processing could not finish. Please retry."}\n\n'
        return StreamingResponse(stream_events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})
    response = None
    async for event in events():
        if "record" in event:
            response = event["record"]
    return response


@router.post("/compare")
async def compare(request: Request, image_a: UploadFile = File(...), image_b: UploadFile = File(...),
                  save: bool = Form(False), retain_images: bool = Form(False),
                  db: Session = Depends(get_db), user: User = Depends(current_user)):
    try:
        data_a, name_a = await validated_upload(image_a, request.app.state.settings)
        data_b, name_b = await validated_upload(image_b, request.app.state.settings)
    finally:
        await image_a.close()
        await image_b.close()
    a, b = await compute(request, data_a, name_a), await compute(request, data_b, name_b)
    comparison = await run_in_threadpool(compare_results, a, b)
    return persist_or_ephemeral(request, db, user, "compare", {"items": [a, b], "comparison": comparison}, save, retain_images)
