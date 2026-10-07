import csv
from io import StringIO, BytesIO

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Query
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from ..auth.security import current_user
from ..database.session import get_db
from ..models import User, Analysis
from ..services.records import response_record, cleanup_expired, report_payload

router = APIRouter(prefix="/api", tags=["history and reports"])


def owned_record(db, user, identifier):
    row = db.scalar(select(Analysis).where(Analysis.id == identifier, Analysis.user_id == user.id))
    if row is None:
        raise HTTPException(404, "Analysis not found.")
    return row


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db), user: User = Depends(current_user)):
    counts = dict(db.execute(select(Analysis.analysis_type, func.count()).where(
        Analysis.user_id == user.id).group_by(Analysis.analysis_type)).all())
    recent = db.scalars(select(Analysis).where(Analysis.user_id == user.id).order_by(Analysis.created_at.desc()).limit(5))
    return {"total": sum(counts.values()), "single": counts.get("single", 0), "live": counts.get("live", 0),
            "batch": counts.get("batch", 0), "compare": counts.get("compare", 0),
            "recent": [{"id": r.id, "filename": r.filename, "analysis_type": r.analysis_type,
                        "created_at": r.created_at.isoformat(), "status": r.status} for r in recent]}


@router.get("/history")
def history(request: Request, limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0),
            db: Session = Depends(get_db), user: User = Depends(current_user)):
    cleanup_expired(db, request.app.state.storage, user.id)
    rows = db.scalars(select(Analysis).where(Analysis.user_id == user.id).order_by(Analysis.created_at.desc()).offset(offset).limit(limit))
    return [{"id": r.id, "analysis_type": r.analysis_type, "filename": r.filename, "status": r.status,
             "created_at": r.created_at.isoformat(), "item_count": len(r.result["items"]),
             "label": ("Side-by-side comparison" if r.analysis_type == "compare" else
                       f'{sum(item["status"] == "complete" for item in r.result["items"])} / {len(r.result["items"])} images analyzed'
                       if r.analysis_type == "batch" else
                       r.result["items"][0].get("classification", {}).get("label", "Analysis incomplete"))} for r in rows]


@router.get("/history/{identifier}")
def detail(identifier: str, request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    cleanup_expired(db, request.app.state.storage, user.id)
    return response_record(owned_record(db, user, identifier))


@router.delete("/history/{identifier}", status_code=204)
def delete(identifier: str, request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    row = owned_record(db, user, identifier)
    for key in row.image_keys + [row.report_key]:
        request.app.state.storage.delete(key)
    db.delete(row)
    db.commit()
    return Response(status_code=204)


@router.get("/history/{identifier}/images/{index}/{view}")
def image(identifier: str, index: int, view: str, request: Request,
          db: Session = Depends(get_db), user: User = Depends(current_user)):
    cleanup_expired(db, request.app.state.storage, user.id)
    row = owned_record(db, user, identifier)
    if view not in ("original", "ela", "fft"):
        raise HTTPException(404, "Image view not found.")
    key = f"images/{row.id}/{index}-{view}.png"
    if key not in row.image_keys:
        raise HTTPException(404, "Image previews were not retained or have expired.")
    try:
        return Response(request.app.state.storage.read(key), media_type="image/png", headers={"Cache-Control": "no-store"})
    except FileNotFoundError:
        raise HTTPException(404, "Image preview no longer exists.")


def csv_cell(value):
    value = str(value)
    return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


@router.get("/reports/{identifier}")
def report(identifier: str, request: Request, format: str = Query("json", pattern="^(json|csv|pdf)$"),
           db: Session = Depends(get_db), user: User = Depends(current_user)):
    row = owned_record(db, user, identifier)
    headers = {"Content-Disposition": f'attachment; filename="securelens-{row.id}.{format}"', "Cache-Control": "no-store"}
    if format == "json":
        return Response(request.app.state.storage.read(row.report_key), media_type="application/json", headers=headers)
    if format == "csv":
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["filename", "status", "result", "ai_indicators", "manipulation_indicators",
                         "ela_mean", "fft_mean", "brightness", "width", "height", "entropy", "edge_density",
                         "jpeg_q90_mae", "decision_source"])
        for item in row.result["items"]:
            metrics, classification = item.get("metrics", {}), item.get("classification", {})
            writer.writerow([csv_cell(item["filename"]), item["status"], classification.get("label", ""),
                             classification.get("ai_indicators", ""), classification.get("manipulation_indicators", ""),
                             metrics.get("ela_mean", ""), metrics.get("frequency_mean", ""),
                             metrics.get("brightness", ""), metrics.get("width", ""), metrics.get("height", ""),
                             metrics.get("entropy", ""), metrics.get("edge_density", ""),
                             metrics.get("ela_raw_mean", ""), classification.get("source", "")])
        return Response(buffer.getvalue(), media_type="text/csv", headers=headers)
    from reportlab.pdfgen import canvas
    from textwrap import wrap
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    y = 790
    def line(text):
        nonlocal y
        for part in wrap(str(text), 90):
            if y < 55:
                pdf.showPage()
                y = 790
            pdf.setFont("Helvetica", 10)
            pdf.drawString(45, y, part.encode("latin-1", errors="replace").decode("latin-1"))
            y -= 17
    payload = report_payload(row)
    line(payload["title"])
    line(payload["timestamp"])
    for item in row.result["items"]:
        line(item["filename"])
        line(item.get("classification", {}).get("label", item.get("error", "Incomplete")))
        classification = item.get("classification", {})
        line(f'AI indicators: {classification.get("ai_indicators", "Unavailable")} / Forensic heuristics')
        line("Manipulation indicators: not established. No trained-model probability.")
        for key in ("ela_mean", "ela_std", "frequency_mean", "frequency_std", "brightness", "std",
                    "width", "height", "analysis_width", "analysis_height", "channels", "format", "mode",
                    "file_size_bytes", "entropy", "edge_density", "ela_raw_mean"):
            line(f'{key}: {item.get("metrics", {}).get(key, "Unavailable")}')
        for compression in item.get("metrics", {}).get("compression", []):
            line(f'JPEG Q{compression["quality"]}: mean pixel loss {compression["mean_abs_difference"]:.3f}, deviation {compression["std_abs_difference"]:.3f}')
        for explanation in classification.get("explanations", []):
            line(explanation)
        y -= 15
    line(payload["limitations"])
    pdf.save()
    return Response(buffer.getvalue(), media_type="application/pdf", headers=headers)
