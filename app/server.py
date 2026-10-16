"""本地 Web 服务：把命令行能力包装成界面可调用的接口。

启动：
    python src/main.py serve --port 8000
然后浏览器打开 http://127.0.0.1:8000
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_file, send_from_directory

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT_DEFAULT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import evaluate_run, load_costs, load_labels, load_predictions  # noqa: E402
from llm import describe_backend  # noqa: E402
from pipeline import IMAGE_SUFFIXES, VIDEO_SUFFIXES, SafetyAnalysisPipeline  # noqa: E402
from self_evolution import (  # noqa: E402
    analyze_cases,
    list_versions,
    promote_strategy,
    propose_strategy_version,
    record_feedback,
)

ALLOWED_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES


def create_app(project_root: Path | None = None) -> Flask:
    project_root = Path(project_root or PROJECT_ROOT_DEFAULT).resolve()
    app = Flask(__name__, static_folder=None)
    app.config["PROJECT_ROOT"] = str(project_root)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB

    ui_dir = project_root / "ui"
    sessions_dir = project_root / "reports" / "sessions"
    uploads_dir = project_root / "reports" / "uploads"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    current_strategy = project_root / "configs" / "strategy_current.json"

    # ------------------------------------------------------------ 基础页面

    @app.get("/")
    def index():
        return send_from_directory(ui_dir, "index.html")

    @app.get("/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(ui_dir, filename)

    # ------------------------------------------------------------ 状态

    @app.get("/api/status")
    def api_status():
        backend = describe_backend()
        return jsonify({
            "backend": backend,
            "versions": list_versions(project_root),
            "cases": analyze_cases(project_root),
            "strategy": json.loads(current_strategy.read_text(encoding="utf-8")).get("version"),
        })

    # ------------------------------------------------------------ 分析

    @app.post("/api/analyze")
    def api_analyze():
        upload = request.files.get("file")
        if not upload or not upload.filename:
            return jsonify({"ok": False, "error": "请先选择一张图片或一段视频"}), 400

        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            return jsonify({
                "ok": False,
                "error": f"不支持的文件类型 {suffix}，请上传图片或视频",
            }), 400

        scenario = request.form.get("scenario", "chemistry_lab")
        force_mock = request.form.get("mock", "false").lower() == "true"

        ablation: Dict[str, bool] = {}
        for key, flag in (("use_rules", "no_rules"), ("use_memory", "no_memory"),
                          ("use_review_agent", "no_review")):
            if request.form.get(flag, "false").lower() == "true":
                ablation[key] = False

        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = Path(upload.filename).name
        saved = uploads_dir / f"{stamp}_{safe_name}"
        upload.save(saved)

        sample_id = f"{stamp}_{Path(safe_name).stem}"
        out_dir = sessions_dir / sample_id

        try:
            pipeline = SafetyAnalysisPipeline(
                project_root, current_strategy, force_mock=force_mock, ablation=ablation or None
            )
            report = pipeline.analyze(saved, scenario, out_dir)
        except Exception as err:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"分析失败：{err}"}), 500

        payload = report.to_dict()
        payload["ok"] = True
        payload["report_dir"] = str(out_dir)
        return jsonify(payload)

    # ------------------------------------------------------------ 反馈与进化

    @app.post("/api/feedback")
    def api_feedback():
        data: Dict[str, Any] = request.get_json(silent=True) or {}
        sample_id = data.get("sample_id")
        if not sample_id:
            return jsonify({"ok": False, "error": "缺少 sample_id"}), 400

        report_json = _find_report_json(sessions_dir, sample_id)
        if report_json is None:
            return jsonify({"ok": False, "error": f"找不到样本 {sample_id} 的报告"}), 404

        original = data.get("original_risk_type")
        if not original:
            return jsonify({"ok": False, "error": "请选择要纠正的风险类型"}), 400

        note = (data.get("note") or "").strip()
        if not note:
            return jsonify({"ok": False, "error": "请填写纠错说明，它将作为自进化的依据"}), 400

        try:
            memory_path = record_feedback(
                project_root,
                report_json,
                original,
                data.get("correct_risk_type"),
                data.get("correct_severity"),
                note,
                reviewer=data.get("reviewer", ""),
            )
            version_path = propose_strategy_version(project_root, current_strategy, note)
        except Exception as err:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"生成策略草案失败：{err}"}), 500

        strategy = json.loads(version_path.read_text(encoding="utf-8"))
        latest = (strategy.get("change_log") or [{}])[-1]
        return jsonify({
            "ok": True,
            "memory_path": str(memory_path),
            "draft": str(version_path),
            "version": strategy.get("version"),
            "changes": latest.get("changes", []),
            "evidence": latest.get("evidence", {}),
            "cases": analyze_cases(project_root),
        })

    @app.post("/api/promote")
    def api_promote():
        data = request.get_json(silent=True) or {}
        version = data.get("version")
        if not version:
            return jsonify({"ok": False, "error": "缺少 version"}), 400
        try:
            target = promote_strategy(project_root, version)
        except FileNotFoundError as err:
            return jsonify({"ok": False, "error": str(err)}), 404
        return jsonify({"ok": True, "current": str(target), "version": version})

    @app.get("/api/versions")
    def api_versions():
        current = json.loads(current_strategy.read_text(encoding="utf-8")).get("version")
        return jsonify({"ok": True, "current": current, "versions": list_versions(project_root)})

    # ------------------------------------------------------------ 报告与评测

    @app.get("/api/report/<sample_id>")
    def api_report(sample_id: str):
        fmt = request.args.get("format", "json")
        path = _find_report_json(sessions_dir, sample_id)
        if path is None:
            return jsonify({"ok": False, "error": "找不到报告"}), 404
        if fmt == "md":
            md = path.with_suffix(".md")
            if not md.exists():
                return jsonify({"ok": False, "error": "Markdown 报告不存在"}), 404
            return send_file(md, mimetype="text/markdown; charset=utf-8", as_attachment=True,
                             download_name=f"{sample_id}_report.md")
        return jsonify(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/compare")
    def api_compare():
        """把 reports/sessions 下所有结果汇总成一次评测，用于消融对比。"""
        labels_path = project_root / "data" / "labels_test.csv"
        if not labels_path.exists():
            return jsonify({"ok": False, "error": "缺少 data/labels_test.csv，无法评测"}), 400
        labels = load_labels(labels_path)
        predictions = load_predictions(sessions_dir)
        return jsonify({"ok": True, "metrics": evaluate_run(labels, predictions, load_costs(sessions_dir))})

    # ------------------------------------------------------------ 错误兜底

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"ok": False, "error": "接口不存在"}), 404

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"ok": False, "error": "文件过大，请上传 64 MB 以内的文件"}), 413

    return app


def _find_report_json(sessions_dir: Path, sample_id: str):
    candidates = [
        sessions_dir / sample_id / f"{sample_id}_report.json",
        sessions_dir / f"{sample_id}_report.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(sessions_dir.glob(f"*{sample_id}*/*_report.json"))
    return matches[0] if matches else None
