"""从 Pexels 公开图库采集实验室场景图片，作为「网络图片数据集」的素材池。

合规与用途说明
--------------
1. 只访问 Pexels 搜索结果的**第一页**（其 robots.txt 明确允许搜索路径，
   只禁止 page=21 之后的深分页与带过滤参数的 URL），每两个查询之间间隔数秒，
   不做并发、不做批量爬取。
2. Pexels License 允许免费用于学习、研究与商业项目，不要求署名但建议保留来源。
   本脚本把 photo id / 页面地址 / 查询词 / 授权说明一并写入 manifest.json，
   便于在项目报告「数据来源」一节如实披露。
3. 本脚本只负责**下载素材**，不做自动标注。图片的真实风险类别必须由人
   （或人工复核后的模型预标注）逐张判定 —— 图库图往往与查询词不符，
   直接拿查询词当标签会污染评测集。

用法：
    python scripts/fetch_web_images.py                 # 采集全部查询词
    python scripts/fetch_web_images.py --query "fume hood" --limit 12
    python scripts/fetch_web_images.py --dry-run       # 只打印将要采集的 id
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "web_raw"
MANIFEST = RAW_DIR / "manifest.json"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: 查询词 -> 采集意图说明（仅作为先验，不是标签）
QUERIES = {
    "chemistry laboratory": "正常实验室环境、器材陈列",
    "laboratory equipment": "正常场景与仪器特写",
    "scientist laboratory": "人物 + 防护装备，可筛出未规范佩戴的样本",
    "chemical bottles": "试剂瓶，用于试剂管理类判定",
    "bunsen burner": "明火/加热源，用于热源附近易燃物判定",
    "laboratory glassware": "玻璃器皿，易混淆样本来源",
    "coffee cup desk": "实验区饮食类素材",
    "wet floor": "地面洒漏/湿滑，通道风险素材",
    "cluttered floor boxes": "通道堵塞素材",
    "safety goggles": "护目镜摆放，易混淆（有护目镜但未佩戴）",
}

PHOTO_RE = re.compile(r"photos/(\d+)/pexels-photo-\1")
IMG_TAG_RE = re.compile(r"<img[^>]*pexels-photo-(\d+)[^>]*>", re.I)
ALT_RE = re.compile(r'alt="([^"]*)"', re.I)


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            # 只在图片请求里接受 JPEG：Pexels 的 CDN 会按 Accept 返回 AVIF，
            # 而 AVIF 文件仍然叫 .jpg，多模态模型接口会直接拒绝（400 code 1210）。
            "Accept": "text/html,application/xhtml+xml,image/jpeg,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.pexels.com/",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def search_page(query: str, retries: int = 3) -> str:
    """取搜索页第一页 HTML（深分页被 robots 禁止，不做）。403 会退避重试。"""
    slug = urllib.request.quote(query.replace(" ", "-"))
    url = f"https://www.pexels.com/search/{slug}/"
    last = None
    for attempt in range(retries):
        try:
            return _get(url).decode("utf-8", "ignore")
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (403, 429):
                raise
            wait = 8 * (attempt + 1)
            print(f"  … HTTP {exc.code}，{wait}s 后重试（{attempt + 1}/{retries}）")
            time.sleep(wait)
    raise last  # type: ignore[misc]


def parse_photos(html: str) -> list:
    """从 HTML 中解析 (photo_id, alt) 列表，按页面出现顺序。"""
    alts = {}
    for match in IMG_TAG_RE.finditer(html):
        pid = match.group(1)
        alt = ALT_RE.search(match.group(0))
        if alt and pid not in alts:
            text = alt.group(1).strip()
            if text:
                alts[pid] = text

    seen, ordered = set(), []
    for pid in PHOTO_RE.findall(html):
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append((pid, alts.get(pid, "")))
    return ordered


def download(pid: str, dest: Path) -> bool:
    url = (
        f"https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg"
        "?auto=compress&cs=tinysrgb&w=1280"
    )
    try:
        data = _get(url, timeout=60)
    except Exception as exc:  # noqa: BLE001
        print(f"    ! {pid} 下载失败：{type(exc).__name__} {str(exc)[:60]}")
        return False
    if len(data) < 8000:
        print(f"    ! {pid} 文件过小（{len(data)}B），跳过")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {"license": "Pexels License (free to use, attribution appreciated)",
            "source": "https://www.pexels.com", "items": []}


def main() -> None:
    parser = argparse.ArgumentParser(description="从 Pexels 采集实验室场景素材")
    parser.add_argument("--query", action="append", help="只采集指定查询词（可重复）")
    parser.add_argument("--limit", type=int, default=18, help="每个查询词最多下载张数")
    parser.add_argument("--delay", type=float, default=0.6, help="图片之间的间隔秒数")
    parser.add_argument("--dry-run", action="store_true", help="只解析不下载")
    args = parser.parse_args()

    queries = args.query or list(QUERIES)
    manifest = load_manifest()
    known = {(it["query"], it["photo_id"]) for it in manifest["items"]}

    total_new = 0
    for qi, query in enumerate(queries, 1):
        intent = QUERIES.get(query, "")
        print(f"[{qi}/{len(queries)}] 查询：{query}（{intent}）")
        try:
            html = search_page(query)
        except urllib.error.HTTPError as exc:
            print(f"  ! HTTP {exc.code}，该查询跳过（可能触发限流，稍后重试）")
            time.sleep(8)
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  ! 失败：{type(exc).__name__} {str(exc)[:70]}")
            time.sleep(5)
            continue

        photos = parse_photos(html)
        print(f"  解析到 {len(photos)} 张候选")
        if args.dry_run:
            for pid, alt in photos[:args.limit]:
                print(f"    {pid}  {alt[:70]}")
            time.sleep(2)
            continue

        got = 0
        for pid, alt in photos:
            if got >= args.limit:
                break
            if (query, pid) in known:
                continue
            dest = RAW_DIR / f"{pid}.jpg"
            if dest.exists():
                got += 1
                continue
            if download(pid, dest):
                got += 1
                total_new += 1
                manifest["items"].append({
                    "photo_id": pid,
                    "query": query,
                    "intent": intent,
                    "alt": alt,
                    "file": f"data/web_raw/{pid}.jpg",
                    "page": f"https://www.pexels.com/search/{query.replace(' ', '-')}/",
                    "image_url": f"https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg",
                })
            time.sleep(args.delay)
        print(f"  下载 {got} 张")
        time.sleep(3)

    if not args.dry_run:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n新增 {total_new} 张，素材池共 {len(manifest['items'])} 条记录")
        print(f"清单：{MANIFEST}")
        print("下一步：人工看图标注，不要把查询词直接当标签。")


if __name__ == "__main__":
    main()
