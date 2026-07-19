#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
巨潮资讯网 A股年报 PDF 批量下载脚本（试点版）
用法示例：
    # 按股票代码下载近3年年报
    python cninfo_downloader.py --stocks 000001,600519,300750 --years 2023,2024,2025

    # 按行业关键词检索（先查询列表，确认后下载）
    python cninfo_downloader.py --stocks 000001 --years 2024 --dry-run

注意：需在本地网络环境运行（沙盒无法访问 cninfo.com.cn）。
依赖：pip install requests
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
DOWNLOAD_BASE = "http://static.cninfo.com.cn/"
STOCK_JSON_URL = "http://www.cninfo.com.cn/new/data/szse_stock.json"  # 股票代码->orgId 列表（覆盖不全）
TOP_SEARCH_URL = "http://www.cninfo.com.cn/new/information/topSearch/query"  # 兜底搜索orgId

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
    "X-Requested-With": "XMLHttpRequest",
}

# 排除非正文公告：摘要、英文版、已取消、更正前的旧版等
EXCLUDE_PATTERNS = re.compile(r"摘要|英文|已取消|取消$|正文修订前|更新前")


def guess_column(stock_code: str) -> str:
    """根据股票代码判断交易所栏目。6开头=沪市，0/3开头=深市。"""
    return "sse" if stock_code.startswith("6") else "szse"


_ORG_ID_CACHE = None


def get_org_id(stock_code: str, session: requests.Session) -> str:
    """巨潮API的stock参数要求'代码,orgId'格式，从股票列表接口获取orgId（首次调用后缓存）。"""
    global _ORG_ID_CACHE
    if _ORG_ID_CACHE is None:
        cache_file = Path("data/szse_stock.json")
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            r = session.get(STOCK_JSON_URL, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        _ORG_ID_CACHE = {s["code"]: s["orgId"] for s in data.get("stockList", [])}
    org_id = _ORG_ID_CACHE.get(stock_code)
    if org_id:
        return org_id
    # 兜底：股票列表覆盖不全时，用搜索接口查orgId
    r = session.post(TOP_SEARCH_URL, data={"keyWord": stock_code, "maxNum": 10},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    for item in r.json() or []:
        if item.get("code") == stock_code and item.get("orgId"):
            _ORG_ID_CACHE[stock_code] = item["orgId"]
            return item["orgId"]
    raise ValueError(f"未找到代码 {stock_code} 的orgId，请检查代码是否正确")


def query_annual_reports(stock_code: str, year: int, session: requests.Session, debug=False):
    """查询某股票某会计年度的年报（次年披露，检索窗口取次年1-8月）。"""
    org_id = get_org_id(stock_code, session)
    se_date = f"{year + 1}-01-01~{year + 1}-08-31"
    payload = {
        "pageNum": 1,
        "pageSize": 30,
        "column": guess_column(stock_code),
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{stock_code},{org_id}",
        "searchkey": "",
        "secid": "",
        "category": "category_ndbg_szsh",  # 年度报告
        "trade": "",
        "seDate": se_date,
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    r = session.post(QUERY_URL, data=payload, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    if debug:
        anns = data.get("announcements") or []
        print(f"  [debug] API返回 {data.get('totalAnnouncement', 0)} 条公告，"
              f"标题: {[re.sub(r'<[^>]+>', '', a.get('announcementTitle', '')) for a in anns]}")
    results = []
    for ann in data.get("announcements") or []:
        title = re.sub(r"<[^>]+>", "", ann.get("announcementTitle", ""))
        if EXCLUDE_PATTERNS.search(title):
            continue
        if str(year) not in title and "年度报告" not in title and "年报" not in title:
            continue
        results.append({
            "sec_code": ann.get("secCode"),
            "sec_name": ann.get("secName"),
            "title": title,
            "adjunct_url": ann.get("adjunctUrl"),
            "announcement_time": ann.get("announcementTime"),
        })
    return results


def download_pdf(item: dict, out_dir: Path, session: requests.Session) -> Path | None:
    url = DOWNLOAD_BASE + item["adjunct_url"]
    safe_title = re.sub(r"[\\/:*?\"<>|\s]+", "_", item["title"])[:80]
    fname = f"{item['sec_code']}_{item['sec_name']}_{safe_title}.pdf"
    fpath = out_dir / fname
    if fpath.exists():
        print(f"  已存在，跳过: {fname}")
        return fpath
    for attempt in range(3):
        try:
            r = session.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                print(f"  [错误] 下载失败(已重试3次): {fname}: {e}")
                return None
            wait = 5 * (attempt + 1)
            print(f"  [重试{attempt + 1}] {e}，{wait}s后重试")
            time.sleep(wait)
    if not r.content.startswith(b"%PDF"):
        print(f"  [警告] 非PDF内容，跳过: {url}")
        return None
    fpath.write_bytes(r.content)
    print(f"  已下载 ({len(r.content) / 1e6:.1f} MB): {fname}")
    return fpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", required=True, help="逗号分隔的股票代码，如 000001,600519")
    ap.add_argument("--years", required=True, help="逗号分隔的会计年度，如 2023,2024")
    ap.add_argument("--out", default="data/raw_pdfs", help="PDF输出目录")
    ap.add_argument("--dry-run", action="store_true", help="仅打印检索结果，不下载")
    ap.add_argument("--debug", action="store_true", help="打印API原始返回，用于排查检索问题")
    ap.add_argument("--sleep", type=float, default=2.0, help="请求间隔秒数（礼貌抓取）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    manifest = []

    for code in [s.strip() for s in args.stocks.split(",") if s.strip()]:
        for year in [int(y) for y in args.years.split(",")]:
            print(f"检索 {code} {year}年年报 ...")
            try:
                items = query_annual_reports(code, year, session, debug=args.debug)
            except Exception as e:
                print(f"  [错误] 查询失败: {e}")
                continue
            if not items:
                print("  未找到符合条件的年报")
            for item in items:
                print(f"  -> {item['title']}")
                if not args.dry_run:
                    p = download_pdf(item, out_dir, session)
                    if p:
                        item["local_path"] = str(p)
                        manifest.append(item)
            time.sleep(args.sleep)

    if manifest:
        mpath = out_dir / "manifest.json"
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n共下载 {len(manifest)} 份，清单已写入 {mpath}")


if __name__ == "__main__":
    main()
