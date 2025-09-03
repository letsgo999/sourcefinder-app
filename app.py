# -*- coding: utf-8 -*-
import re
import io
import glob
import math
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import numpy as np
import chardet
import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="자료 소스 탐색기 GPT (SourceFinder)", layout="wide")
st.title("자료 소스 탐색기 GPT (SourceFinder)")
st.caption("업로드된 데이터셋을 1차 근거로 정렬합니다. 기준: (1) 의미 유사도 (2) 직매칭 (3) 국내/공공 가중치")

DATA_DIR = "data"
DEFAULT_TOPK = 10

# ── 유틸 ─────────────────────────────────────────────────────
def detect_encoding(path):
    with open(path, "rb") as f:
        raw = f.read()
    res = chardet.detect(raw)
    return res.get("encoding") or "utf-8"

def find_url_column(df: pd.DataFrame) -> str | None:
    candidates = ["url", "url주소", "주소", "link", "링크"]
    for c in df.columns:
        cl = str(c).strip().lower()
        if any(k in cl for k in candidates) or cl == "url":
            return c
    return None

def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x)

def concat_fields(row: pd.Series, text_cols: list[str]) -> str:
    return " ".join(normalize_text(row.get(c, "")) for c in text_cols)

def has_hangul(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))

def extract_domain(url: str) -> str:
    if not url or not isinstance(url, str):
        return ""
    try:
        ext = tldextract.extract(url)
        if ext.registered_domain:
            return ext.registered_domain
    except Exception:
        pass
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def make_markdown_table(rows: list[dict]) -> str:
    header = "| 순위 | 카테고리 | 사이트명 | URL | 연관성 | 한줄 근거 |\n|---:|---|---|---|---:|---|\n"
    lines = []
    for r in rows:
        name = r["사이트명"]
        url = r["URL"]
        score = f"{r['연관성']:.2f}"
        if url and isinstance(url, str) and url.startswith(("http://", "https://")):
            name_md = f"[{name}]({url})"
            url_md = url
        else:
            name_md = name
            url_md = "(URL 미확인)"
        line = f"| {r['순위']} | {r['카테고리']} | {name_md} | {url_md} | {score} | {r['한줄 근거']} |"
        lines.append(line)
    return header + "\n".join(lines)

# ── 데이터 로딩 & 병합 ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_datasets(data_dir: str) -> pd.DataFrame:
    paths = []
    paths += glob.glob(f"{data_dir}/**/*.csv", recursive=True)
    paths += glob.glob(f"{data_dir}/**/*.xlsx", recursive=True)
    dfs = []

    for p in paths:
        try:
            if p.lower().endswith(".csv"):
                enc = detect_encoding(p)
                df = pd.read_csv(p, encoding=enc)
            else:
                df = pd.read_excel(p)
        except Exception:
            continue

        # 표준화
        colmap = {}
        url_col = find_url_column(df)
        if url_col:
            colmap[url_col] = "URL"

        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ["site", "sitename", "name", "사이트", "사이트명"]:
                colmap[c] = "사이트명"
            if cl in ["category", "카테고리", "분류", "대분류"]:
                colmap[c] = "카테고리"
            if cl in ["notes", "메모", "간략메모", "설명", "비고"]:
                colmap[c] = "간략메모"

        df2 = df.rename(columns=colmap)
        for c in ["카테고리", "사이트명", "URL", "간략메모"]:
            if c not in df2.columns:
                df2[c] = ""
        dfs.append(df2[["카테고리", "사이트명", "URL", "간략메모"]])

    if not dfs:
        return pd.DataFrame(columns=["카테고리", "사이트명", "URL", "간략메모"])

    merged = pd.concat(dfs, ignore_index=True).drop_duplicates()
    for c in ["카테고리", "사이트명", "URL", "간략메모"]:
        merged[c] = merged[c].fillna("").astype(str).str.strip()
    return merged

# ✅ 세션 상태 초기화 (load_all_datasets ‘정의 후’ 여기에!)
if "base_df" not in st.session_state:
    st.session_state["base_df"] = load_all_datasets(DATA_DIR)

# ── 랭킹 로직 ────────────────────────────────────────────────
def build_corpus(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    text_cols = ["카테고리", "사이트명", "간략메모"]
    docs = [concat_fields(r, text_cols) for _, r in df.iterrows()]
    return docs, text_cols

def direct_match_score(row: pd.Series, tokens: list[str]) -> float:
    text = (
        normalize_text(row.get("카테고리", "")) + " " +
        normalize_text(row.get("사이트명", "")) + " " +
        normalize_text(row.get("간략메모", "")) + " " +
        normalize_text(row.get("URL", ""))
    ).lower()
    uniq = set([t for t in tokens if t])
    hit = sum(1 for t in uniq if t in text)
    dom = extract_domain(row.get("URL", ""))
    dom_hit = sum(1 for t in uniq if t in dom)
    raw = hit + 1.5 * dom_hit
    return 1 - math.exp(-0.4 * raw)

def public_locale_boost(row: pd.Series, wants_public: bool, query_text: str) -> float:
    boost = 0.0
    url = normalize_text(row.get("URL", ""))
    dom = extract_domain(url)
    is_kr = dom.endswith(".kr") or ".kr/" in url.lower()
    is_go_kr = dom.endswith(".go.kr")
    name = normalize_text(row.get("사이트명", ""))
    q = query_text.lower()
    hint_domestic = any(k in q for k in ["국내", "한국", "코리아", "korea"])
    hint_public = any(k in q for k in ["공공", "정부", "stat", "통계", "data"])
    if wants_public and is_go_kr:
        boost += 0.6
    elif wants_public and is_kr:
        boost += 0.35
    if hint_domestic and (is_kr or has_hangul(name)):
        boost += 0.2
    if hint_public and (is_go_kr or "data.go.kr" in url.lower()):
        boost += 0.2
    return min(boost, 1.0)

def brief_reason(row: pd.Series, tokens: list[str], sim: float, dm: float, pb: float) -> str:
    reasons = []
    cat = normalize_text(row.get("카테고리", ""))
    if cat:
        reasons.append(f"카테고리 '{cat}'")
    hits = [t for t in tokens if t and (
        t in normalize_text(row.get("사이트명","")).lower() or
        t in normalize_text(row.get("간략메모","")).lower() or
        t in normalize_text(row.get("URL","")).lower()
    )]
    if hits:
        reasons.append(f"키워드 매칭({', '.join(sorted(set(hits))[:3])})")
    dom = extract_domain(normalize_text(row.get("URL","")))
    if dom.endswith(".go.kr"):
        reasons.append("국내 공공 도메인")
    if not reasons:
        reasons.append("텍스트 유사도 상위")
    return " · ".join(reasons)[:80]

def rank_results(df: pd.DataFrame, query_text: str, wants_public=True, selected_cats=None, topk=10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["순위","카테고리","사이트명","URL","연관성","한줄 근거"])
    if selected_cats:
        df = df[df["카테고리"].isin(selected_cats)].copy()
        if df.empty:
            return pd.DataFrame(columns=["순위","카테고리","사이트명","URL","연관성","한줄 근거"])

    docs, _ = build_corpus(df)
    corpus = docs + [query_text]
    tf = TfidfVectorizer(min_df=1, ngram_range=(1,2)).fit_transform(corpus)
    sims = cosine_similarity(tf[:-1], tf[-1]).reshape(-1)
    sims = np.clip(sims, 0, 1)

    tokens = re.findall(r"[가-힣A-Za-z0-9]+", query_text.lower())
    direct_scores, public_boosts = [], []
    for _, row in df.iterrows():
        direct_scores.append(direct_match_score(row, tokens))
        public_boosts.append(public_locale_boost(row, wants_public, query_text))
    direct_scores = np.array(direct_scores)
    public_boosts = np.array(public_boosts)

    final = 0.6*sims + 0.3*direct_scores + 0.1*np.minimum(1.0, public_boosts)
    final = np.clip(final, 0, 1)

    out = df.copy().reset_index(drop=True)
    out["연관성"] = final
    out["_sim"] = sims
    out["_dm"] = direct_scores
    out["_pb"] = public_boosts
    out["_domain"] = out["URL"].apply(extract_domain).fillna("")
    out = out.sort_values("연관성", ascending=False)

    dom = out["_domain"].fillna("").astype(str)
    site = out["사이트명"].fillna("").astype(str)
    group_key = np.where(dom.str.len() == 0, site, dom)
    out = out.groupby(group_key).head(1)

    out = out.sort_values("연관성", ascending=False).head(topk).copy()

    reasons = []
    for _, row in out.iterrows():
        reasons.append(brief_reason(row, tokens, row["_sim"], row["_dm"], row["_pb"]))
    out["한줄 근거"] = reasons

    out.insert(0, "순위", range(1, len(out)+1))
    out = out.drop(columns=["_sim","_
