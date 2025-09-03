# app.py
# -*- coding: utf-8 -*-

import re
import io
import glob
import math
import hashlib
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import numpy as np
import chardet
import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────
# 기본 세팅
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="자료 소스 탐색기 GPT (SourceFinder)", layout="wide")

st.title("자료 소스 탐색기 GPT (SourceFinder)")
st.caption(
    "업로드된 데이터셋을 1차 근거로 정렬합니다. 정렬 기준: "
    "(1) 의미 유사도 (2) 키워드/도메인 직매칭 (3) 국내/공공 가중치"
)

DATA_DIR = "data"
DEFAULT_TOPK = 10


# ─────────────────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────────────────
def detect_encoding(path: str) -> str:
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


def normalize_text(x) -> str:
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


# ─────────────────────────────────────────────────────────
# 데이터 로딩 & 병합 (레포의 data/ 고정)
# ─────────────────────────────────────────────────────────
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

        # 표준 컬럼 매핑
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
        for col in ["카테고리", "사이트명", "URL", "간략메모"]:
            if col not in df2.columns:
                df2[col] = ""

        dfs.append(df2[["카테고리", "사이트명", "URL", "간략메모"]])

    if not dfs:
        return pd.DataFrame(columns=["카테고리", "사이트명", "URL", "간략메모"])

    merged = pd.concat(dfs, ignore_index=True).drop_duplicates()
    for col in ["카테고리", "사이트명", "URL", "간략메모"]:
        merged[col] = merged[col].fillna("").astype(str).str.strip()
    return merged


# ✅ 세션 초기화: 레포의 data/만 불러와 고정(누구나 새로고침해도 동일 시작점)
if "base_df" not in st.session_state:
    st.session_state["base_df"] = load_all_datasets(DATA_DIR)

# (선택) 업로드 해시 초기화
if "last_upload_hash" not in st.session_state:
    st.session_state["last_upload_hash"] = None


# ─────────────────────────────────────────────────────────
# 랭킹 로직
# ─────────────────────────────────────────────────────────
def build_corpus(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    text_cols = ["카테고리", "사이트명", "간략메모"]
    docs = [concat_fields(r, text_cols) for _, r in df.iterrows()]
    return docs, text_cols


def direct_match_score(row: pd.Series, tokens: list[str]) -> float:
    text = (
        normalize_text(row.get("카테고리", "")) + " "
        + normalize_text(row.get("사이트명", "")) + " "
        + normalize_text(row.get("간략메모", "")) + " "
        + normalize_text(row.get("URL", ""))
    ).lower()

    uniq = set(t for t in tokens if t)
    hit = sum(1 for t in uniq if t in text)
    dom = extract_domain(row.get("URL", ""))
    dom_hit = sum(1 for t in uniq if t in dom)

    raw = hit + 1.5 * dom_hit
    return 1 - math.exp(-0.4 * raw)  # 0~1 스케일


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
    hits = [
        t for t in tokens if t and (
            t in normalize_text(row.get("사이트명", "")).lower()
            or t in normalize_text(row.get("간략메모", "")).lower()
            or t in normalize_text(row.get("URL", "")).lower()
        )
    ]
    if hits:
        reasons.append(f"키워드 매칭({', '.join(sorted(set(hits))[:3])})")
    dom = extract_domain(normalize_text(row.get("URL", "")))
    if dom.endswith(".go.kr"):
        reasons.append("국내 공공 도메인")
    if not reasons:
        reasons.append("텍스트 유사도 상위")
    return " · ".join(reasons)[:80]


def rank_results(
    df: pd.DataFrame,
    query_text: str,
    wants_public: bool = True,
    selected_cats: list[str] | None = None,
    topk: int = 10,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["순위", "카테고리", "사이트명", "URL", "연관성", "한줄 근거"])

    if selected_cats:
        df = df[df["카테고리"].isin(selected_cats)].copy()
        if df.empty:
            return pd.DataFrame(columns=["순위", "카테고리", "사이트명", "URL", "연관성", "한줄 근거"])

    # 1) 의미 유사도
    docs, _ = build_corpus(df)
    corpus = docs + [query_text]
    tf = TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit_transform(corpus)
    sims = cosine_similarity(tf[:-1], tf[-1]).reshape(-1)
    sims = np.clip(sims, 0, 1)

    # 2) 직매칭 / 3) 공공지표
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", query_text.lower())
    direct_scores, public_boosts = [], []
    for _, row in df.iterrows():
        direct_scores.append(direct_match_score(row, tokens))
        public_boosts.append(public_locale_boost(row, wants_public, query_text))
    direct_scores = np.array(direct_scores)
    public_boosts = np.array(public_boosts)

    # 최종 점수
    final = 0.6 * sims + 0.3 * direct_scores + 0.1 * np.minimum(1.0, public_boosts)
    final = np.clip(final, 0, 1)

    out = df.copy().reset_index(drop=True)
    out["연관성"] = final
    out["_sim"] = sims
    out["_dm"] = direct_scores
    out["_pb"] = public_boosts
    out["_domain"] = out["URL"].apply(extract_domain).fillna("")

    # 점수 정렬 → 도메인 중복 제거 → 상위 N
    out = out.sort_values("연관성", ascending=False)
    dom = out["_domain"].fillna("").astype(str)
    site = out["사이트명"].fillna("").astype(str)
    group_key = np.where(dom.str.len() == 0, site, dom)
    out = out.groupby(group_key).head(1)
    out = out.sort_values("연관성", ascending=False).head(topk).copy()

    # 한줄 근거
    reasons = []
    for _, row in out.iterrows():
        reasons.append(brief_reason(row, tokens, row["_sim"], row["_dm"], row["_pb"]))
    out["한줄 근거"] = reasons

    # 표정리
    out.insert(0, "순위", range(1, len(out) + 1))
    out = out.drop(columns=["_sim", "_dm", "_pb", "_domain"], errors="ignore")
    return out


# ─────────────────────────────────────────────────────────
# 사이드바 (검색 옵션 + 업로드 위젯)
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("검색 옵션")
    query = st.text_input("질의(키워드)", value="국내 공공 데이터 소비 트렌드")
    boost_public = st.toggle("국내 공공 도메인(.go.kr, .kr) 가중치", value=True)
    topk = st.slider("Top N", min_value=5, max_value=50, value=DEFAULT_TOPK, step=1)

    st.markdown("---")
    st.caption("데이터 추가 (업로드는 현재 세션에만 적용됩니다)")
    uploaded = st.file_uploader(
        "CSV/XLSX 추가 업로드(선택)",
        type=["csv", "xlsx"],
        key="uploader",
    )

# ─────────────────────────────────────────────────────────
# 업로드 처리(세션 한정 병합, 무한루프 방지)
# ─────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        up_bytes = uploaded.getvalue()
        fhash = hashlib.md5(up_bytes).hexdigest()
        if st.session_state.get("last_upload_hash") != fhash:
            if uploaded.name.lower().endswith(".csv"):
                enc = chardet.detect(up_bytes).get("encoding") or "utf-8"
                up_df = pd.read_csv(io.BytesIO(up_bytes), encoding=enc)
            else:
                up_df = pd.read_excel(io.BytesIO(up_bytes))

            url_col = find_url_column(up_df)
            colmap = {}
            if url_col:
                colmap[url_col] = "URL"
            for c in up_df.columns:
                cl = c.strip().lower()
                if cl in ["site", "sitename", "name", "사이트", "사이트명"]:
                    colmap[c] = "사이트명"
                if cl in ["category", "카테고리", "분류", "대분류"]:
                    colmap[c] = "카테고리"
                if cl in ["notes", "메모", "간략메모", "설명", "비고"]:
                    colmap[c] = "간략메모"

            up_df2 = up_df.rename(columns=colmap)
            for col in ["카테고리", "사이트명", "URL", "간략메모"]:
                if col not in up_df2.columns:
                    up_df2[col] = ""
            up_df2 = up_df2[["카테고리", "사이트명", "URL", "간략메모"]]

            st.session_state["base_df"] = (
                pd.concat([st.session_state["base_df"], up_df2], ignore_index=True)
                .drop_duplicates()
            )
            st.session_state["last_upload_hash"] = fhash
            st.success(f"추가 데이터 병합 완료! (현재 세션 기준 총 {len(st.session_state['base_df'])}건)")
    except Exception as e:
        st.error(f"업로드 처리 실패: {e}")

# ─────────────────────────────────────────────────────────
# 카테고리 옵션 (항상 최신 세션 DF 기준)
# ─────────────────────────────────────────────────────────
with st.sidebar:
    _cats = (
        st.session_state["base_df"]["카테고리"].dropna().astype(str).str.strip()
        if not st.session_state["base_df"].empty else pd.Series([], dtype=str)
    )
    sel_categories = sorted([c for c in _cats.unique() if c])
    selected_cats = st.multiselect("카테고리(선택)", sel_categories, default=[], key="cats")

    st.markdown("---")
    st.markdown("**연관성 점수 읽기**")
    st.markdown(
        "0.0 ~ 0.1 → 거의 무관  \n"
        "0.1 ~ 0.2 → 약간 관련  \n"
        "0.2 ~ 0.4 → 보통 관련  \n"
        "0.4 ~     → 강하게 관련"
    )


# ─────────────────────────────────────────────────────────
# 본문: 검색 실행
# ─────────────────────────────────────────────────────────
base_df = st.session_state["base_df"]

if base_df.empty:
    st.warning("레포의 `data/` 폴더에 기본 CSV/XLSX를 커밋해 두면 새로고침/다른 사용자도 항상 그 데이터를 사용합니다.")
else:
    if st.button("검색 실행", type="primary") or True:
        result_df = rank_results(
            base_df.copy(),
            query_text=query,
            wants_public=boost_public,
            selected_cats=selected_cats,
            topk=topk,
        )

        if result_df.empty:
            st.info("결과가 없습니다. 키워드를 더 구체화하거나 카테고리 필터를 해제하세요.")
        else:
            rows = result_df[
                ["순위", "카테고리", "사이트명", "URL", "연관성", "한줄 근거"]
            ].to_dict(orient="records")
            md_table = make_markdown_table(rows)

            st.markdown("### 연관 자료 소스")
            st.markdown(md_table, unsafe_allow_html=False)

            st.download_button(
                label="결과 CSV 다운로드",
                data=result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="sourcefinder_results.csv",
                mime="text/csv",
            )
