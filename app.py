# app.py
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

############################
# 기본 설정
############################
st.set_page_config(page_title="자료 소스 탐색기 GPT (SourceFinder)", layout="wide")

st.title("자료 소스 탐색기 GPT (SourceFinder)")
st.caption("업로드된 데이터셋을 1차 근거로, ‘카테고리·사이트명·URL·메모’를 기반 정렬합니다. "
           "정렬 기준: (1) 의미 유사도 (2) 키워드/도메인 직매칭 (3) 국내/공공 가중치")

DATA_DIR = "data"
DEFAULT_TOPK = 10

############################
# 유틸 함수
############################
URL_CANDIDATE_COLS = ["url", "url주소", "주소", "link", "링크"]

def detect_encoding(path):
    with open(path, "rb") as f:
        raw = f.read()
    res = chardet.detect(raw)
    return res.get("encoding") or "utf-8"

def find_url_column(df: pd.DataFrame) -> str | None:
    cols = [c for c in df.columns]
    lc_map = {c: c.lower() for c in cols}
    for c in cols:
        cl = lc_map[c]
        if any(k in cl for k in URL_CANDIDATE_COLS):
            return c
    # 백업: 정확히 "URL" 같은 컬럼 존재 시
    for c in cols:
        if c.strip().lower() == "url":
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
    # tldextract: subdomain.domain.suffix → registered_domain
    try:
        ext = tldextract.extract(url)
        if ext.registered_domain:
            return ext.registered_domain
    except Exception:
        pass
    # fallback
    try:
        netloc = urlparse(url).netloc
        return netloc.lower()
    except Exception:
        return ""

def make_markdown_table(rows: list[dict]) -> str:
    # rows: dict with keys ["순위","카테고리","사이트명","URL","연관성","한줄 근거"]
    # '사이트명'은 클릭 가능한 링크로, URL 미확인 시 plain 텍스트
    header = "| 순위 | 카테고리 | 사이트명 | URL | 연관성 | 한줄 근거 |\n|---:|---|---|---|---:|---|\n"
    lines = []
    for r in rows:
        name = r["사이트명"]
        url = r["URL"]
        score = f"{r['연관성']:.2f}"
        if url and url != "URL 미확인" and isinstance(url, str) and url.startswith(("http://", "https://")):
            name_md = f"[{name}]({url})"
            url_md = url
        else:
            name_md = name
            url_md = "(URL 미확인)"
        line = f"| {r['순위']} | {r['카테고리']} | {name_md} | {url_md} | {score} | {r['한줄 근거']} |"
        lines.append(line)
    return header + "\n".join(lines)

############################
# 데이터 로딩 & 병합 (캐시)
############################
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
        except Exception as e:
            st.warning(f"파일 로딩 실패: {p} ({e})")
            continue

        # 표준 컬럼 정리
        # 사이트명, 카테고리, URL, 간략메모가 이상적인 스키마
        colmap = {}
        # 사이트명
        for c in df.columns:
            if str(c).strip() in ["사이트명", "site", "sitename", "name", "사이트"]:
                colmap[c] = "사이트명"
        if "사이트명" not in colmap.values():
            # 추정: 첫 번째 문자열형 컬럼을 사이트명으로 사용
            for c in df.columns:
                if df[c].dtype == "object":
                    colmap[c] = "사이트명"
                    break

        # 카테고리
        for c in df.columns:
            if str(c).strip() in ["카테고리", "category", "분류", "대분류"]:
                colmap[c] = "카테고리"

        # URL
        url_col = find_url_column(df)
        if url_col:
            colmap[url_col] = "URL"

        # 간략메모
        for c in df.columns:
            if str(c).strip() in ["간략메모", "메모", "notes", "설명", "비고"]:
                colmap[c] = "간략메모"

        df2 = df.copy()
        df2 = df2.rename(columns=colmap)

        # 표준 컬럼이 없다면 생성
        for c in ["사이트명", "카테고리", "URL", "간략메모"]:
            if c not in df2.columns:
                df2[c] = ""

        dfs.append(df2[["카테고리", "사이트명", "URL", "간략메모"]])

    if not dfs:
        return pd.DataFrame(columns=["카테고리", "사이트명", "URL", "간략메모"])

    merged = pd.concat(dfs, ignore_index=True)

    # 공백/NA 정리
    for c in ["카테고리", "사이트명", "URL", "간략메모"]:
        merged[c] = merged[c].fillna("").astype(str).str.strip()

    # 완전 중복 제거
    merged = merged.drop_duplicates()

    return merged

base_df = load_all_datasets(DATA_DIR)

# ─────────────────────────────────────────
# 1) 사이드바 1차 구성: 질의/가중치/TopN + 파일 업로더
# ─────────────────────────────────────────
with st.sidebar:
    st.header("검색 옵션")
    query = st.text_input("질의(키워드)", value="국내 공공 데이터 소비 트렌드")
    boost_public = st.toggle("국내 공공 도메인(.go.kr, .kr) 가중치", value=True)
    topk = st.slider("Top N", min_value=5, max_value=50, value=DEFAULT_TOPK, step=1)

    st.markdown("---")
    st.caption("데이터 추가")
    uploaded = st.file_uploader("CSV/XLSX 추가 업로드(선택)", type=["csv", "xlsx"])

# 2) 업로드 데이터 병합(있으면)
if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            up_bytes = uploaded.read()
            enc = chardet.detect(up_bytes).get("encoding") or "utf-8"
            up_df = pd.read_csv(io.BytesIO(up_bytes), encoding=enc)
        else:
            up_df = pd.read_excel(uploaded)

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
        for c in ["카테고리", "사이트명", "URL", "간략메모"]:
            if c not in up_df2.columns:
                up_df2[c] = ""
        up_df2 = up_df2[["카테고리", "사이트명", "URL", "간략메모"]]

        base_df = pd.concat([base_df, up_df2], ignore_index=True).drop_duplicates()
        st.success(f"추가 데이터 병합 완료! (총 {len(base_df)}건)")
    except Exception as e:
        st.error(f"업로드 처리 실패: {e}")

# 3) 병합 이후에 카테고리 옵션 생성 (👉 여기서 생성!)
with st.sidebar:
    sel_categories = sorted([c for c in base_df["카테고리"].unique() if c])
    selected_cats = st.multiselect("카테고리(선택)", sel_categories, default=[])
with st.sidebar:
    sel_categories = sorted([c for c in base_df["카테고리"].unique() if c])
    selected_cats = st.multiselect("카테고리(선택)", sel_categories, default=[])

    # ← 여기서도 반드시 같은 레벨(스페이스 4칸)
    st.markdown("---")
    st.markdown("**연관성 점수 읽기**")
    st.markdown(
        """
0.0 ~ 0.1 → 거의 무관  
0.1 ~ 0.2 → 약간 관련  
0.2 ~ 0.4 → 보통 관련  
0.4 ~     → 강하게 관련
        """
    )

############################
# 랭킹 로직
############################
def build_corpus(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    text_cols = ["카테고리", "사이트명", "간략메모"]
    docs = [concat_fields(r, text_cols) for _, r in df.iterrows()]
    return docs, text_cols

def direct_match_score(row: pd.Series, tokens: list[str]) -> float:
    # (2) 키워드/도메인 직매칭 점수
    text = (normalize_text(row.get("카테고리", "")) + " " +
            normalize_text(row.get("사이트명", "")) + " " +
            normalize_text(row.get("간략메모", "")) + " " +
            normalize_text(row.get("URL", ""))).lower()

    # 토큰 매칭 (중복 제거)
    uniq_tokens = set([t for t in tokens if t])
    hit = sum(1 for t in uniq_tokens if t in text)
    # 도메인 내 키워드 매칭 가중
    dom = extract_domain(row.get("URL", ""))
    dom_hit = sum(1 for t in uniq_tokens if t in dom)

    raw = hit + 1.5 * dom_hit
    # 정규화 (대략 0~1 범위)
    return 1 - math.exp(-0.4 * raw)  # 완만한 S-curve

def public_locale_boost(row: pd.Series, wants_public: bool, query_text: str) -> float:
    # (3) 국내/공공 가중치
    boost = 0.0
    url = normalize_text(row.get("URL", ""))
    dom = extract_domain(url)
    is_kr = dom.endswith(".kr") or ".kr/" in url.lower()
    is_go_kr = dom.endswith(".go.kr")
    name = normalize_text(row.get("사이트명", ""))
    # 질의에 국내/한국/공공 포함?
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

    # 상한
    return min(boost, 1.0)

def brief_reason(row: pd.Series, tokens: list[str], sim: float, dm: float, pb: float) -> str:
    reasons = []
    cat = normalize_text(row.get("카테고리", ""))
    if cat:
        reasons.append(f"카테고리 '{cat}'")
    hits = [t for t in tokens if t and (t in normalize_text(row.get("사이트명","")).lower() 
                                        or t in normalize_text(row.get("간략메모","")).lower()
                                        or t in normalize_text(row.get("URL","")).lower())]
    if hits:
        reasons.append(f"키워드 매칭({', '.join(sorted(set(hits))[:3])})")
    url = normalize_text(row.get("URL", ""))
    dom = extract_domain(url)
    if dom.endswith(".go.kr"):
        reasons.append("국내 공공 도메인")
    if not reasons:
        reasons.append("텍스트 유사도 상위")
    # 너무 길면 컷
    txt = " · ".join(reasons)
    return txt[:80]

def rank_results(df: pd.DataFrame, query_text: str, wants_public=True, selected_cats=None, topk=10) -> pd.DataFrame:
    # 빈 데이터 가드
    if df is None or df.empty:
        return pd.DataFrame(columns=["순위", "카테고리", "사이트명", "URL", "연관성", "한줄 근거"])

    # 카테고리 필터
    if selected_cats:
        df = df[df["카테고리"].isin(selected_cats)].copy()
        if df.empty:
            return pd.DataFrame(columns=["순위", "카테고리", "사이트명", "URL", "연관성", "한줄 근거"])

    # 1) 의미 유사도 (TF-IDF)
    docs, _ = build_corpus(df)
    corpus = docs + [query_text]
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    tf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tf[:-1], tf[-1]).reshape(-1)
    sims = np.clip(sims, 0, 1)

    # 2) 키워드/도메인 직매칭
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", query_text.lower())
    direct_scores, public_boosts = [], []
    for _, row in df.iterrows():
        dm = direct_match_score(row, tokens)
        pb = public_locale_boost(row, wants_public, query_text)
        direct_scores.append(dm)
        public_boosts.append(pb)
    direct_scores = np.array(direct_scores)
    public_boosts = np.array(public_boosts)

    # 3) 최종 점수
    final = 0.6 * sims + 0.3 * direct_scores + 0.1 * np.minimum(1.0, public_boosts)
    final = np.clip(final, 0, 1)

    out = df.copy().reset_index(drop=True)
    out["연관성"] = final
    out["_sim"] = sims
    out["_dm"] = direct_scores
    out["_pb"] = public_boosts

    # 도메인 추출
    out["_domain"] = out["URL"].apply(extract_domain).fillna("")

    # 점수 내림차순 정렬
    out = out.sort_values("연관성", ascending=False)

    # ✅ URL 없으면 사이트명으로 그룹화 (탭/스페이스 혼용 금지!)
    dom = out["_domain"].fillna("").astype(str)
    site = out["사이트명"].fillna("").astype(str)
    group_key = np.where(dom.str.len() == 0, site, dom)
    out = out.groupby(group_key).head(1)

    # 상위 N
    out = out.sort_values("연관성", ascending=False).head(topk).copy()

    # 한줄 근거
    reasons = []
    for _, row in out.iterrows():
        reasons.append(brief_reason(row, tokens, row["_sim"], row["_dm"], row["_pb"]))
    out["한줄 근거"] = reasons

    # 순위 컬럼 + 내부 컬럼 정리
    out.insert(0, "순위", range(1, len(out) + 1))
    out = out.drop(columns=["_sim", "_dm", "_pb", "_domain"], errors="ignore")

    # 출력 스키마 보장
    cols = ["순위", "카테고리", "사이트명", "URL", "연관성", "한줄 근거"]
    return out[cols]

# ─────────────────────────────────────────
# 4) 결과 랭킹 & 표시
# ─────────────────────────────────────────
if st.button("검색 실행", type="primary") or query:
    if base_df.empty:
        st.error("데이터가 비어 있습니다. data/ 폴더에 CSV/XLSX 파일을 넣거나 상단에서 업로드하세요.")
    else:
        result_df = rank_results(
            base_df.copy(),
            query_text=query,
            wants_public=boost_public,
            selected_cats=selected_cats,
            topk=topk
        )

        if result_df.empty:
            st.info("결과가 없습니다. 키워드를 더 구체화하거나 카테고리 필터를 해제하세요.")
        else:
            rows = result_df[["순위","카테고리","사이트명","URL","연관성","한줄 근거"]].to_dict(orient="records")
            md_table = make_markdown_table(rows)
            st.markdown("### 연관 자료 소스")
            st.markdown(md_table)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("결과 CSV 다운로드", csv_bytes, "sourcefinder_results.csv", "text/csv")

        # 추가 탐색 팁 (3줄 이내)
        st.markdown("—")
        st.markdown("**추가 탐색 팁**")
        st.markdown("- 질의에 지역/산업/데이터형(예: ‘국내 공공 카드매출 통계 월간’)을 넣으면 정확도↑")
        st.markdown("- 같은 도메인은 1개만 노출됩니다. 세부 페이지는 ‘간략메모’ 키워드로 좁혀보세요.")
        st.markdown("- 공공 도메인(.go.kr, .kr) 우선 옵션을 토글하여 공공성 가중치를 조절하세요.")
