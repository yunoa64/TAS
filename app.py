import streamlit as st
import pandas as pd
import re
from openai import OpenAI
import anthropic
import time
import hashlib
import math
import json
import google.generativeai as genai

# --- 페이지 설정 ---
st.set_page_config(page_title="📄 논문 스크리닝 도우미", layout="wide")
st.title("📑 LLM 기반 논문 스크리닝 지원 도구")

# --- ✅ 사이드바: 항상 표시 ---
st.sidebar.header("⚙️ 모델 설정")
model_choice = st.sidebar.selectbox(
    "모델 선택",
    [
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5-mini",
        "gpt-5-nano",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "gemini-2.5-flash-lite"
    ]
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max tokens", 100, 4000, 800, 100)
show_reason = True  # 계속 True로 사용

# --- CSV 업로드 ---
uploaded_file = st.file_uploader("논문 목록 CSV 업로드", type=["csv"])

if uploaded_file:
    # ✅ 파일 내용 기반 해시 계산
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # 🔁 파일 내용 변경 감지 → 세션 초기화
    if "uploaded_file_hash" not in st.session_state or st.session_state["uploaded_file_hash"] != file_hash:
        st.session_state["uploaded_file_hash"] = file_hash
        st.session_state.pop("df", None)
        st.session_state.pop("results", None)
        st.session_state.pop("completion_message", None)
        st.session_state.pop("error_count", None)

    df = pd.read_csv(uploaded_file)

    # 선택 컬럼 추가
    if "select" not in df.columns:
        df.insert(0, "select", False)

    # --- 세션 상태 초기화 ---
    if "df" not in st.session_state:
        st.session_state.df = df.copy()

    # --- 사용자 입력 질문 ---
    st.subheader("🔍 Screening 질문 입력")
    user_question = st.text_input(
        "이 논문이 포함 기준에 부합하는지 묻고 싶은 질문을 입력하세요:",
        placeholder="이 논문에서 LLM 모델을 이용한 실험을 하고 있는 지 알려 줘.",
        key="screening_question_input"
    )

    # --- 결과 컬럼 이름 입력 ---
    result_col_name = st.text_input(
        "판단 결과 컬럼 이름 지정 (입력하지 않으면 질문이 자동으로 사용됩니다)",
        value="screening_result",
        key="result_col_name_input"
    )

    # --- 전체 선택 / 해제 버튼 ---
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("✅ 전체 선택"):
            st.session_state.df["select"] = True
    with col2:
        if st.button("❌ 전체 해제"):
            st.session_state.df["select"] = False

    # --- 데이터 테이블 표시 ---
    st.subheader("📋 논문 목록 (왼쪽 체크박스로 선택)")
    edited_df = st.data_editor(
        st.session_state.df,
        use_container_width=True,
        height=500,
        column_config={
            "select": st.column_config.CheckboxColumn("선택", help="스크리닝할 논문 선택"),
        },
        hide_index=True,
    )

    # 편집 내용 저장
    st.session_state.df = edited_df
    selected_rows = edited_df[edited_df["select"] == True]

    # --- 선택된 논문 표시 ---
    st.subheader("✅ 선택된 논문")
    if selected_rows.empty:
        st.info("먼저 스크리닝할 논문을 선택하세요.")
    elif not user_question.strip():
        st.warning("Screening 질문을 입력해주세요.")
    else:
        st.dataframe(selected_rows, use_container_width=True, height=300)

        # --- LLM 판단 실행 ---
        if st.button("🧠 Screening 실행"):
            total = len(selected_rows)

            # ⏱ 시작 시간 기록
            start_time = time.perf_counter()

            progress_text = st.empty()
            progress_bar = st.progress(0)
            st.info(f"{total}개의 논문에 대해 LLM 판단 중...")

            # 결과 컬럼 준비
            if result_col_name not in st.session_state.df.columns:
                st.session_state.df[result_col_name] = ""
            reason_col_name = f"{result_col_name}_reason"
            if show_reason and reason_col_name not in st.session_state.df.columns:
                st.session_state.df[reason_col_name] = ""

            results = []
            error_count = 0

            for i, (idx, row) in enumerate(selected_rows.iterrows(), start=1):
                title = str(row.get("Title", "제목 정보 없음")).strip()
                abstract = str(row.get("Abstract", "초록 정보 없음")).strip()

                prompt = f"""
당신은 연구 논문을 스크리닝하는 보조자입니다.
주어진 논문 정보와 초록을 기반으로 아래 질문에 대해 "Yes" 또는 "No"로 판단하세요.

[질문]
{user_question}

[논문 제목]
{title}

[논문 초록]
{abstract}

[출력 형식]
result: "Yes" 또는 "No" 중 하나
reason: 판단 이유를 2~3문장으로 설명
"""

                try:
                    reply = ""

                    # --- 모델별 호출 ---
                    if model_choice.startswith("gpt-4.1-mini"):
                        client = OpenAI(api_key=st.secrets["API"]["OPENAI_API_KEY"])
                        completion = client.responses.create(
                            model="gpt-4.1-mini",
                            input=prompt,
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                        reply = completion.output_text
                    elif model_choice.startswith("gpt-4.1-nano"):
                        client = OpenAI(api_key=st.secrets["API"]["OPENAI_API_KEY"])
                        completion = client.responses.create(
                            model="gpt-4.1-nano",
                            input=prompt,
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                        reply = completion.output_text
                    elif model_choice.startswith("gpt-5-mini"):
                        client = OpenAI(api_key=st.secrets["API"]["OPENAI_API_KEY"])
                        completion = client.responses.create(
                            model="gpt-4.1-mini",
                            input=prompt,
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                        reply = completion.output_text
                    elif model_choice.startswith("gpt-5-nano"):
                        client = OpenAI(api_key=st.secrets["API"]["OPENAI_API_KEY"])
                        completion = client.responses.create(
                            model="gpt-4.1-mini",
                            input=prompt,
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                        reply = completion.output_text
                    elif model_choice.startswith("claude-sonnet-4-5"):
                        client = anthropic.Anthropic(api_key=st.secrets["API"]["ANTHROPIC_API_KEY"])
                        completion = client.messages.create(
                            model="claude-sonnet-4-5",
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        reply = completion.content[0].text
                    elif model_choice.startswith("claude-haiku-4-5"):
                        client = anthropic.Anthropic(api_key=st.secrets["API"]["ANTHROPIC_API_KEY"])
                        completion = client.messages.create(
                            model="claude-sonnet-4-5",
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        reply = completion.content[0].text
                    elif model_choice.startswith("gemini-2.5-flash-lite"):
                        genai.configure(api_key=st.secrets["API"]["GEMINI_API_KEY"])
                        model = genai.GenerativeModel(
                            "gemini-2.5-flash-lite",
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": max_tokens,
                            },
                        )
                        completion = model.generate_content(prompt)
                        reply = completion.text

                    # --- 결과 추출 ---
                    result_value = "Yes" if "yes" in reply.lower() else "No"
                    reason_text = ""
                    if show_reason:
                        match = re.search(r"reason\s*[:：]\s*(.*)", reply, re.IGNORECASE | re.DOTALL)
                        if match:
                            reason_text = match.group(1).strip()
                        else:
                            reason_text = re.sub(r"result\s*[:：]\s*(yes|no)", "", reply, flags=re.IGNORECASE).strip()

                    # --- 결과 반영 ---
                    st.session_state.df.loc[idx, result_col_name] = result_value
                    if show_reason:
                        st.session_state.df.loc[idx, reason_col_name] = reason_text

                    results.append({
                        "index": idx,
                        "title": title,
                        "abstract": abstract,
                        result_col_name: result_value,
                        "reason": reason_text,
                    })

                except Exception:
                    # ✅ 오류 시: 해당 행의 결과 컬럼에 오류를 직접 기록하고 계속 진행
                    error_count += 1
                    st.session_state.df.loc[idx, result_col_name] = "Error"
                    if show_reason:
                        st.session_state.df.loc[idx, reason_col_name] = "LLM 모델 동작 중 에러가 발생했습니다"

                    results.append({
                        "index": idx,
                        "title": title,
                        "abstract": abstract,
                        result_col_name: "Error",
                        "reason": "LLM 모델 동작 중 에러가 발생했습니다",
                    })

                # ✅ 진행률 업데이트 (성공/오류와 무관하게 업데이트)
                percent = int(i / total * 100)
                progress_bar.progress(percent / 100)
                progress_text.markdown(f"**진행 중:** {i}/{total} ({percent}%) 완료")
                time.sleep(0.05)

            # ⏱ 경과 시간 계산
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            # 상태 저장
            st.session_state["results"] = results
            st.session_state["error_count"] = error_count

            progress_bar.empty()

            if error_count > 0:
                progress_text.markdown("⚠️ **처리 완료 (오류 포함)** — 일부 논문에서 오류가 발생했습니다. 결과 컬럼에서 'Error'로 확인하세요.")
                st.warning(
                    f"⚠️ 처리 완료: 총 {total}개 중 {error_count}개에서 오류 발생 — (경과 시간: {minutes:02d}:{seconds:02d}, 약 {elapsed:.1f}초)"
                )
            else:
                progress_text.markdown("✅ **모든 논문 처리 완료!**")
                st.success(
                    f"✅ 모든 논문 처리 완료! Screening 결과가 원본 CSV의 오른쪽에 추가되었습니다. (경과 시간: {minutes:02d}:{seconds:02d}, 약 {elapsed:.1f}초)"
                )

    # --- 결과 표시 & 즉시 CSV 다운로드 영역 ---
    if "results" in st.session_state and st.session_state["results"]:
        st.subheader("📊 Screening 결과 미리보기")
        st.dataframe(st.session_state.df, use_container_width=True, height=400)

        # ✅ CSV 다운로드
        st.subheader("📥 결과 CSV 다운로드")
        include_reason_csv = st.checkbox("결과 CSV에 판단 이유(reason) 포함", value=True)
        export_df = st.session_state.df.copy()
        export_df = export_df.drop(columns=["select"], errors="ignore")
        if not include_reason_csv:
            reason_cols = [c for c in export_df.columns if c.endswith("_reason")]
            export_df = export_df.drop(columns=reason_cols)
        csv = export_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button("⬇️ CSV 다운로드", csv, "screening_results.csv", "text/csv")

        # ✅ 세부 근거/페이지네이션
        total_items = len(st.session_state["results"])
        page_size = 50
        total_pages = max(1, math.ceil(total_items / page_size))

        if "current_page" not in st.session_state or not isinstance(st.session_state["current_page"], int):
            st.session_state["current_page"] = 1

        current_page = max(1, min(st.session_state["current_page"], total_pages))
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)

        st.subheader("🔍 세부 판단 근거 보기")
        st.caption(
            f"총 {total_items}개 중 {start_idx + 1}–{end_idx} 표시 (페이지 {current_page}/{total_pages}, 페이지당 {page_size}개)"
        )

        page_slice = st.session_state["results"][start_idx:end_idx]
        for r in page_slice:
            color = "🟩" if r[result_col_name] == "Yes" else ("🟥" if r[result_col_name] == "No" else "🟨")
            st.markdown(f"### {color} **{r[result_col_name]}** — {r['title']}")
            st.markdown(f"📄 **Abstract:** {r['abstract'][:300]}{'...' if len(r['abstract']) > 300 else ''}")
            if show_reason and "reason" in r:
                with st.expander("💡 판단 근거 보기"):
                    st.write(r["reason"])
            st.divider()

        # ✅ 페이지 버튼
        st.write("페이지:")
        pages_per_row = 10
        rows = math.ceil(total_pages / pages_per_row)
        for row in range(rows):
            start = row * pages_per_row + 1
            end = min((row + 1) * pages_per_row, total_pages)
            cols = st.columns(end - start + 1)
            for i, p in enumerate(range(start, end + 1)):
                with cols[i]:
                    if st.button(f"{p}", key=f"page_{p}"):
                        st.session_state["current_page"] = p
                        st.rerun()

        # ✅ 이전/다음/처음/마지막
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("⏮ 처음"):
                if current_page != 1:
                    st.session_state["current_page"] = 1
                    st.rerun()
        with c2:
            if st.button("◀ 이전"):
                if current_page > 1:
                    st.session_state["current_page"] = current_page - 1
                    st.rerun()
        with c3:
            if st.button("다음 ▶"):
                if current_page < total_pages:
                    st.session_state["current_page"] = current_page + 1
                    st.rerun()
        with c4:
            if st.button("마지막 ⏭"):
                if current_page != total_pages:
                    st.session_state["current_page"] = total_pages
                    st.rerun()

else:
    st.info("📂 CSV 파일을 업로드하면 논문 스크리닝을 시작할 수 있습니다.")
