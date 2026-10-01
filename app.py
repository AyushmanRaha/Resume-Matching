"""
Resume <-> JD Matcher — local, single-process app.

Run with:
    streamlit run app.py

No database server, no queue, no external API required (aside from the
one-time model download the first time you run it).
"""

import streamlit as st
from parsing import extract_text
from extraction import extract_jd_profile, extract_resume_profile
from matching import score_candidate

st.set_page_config(page_title="Resume Matcher", layout="wide")

st.title("Resume ↔ Job Description Matcher")
st.caption("Runs fully locally. Upload a job description and a batch of resumes to get a ranked, evidence-backed match score.")

# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Job description")
    jd_input_mode = st.radio("Provide JD as", ["Paste text", "Upload file"], horizontal=True)
    jd_text = ""
    if jd_input_mode == "Paste text":
        jd_text = st.text_area("Paste the job description here", height=250)
    else:
        jd_file = st.file_uploader("Upload JD (pdf, docx, txt)", type=["pdf", "docx", "txt"])
        if jd_file is not None:
            jd_text = extract_text(jd_file.name, jd_file.read())

    st.header("2. Resumes")
    resume_files = st.file_uploader(
        "Upload one or more resumes (pdf, docx, txt)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    run_button = st.button("Run matching", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main: results
# ---------------------------------------------------------------------------
if run_button:
    if not jd_text.strip():
        st.error("Please provide a job description first.")
        st.stop()
    if not resume_files:
        st.error("Please upload at least one resume.")
        st.stop()

    with st.spinner("Parsing job description..."):
        jd_profile = extract_jd_profile(jd_text)

    st.subheader("Job description — extracted requirements")
    c1, c2, c3 = st.columns(3)
    c1.metric("Skills detected", len(jd_profile["skills"]))
    c2.metric("Required experience", f"{jd_profile['required_years']:.0f} yrs" if jd_profile["required_years"] else "Not specified")
    c3.metric("Required education", jd_profile["required_education"] or "Not specified")
    if jd_profile["mandatory_skills"]:
        st.caption(f"Mandatory skills detected: {', '.join(sorted(jd_profile['mandatory_skills']))}")

    results = []
    progress = st.progress(0.0, text="Scoring resumes...")
    for i, f in enumerate(resume_files):
        text = extract_text(f.name, f.read())
        if not text.strip():
            st.warning(f"Could not extract text from **{f.name}** — skipping (unsupported format or scanned image without OCR).")
            continue
        resume_profile = extract_resume_profile(text, filename=f.name)
        result = score_candidate(jd_profile, resume_profile)
        results.append(result)
        progress.progress((i + 1) / len(resume_files), text=f"Scored {f.name}")
    progress.empty()

    if not results:
        st.error("No resumes could be parsed. Check the file formats and try again.")
        st.stop()

    results.sort(key=lambda r: r["total_score"], reverse=True)

    st.subheader("Ranked candidates")
    table_rows = []
    for rank, r in enumerate(results, start=1):
        n_missing = len(r["skills"]["missing_mandatory"])
        flag = f"⚠️ {n_missing} mandatory skill(s) missing" if n_missing else ""
        table_rows.append({
            "Rank": rank,
            "Resume": r["filename"],
            "Overall score": f"{r['total_score']}%",
            "Skills": f"{r['skills']['score']}%",
            "Experience": f"{r['experience']['score']}%",
            "Education": f"{r['education']['score']}%",
            "Flag": flag,
        })
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Candidate detail & evidence")
    for rank, r in enumerate(results, start=1):
        with st.expander(f"#{rank} — {r['filename']} — {r['total_score']}%"):
            colA, colB = st.columns(2)

            with colA:
                st.markdown("**Matched skills (exact)**")
                if r["skills"]["exact_matches"]:
                    for s in r["skills"]["exact_matches"]:
                        st.markdown(f"- ✅ **{s}**")
                        evidence = r["skills"]["exact_match_evidence"].get(s, [])
                        for line in evidence:
                            st.caption(f"  ↳ \"{line}\"")
                else:
                    st.caption("None")

                st.markdown("**Related / semantic matches**")
                if r["skills"]["semantic_matches"]:
                    for jd_skill, (resume_skill, sim) in r["skills"]["semantic_matches"].items():
                        st.markdown(f"- 🟡 JD wants **{jd_skill}** → resume has **{resume_skill}** (similarity {sim})")
                else:
                    st.caption("None")

            with colB:
                st.markdown("**Missing skills**")
                if r["skills"]["missing"]:
                    for s in r["skills"]["missing"]:
                        mandatory_tag = " (mandatory)" if s in r["skills"]["missing_mandatory"] else ""
                        st.markdown(f"- ❌ {s}{mandatory_tag}")
                else:
                    st.caption("None — all JD skills covered")

                st.markdown("**Experience & education**")
                st.markdown(
                    f"- Experience: required {r['experience']['required']:.0f} yrs, "
                    f"candidate ~{r['experience']['candidate']:.0f} yrs "
                    f"({r['experience']['score']}%)"
                )
                st.markdown(
                    f"- Education: required {r['education']['required'] or 'n/a'}, "
                    f"candidate {r['education']['candidate'] or 'not detected'} "
                    f"({r['education']['score']}%)"
                )

else:
    st.info("Add a job description and resumes in the sidebar, then click **Run matching**.")
