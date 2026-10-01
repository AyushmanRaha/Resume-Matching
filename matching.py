"""
The matching engine. Combines:
  - exact skill matching (from extraction.py)
  - semantic matching for JD skills that have no exact resume match
    (catches things like "FastAPI" vs "Flask" — related but not identical)
  - rule-based scoring for experience and education
  - a weighted overall score, plus evidence for every match

Runs fully locally via sentence-transformers — no API calls, no internet
needed after the model is downloaded once.
"""

from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer, util
from skills_data import RELATED_SKILLS

# Small, fast, runs comfortably on CPU. Swap for "BAAI/bge-small-en-v1.5"
# if you want slightly better semantic quality at a small speed cost.
MODEL_NAME = "all-MiniLM-L6-v2"

# Below this cosine similarity, we don't consider it a match at all.
SEMANTIC_MATCH_THRESHOLD = 0.60
# Above this, we consider it a strong / "related technology" match.
SEMANTIC_STRONG_THRESHOLD = 0.70

WEIGHTS = {
    "skills": 0.50,
    "experience": 0.20,
    "education": 0.15,
    "responsibilities": 0.15,  # overall semantic similarity of resume vs JD body text
}


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(MODEL_NAME)


def semantic_match_missing_skills(
    missing_jd_skills,
    resume_skill_names,
    model,
):
    """
    Find related skills without treating generic semantic similarity
    as equivalent to an exact skill.
    """
    results = {}

    if not missing_jd_skills or not resume_skill_names:
        for skill in missing_jd_skills:
            results[skill] = None

        return results

    jd_list = list(missing_jd_skills)
    resume_list = list(resume_skill_names)

    jd_emb = model.encode(
        jd_list,
        convert_to_tensor=True,
    )

    resume_emb = model.encode(
        resume_list,
        convert_to_tensor=True,
    )

    similarities = util.cos_sim(
        jd_emb,
        resume_emb,
    ).cpu().numpy()

    for index, jd_skill in enumerate(jd_list):
        best_index = int(np.argmax(similarities[index]))
        best_skill = resume_list[best_index]
        best_score = float(similarities[index][best_index])

        explicitly_related = (
            best_skill in RELATED_SKILLS.get(jd_skill, [])
        )

        # Generic semantic similarity alone must be stronger before it
        # can be treated as a related match.
        semantic_related = best_score >= 0.60

        if explicitly_related or semantic_related:
            results[jd_skill] = (
                best_skill,
                round(best_score, 2),
            )
        else:
            results[jd_skill] = None

    return results


def score_skills(jd_profile, resume_profile, model):
    jd_skills = set(jd_profile["skills"].keys())
    resume_skills = set(resume_profile["skills"].keys())
    mandatory = jd_profile["mandatory_skills"]

    exact_matches = jd_skills & resume_skills
    missing = jd_skills - resume_skills

    exact_match_evidence = {
        skill: resume_profile["skills"].get(skill, [])
        for skill in exact_matches
    }
    semantic_results = semantic_match_missing_skills(missing, resume_skills, model)
    semantic_matches = {k: v for k, v in semantic_results.items() if v is not None}
    still_missing = {k for k, v in semantic_results.items() if v is None}

    # scoring: exact match = full credit, semantic/related match = partial credit
    # Exact matches receive full credit.
# Related/semantic matches receive partial credit only.
    if jd_skills:
        exact_credit = len(exact_matches)
        semantic_credit = 0.6 * len(semantic_matches)

        credit = exact_credit + semantic_credit

        skill_score = min(
            100.0,
            100.0 * credit / len(jd_skills),
        )
    else:
        skill_score = 100.0

    # hard gate: missing mandatory skills reduces the ceiling on the skill
    # score, scaled by how many are missing rather than a flat penalty —
    # missing 1 of 8 mandatory skills shouldn't score the same as missing 7 of 8.
    missing_mandatory = mandatory & still_missing
    gated = False
    if missing_mandatory and mandatory:
        missing_fraction = len(missing_mandatory) / len(mandatory)
        # scales the cap from ~95 (barely penalized) down to 35 (missing everything)
        cap = 95.0 - (missing_fraction * 60.0)
        if skill_score > cap:
            skill_score = cap
            gated = True

    return {
        "score": round(skill_score, 1),
        "exact_matches": sorted(exact_matches),
        "exact_match_evidence": exact_match_evidence,
        "semantic_matches": semantic_matches,   # {jd_skill: (resume_skill, similarity)}
        "missing": sorted(still_missing),
        "missing_mandatory": sorted(missing_mandatory),
        "gated": gated,
    }


def score_experience(jd_profile, resume_profile):
    required = jd_profile.get("required_years")
    candidate = resume_profile.get("years_experience")

    if required is None:
        required = 0.0

    if candidate is None:
        candidate = 0.0

    if required <= 0:
        return {
            "score": 100.0,
            "required": required,
            "candidate": candidate,
        }

    score = min(
        100.0,
        100.0 * candidate / required,
    )

    confidence = resume_profile.get(
        "experience_confidence",
        0.0,
    )

    # Confidence only prevents uncertain extraction from receiving a
    # completely unjustified perfect score.
    if confidence > 0:
        score *= 0.85 + (0.15 * confidence)

    return {
        "score": round(score, 1),
        "required": required,
        "candidate": candidate,
    }


def score_education(jd_profile, resume_profile):
    required_rank = jd_profile["required_education_rank"]
    candidate_rank = resume_profile["education_rank"]

    if required_rank == 0:
        return {
            "score": 100.0,
            "required": jd_profile["required_education"],
            "candidate": resume_profile["education"],
        }

    if candidate_rank >= required_rank:
        score = 100.0

    elif candidate_rank == required_rank - 1:
        score = 60.0

    elif candidate_rank > 0:
        score = 30.0

    else:
        score = 0.0

    confidence = resume_profile.get(
        "education_confidence",
        0.0,
    )

    if score > 0 and confidence > 0:
        score *= 0.85 + (0.15 * confidence)

    return {
        "score": round(score, 1),
        "required": jd_profile["required_education"],
        "candidate": resume_profile["education"],
    }


def score_responsibilities(jd_profile, resume_profile, model):
    """
    Compare JD requirements against the candidate's actual experience
    entries rather than blindly comparing the first 2,000 characters.
    """
    jd_text = jd_profile["raw_text"]

    experience_entries = resume_profile.get(
        "experience_entries",
        [],
    )

    if not jd_text.strip():
        return {"score": 50.0}

    if not experience_entries:
        return {"score": 50.0}

    # Use the actual evidence associated with each employment entry.
    resume_chunks = [
        entry["evidence"]
        for entry in experience_entries
        if entry.get("evidence")
    ]

    if not resume_chunks:
        return {"score": 50.0}

    jd_emb = model.encode(
        [jd_text],
        convert_to_tensor=True,
    )

    resume_emb = model.encode(
        resume_chunks,
        convert_to_tensor=True,
    )

    similarities = util.cos_sim(
        jd_emb,
        resume_emb,
    ).cpu().numpy()[0]

    # Strongest relevant experience is more meaningful than averaging
    # unrelated resume sections.
    best_similarity = float(np.max(similarities))

    score = max(
        0.0,
        min(
            100.0,
            (best_similarity / 0.7) * 100.0,
        ),
    )

    return {
        "score": round(score, 1),
        "best_similarity": round(best_similarity, 2),
    }


def score_candidate(jd_profile, resume_profile):
    model = get_model()

    skills_result = score_skills(jd_profile, resume_profile, model)
    exp_result = score_experience(jd_profile, resume_profile)
    edu_result = score_education(jd_profile, resume_profile)
    resp_result = score_responsibilities(jd_profile, resume_profile, model)

    total = (
        skills_result["score"] * WEIGHTS["skills"]
        + exp_result["score"] * WEIGHTS["experience"]
        + edu_result["score"] * WEIGHTS["education"]
        + resp_result["score"] * WEIGHTS["responsibilities"]
    )

    return {
        "filename": resume_profile["filename"],
        "total_score": round(total, 1),
        "skills": skills_result,
        "experience": exp_result,
        "education": edu_result,
        "responsibilities": resp_result,
    }
