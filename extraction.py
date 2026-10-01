"""
Turns raw resume/JD text into structured fields.

The extraction pipeline is deliberately local and explainable:

1. Normalize document text.
2. Detect sections.
3. Extract skills with evidence.
4. Extract and normalize dates.
5. Extract structured employment entries.
6. Calculate total experience without double-counting.
7. Extract education using contextual evidence.
8. Extract mandatory JD skills.

No external API or LLM is required.
"""

import re
from datetime import datetime

from skills_data import (
    SYNONYM_TO_CANONICAL,
    EDUCATION_LEVELS,
    MANDATORY_GATE_EXCLUDED,
)


CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month


# ---------------------------------------------------------------------------
# General normalization
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize Unicode punctuation and whitespace without destroying lines."""
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2212", "-")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2022", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _phrase_pattern(phrase: str) -> re.Pattern:
    escaped = re.escape(phrase.strip())
    return re.compile(
        r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])",
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def find_skills(text: str):
    """Return canonical skills with up to two evidence lines each."""
    text = _normalize_text(text)
    lines = text.splitlines()
    found = {}

    for phrase, canonical in SYNONYM_TO_CANONICAL.items():
        pattern = _phrase_pattern(phrase)

        if not pattern.search(text):
            continue

        if canonical not in found:
            found[canonical] = []

        for line in lines:
            if pattern.search(line) and len(found[canonical]) < 2:
                clean_line = line.strip()

                if clean_line and clean_line not in found[canonical]:
                    found[canonical].append(clean_line)

    return found


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

SECTION_ALIASES = {
    "experience": {
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "career history",
        "work history",
        "professional background",
        "relevant experience",
        "industry experience",
    },
    "education": {
        "education",
        "academic background",
        "academic qualifications",
        "educational qualifications",
        "academics",
        "education and training",
        "education & training",
    },
    "skills": {
        "skills",
        "technical skills",
        "core skills",
        "key skills",
        "technical expertise",
        "competencies",
    },
    "projects": {
        "projects",
        "selected projects",
        "key projects",
        "academic projects",
        "personal projects",
    },
    "certifications": {
        "certifications",
        "certificates",
        "licenses and certifications",
        "licenses & certifications",
    },
    "other": {
        "summary",
        "profile",
        "objective",
        "publications",
        "awards",
        "interests",
        "volunteer",
        "languages",
        "associations",
        "extracurricular",
        "achievements",
        "references",
        "courses",
        "activities",
        "hobbies",
        "contact",
    },
}


def _is_header_line(line: str):
    stripped = re.sub(r"[:|•\-]+$", "", line.strip())
    stripped = re.sub(r"^[•\-|]+", "", stripped).strip().lower()

    if not stripped:
        return None

    if len(stripped.split()) > 7:
        return None

    for section, aliases in SECTION_ALIASES.items():
        if stripped in aliases:
            return section

    # Handle common decorated headers such as:
    # "PROFESSIONAL EXPERIENCE:"
    compact = re.sub(r"[^a-z0-9& ]", "", stripped)

    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            alias_compact = re.sub(r"[^a-z0-9& ]", "", alias)

            if compact == alias_compact:
                return section

    return None


def _extract_section(text: str, target_section: str) -> str:
    lines = text.splitlines()
    collected = []
    current = None

    for line in lines:
        section = _is_header_line(line)

        if section is not None:
            current = section
            continue

        if current == target_section:
            collected.append(line)

    return "\n".join(collected).strip()


def _extract_experience_section(text: str) -> str:
    return _extract_section(text, "experience")


def _extract_education_section(text: str) -> str:
    return _extract_section(text, "education")


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

MONTHS_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

ONGOING_WORDS = {
    "present",
    "current",
    "now",
    "ongoing",
    "till date",
    "to date",
    "until date",
    "till now",
    "to present",
}


DATE_TOKEN = rf"""
(?:
    (?:{MONTHS_PATTERN})\.?\s*[,'-]?\s*\d{{2,4}}
    |
    \d{{1,2}}\s+(?:{MONTHS_PATTERN})\.?\s*,?\s*\d{{2,4}}
    |
    \d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}
    |
    \d{{1,2}}[./-]\d{{2,4}}
    |
    \d{{4}}[./-]\d{{1,2}}
    |
    \d{{4}}
    |
    (?:present|current|now|ongoing|till\s+date|to\s+date|until\s+date|till\s+now)
)
"""

DATE_RANGE_RE = re.compile(
    rf"(?P<start>{DATE_TOKEN})"
    rf"\s*(?:-|–|—|to|until|through|thru)\s*"
    rf"(?P<end>{DATE_TOKEN})",
    re.IGNORECASE | re.VERBOSE,
)

DATE_SINGLE_RE = re.compile(
    rf"(?P<date>{DATE_TOKEN})",
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_year(year: int) -> int:
    """Convert two-digit years to a sensible 19xx/20xx value."""
    if year < 100:
        return 2000 + year if year <= 40 else 1900 + year

    return year


def _parse_date_token(raw: str):
    """
    Return (year, month, precision, ongoing).

    precision is:
      "month" for month/year dates
      "year" for year-only dates
    """
    value = raw.strip().lower()
    value = value.replace(",", "")
    value = re.sub(r"\s+", " ", value)

    if value in ONGOING_WORDS:
        return CURRENT_YEAR, CURRENT_MONTH, "month", True

    # 1 January 2020
    match = re.fullmatch(
        rf"(\d{{1,2}})\s+({MONTHS_PATTERN})\.?\s+(\d{{2,4}})",
        value,
        re.IGNORECASE,
    )

    if match:
        month = MONTH_LOOKUP[match.group(2).lower().rstrip(".")]
        year = _normalize_year(int(match.group(3)))
        return year, month, "month", False

    # January 2020 / Jan'20
    match = re.fullmatch(
        rf"({MONTHS_PATTERN})\.?\s*[']?\s*(\d{{2,4}})",
        value,
        re.IGNORECASE,
    )

    if match:
        month = MONTH_LOOKUP[match.group(1).lower().rstrip(".")]
        year = _normalize_year(int(match.group(2)))
        return year, month, "month", False

    # MM/YYYY or MM-YYYY
    match = re.fullmatch(r"(\d{1,2})[./-](\d{2,4})", value)

    if match:
        month = int(match.group(1))
        year = _normalize_year(int(match.group(2)))

        if 1 <= month <= 12:
            return year, month, "month", False

    # YYYY/MM
    match = re.fullmatch(r"(\d{4})[./-](\d{1,2})", value)

    if match:
        year = int(match.group(1))
        month = int(match.group(2))

        if 1 <= month <= 12:
            return year, month, "month", False

    # Full numeric date
    match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", value)

    if match:
        month = int(match.group(2))
        year = _normalize_year(int(match.group(3)))

        if 1 <= month <= 12:
            return year, month, "month", False

        month = int(match.group(1))

        if 1 <= month <= 12:
            return year, month, "month", False

    # Year only
    match = re.fullmatch(r"(19\d{2}|20\d{2})", value)

    if match:
        return int(match.group(1)), None, "year", False

    return None


def _date_to_index(parsed):
    if parsed is None:
        return None

    year, month, _, _ = parsed

    if month is None:
        return year * 12

    return year * 12 + month - 1


def _parse_date_range(start_raw: str, end_raw: str):
    start = _parse_date_token(start_raw)
    end = _parse_date_token(end_raw)

    if start is None or end is None:
        return None

    start_index = _date_to_index(start)
    end_index = _date_to_index(end)

    if end_index < start_index:
        return None

    return {
        "start_raw": start_raw.strip(),
        "end_raw": end_raw.strip(),
        "start": start,
        "end": end,
        "start_index": start_index,
        "end_index": end_index,
        "confidence": 0.95,
    }


def _extract_date_ranges(text: str):
    ranges = []

    for match in DATE_RANGE_RE.finditer(text):
        parsed = _parse_date_range(
            match.group("start"),
            match.group("end"),
        )

        if parsed is not None:
            ranges.append(parsed)

    return ranges


# ---------------------------------------------------------------------------
# Experience extraction
# ---------------------------------------------------------------------------

JOB_TITLE_TERMS = {
    "engineer",
    "developer",
    "programmer",
    "analyst",
    "scientist",
    "architect",
    "consultant",
    "manager",
    "director",
    "designer",
    "administrator",
    "specialist",
    "intern",
    "trainee",
    "associate",
    "lead",
    "principal",
    "officer",
    "executive",
    "coordinator",
    "researcher",
    "assistant",
    "supervisor",
    "representative",
    "technician",
}


def _looks_like_job_title(text: str) -> bool:
    lower = text.lower()

    return any(
        re.search(rf"\b{re.escape(term)}\b", lower)
        for term in JOB_TITLE_TERMS
    )

def _looks_like_employment_context(text: str) -> bool:
    lower = text.lower()

    employment_terms = [
        "worked at",
        "worked with",
        "employed at",
        "joined",
        "promoted",
        "responsible for",
        "responsibilities",
        "professional experience",
        "work experience",
        "employment",
        "company",
        "organization",
        "corporation",
        "technologies",
        "client",
        "clients",
        "role",
        "position",
        "designation",
        "salary",
        "full-time",
        "full time",
        "part-time",
        "part time",
    ]

    return any(
        re.search(rf"\b{re.escape(term)}\b", lower)
        for term in employment_terms
    )


def _is_project_or_non_employment_context(lines, index):
    context = " ".join(
        lines[max(0, index - 3):min(len(lines), index + 4)]
    ).lower()

    excluded_terms = [
        "education",
        "university",
        "college",
        "school",
        "academic",
        "course",
        "coursework",
        "certification",
        "certificate",
        "publication",
        "project",
        "academic project",
        "personal project",
        "final year project",
        "minor project",
        "major project",
        "thesis",
        "dissertation",
        "internship project",
        "research project",
    ]

    return any(
        re.search(rf"\b{re.escape(term)}\b", context)
        for term in excluded_terms
    )


def _extract_experience_entries(text: str):
    """
    Extract employment date ranges using positive employment evidence
    while excluding education, projects and other non-employment contexts.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    entries = []

    for index, line in enumerate(lines):
        candidate_lines = lines[
            max(0, index - 3):min(len(lines), index + 4)
        ]

        nearby = " ".join(candidate_lines)

        matches = list(DATE_RANGE_RE.finditer(line))

        if not matches and index + 1 < len(lines):
            combined = line + " " + lines[index + 1]
            matches = list(DATE_RANGE_RE.finditer(combined))

        for match in matches:
            parsed = _parse_date_range(
                match.group("start"),
                match.group("end"),
            )

            if parsed is None:
                continue

            title_evidence = _looks_like_job_title(nearby)
            employment_evidence = _looks_like_employment_context(nearby)

            # A clear job title right next to the date is stronger evidence
            # of real employment than a stray institution word is evidence
            # of an education entry — e.g. an employer literally named
            # "... University", or a bullet mentioning "university policy",
            # shouldn't disqualify an otherwise obvious job entry.
            if _is_project_or_non_employment_context(
                lines,
                index,
            ) and not title_evidence:
                continue

            confidence = 0.50

            if title_evidence:
                confidence += 0.20

            if employment_evidence:
                confidence += 0.15

            if _extract_experience_section(text).strip():
                confidence += 0.10

            if title_evidence and employment_evidence:
                confidence += 0.05

            confidence = min(confidence, 0.98)

            # Require meaningful employment evidence.
            if not title_evidence and not employment_evidence:
                continue

            entries.append({
                "start_index": parsed["start_index"],
                "end_index": parsed["end_index"],
                "start_raw": parsed["start_raw"],
                "end_raw": parsed["end_raw"],
                "confidence": confidence,
                "evidence": nearby,
            })

    # Remove duplicate ranges produced by adjacent-line reconstruction.
    unique = {}

    for entry in entries:
        key = (
            entry["start_index"],
            entry["end_index"],
        )

        if (
            key not in unique
            or entry["confidence"] > unique[key]["confidence"]
        ):
            unique[key] = entry

    return list(unique.values())


def _merge_index_ranges(ranges):
    if not ranges:
        return []

    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [list(ranges[0])]

    for start, end in ranges[1:]:
        previous = merged[-1]

        if start <= previous[1] + 1:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    return merged


def _ranges_to_years(ranges):
    merged = _merge_index_ranges(ranges)

    if not merged:
        return 0.0

    total_months = sum(
        end - start + 1
        for start, end in merged
    )

    return round(total_months / 12.0, 1)


def _extract_explicit_experience(text: str):
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s*(?:of\s+)?(?:professional\s+|total\s+|relevant\s+)?experience",
        r"experience\s*(?:required)?\s*:?\s*(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)",
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+in\s+(?:software|technology|IT|engineering|development)",
    ]

    values = []

    for pattern in patterns:
        for value in re.findall(pattern, text.lower()):
            values.append(float(value))

    return max(values) if values else None


def extract_years_experience(text: str) -> float:
    """
    Extract total professional experience.

    Priority:
    1. Explicit experience statement.
    2. Structured date ranges inside Experience.
    3. Whole-document date ranges only when no experience section exists.
    """
    text = _normalize_text(text)

    explicit = _extract_explicit_experience(text)

    if explicit is not None:
        return explicit

    experience_section = _extract_experience_section(text)

    if experience_section.strip():
        entries = _extract_experience_entries(experience_section)

        if entries:
            ranges = [
                (entry["start_index"], entry["end_index"])
                for entry in entries
            ]

            return _ranges_to_years(ranges)

    # No reliable experience section (none was detected, or it had no
    # entries with real employment evidence). Fall back to scanning the
    # whole document — _extract_experience_entries still requires a job
    # title or employment-context clue and excludes education/project
    # contexts, so this stays safe rather than counting arbitrary dates.
    entries = _extract_experience_entries(text)

    if entries:
        ranges = [
            (entry["start_index"], entry["end_index"])
            for entry in entries
        ]

        return _ranges_to_years(ranges)

    return 0.0


# ---------------------------------------------------------------------------
# Education extraction
# ---------------------------------------------------------------------------

EDUCATION_PATTERNS = {
    "PhD": [
        r"\bph\.?\s*d\.?\b",
        r"\bdoctor(?:ate)?\s+of\s+philosophy\b",
        r"\bdoctorate\b",
    ],
    "Master": [
        r"\bmaster(?:'s)?\b",
        r"\bmasters\b",
        r"\bm\.?\s*s\.?\b",
        r"\bm\.?\s*sc\.?\b",
        r"\bm\.?\s*tech\.?\b",
        r"\bm\.?\s*e\.?\b",
        r"\bm\.?\s*a\.?\b",
        r"\bm\.?\s*com\.?\b",
        r"\bmca\b",
        r"\bmba\b",
    ],
    "Bachelor": [
        r"\bbachelor(?:'s)?\b",
        r"\bbachelors\b",
        r"\bb\.?\s*s\.?\b",
        r"\bb\.?\s*sc\.?\b",
        r"\bb\.?\s*tech\.?\b",
        r"\bb\.?\s*e\.?\b",
        r"\bb\.?\s*a\.?\b",
        r"\bb\.?\s*com\.?\b",
        r"\bbca\b",
        r"\bbba\b",
    ],
    "Diploma": [
        r"\bpg\s*diploma\b",
        r"\bpost\s*graduate\s+diploma\b",
        r"\bdiploma\b",
        r"\bassociate\s+degree\b",
    ],
    "High School": [
        r"\bhigh\s+school\b",
        r"\bsecondary\s+school\b",
        r"\bhigher\s+secondary\b",
        r"\bhsc\b",
        r"\bssc\b",
        r"\bclass\s+(?:10|12|x|xii)\b",
    ],
}


EDUCATION_RANKS = {
    "High School": 1,
    "Diploma": 2,
    "Bachelor": 3,
    "Master": 4,
    "PhD": 5,
}


def _education_candidates(text: str):
    section = _extract_education_section(text)

    search_text = section if section.strip() else text

    candidates = []

    for level, patterns in EDUCATION_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, search_text, re.IGNORECASE):
                start = max(0, match.start() - 100)
                end = min(len(search_text), match.end() + 180)

                context = search_text[start:end]

                # Penalize tentative/future education language.
                negative_context = re.search(
                    r"\b(pursuing|planning|plan to|aspiring|interested in)\b",
                    context,
                    re.IGNORECASE,
                )

                confidence = 0.88

                if section.strip():
                    confidence += 0.08

                if negative_context:
                    confidence -= 0.35

                candidates.append({
                    "level": level,
                    "rank": EDUCATION_RANKS[level],
                    "confidence": max(0.0, min(confidence, 0.99)),
                    "evidence": context.strip(),
                })

    return candidates


def extract_education_details(text: str):
    candidates = _education_candidates(text)

    if not candidates:
        return {
            "level": None,
            "rank": 0,
            "confidence": 0.0,
            "evidence": [],
        }

    candidates.sort(
        key=lambda item: (
            item["rank"],
            item["confidence"],
        ),
        reverse=True,
    )

    best = candidates[0]

    return {
        "level": best["level"],
        "rank": best["rank"],
        "confidence": round(best["confidence"], 2),
        "evidence": [best["evidence"]],
    }


def extract_education_level(text: str):
    details = extract_education_details(text)

    return details["level"], details["rank"]


# ---------------------------------------------------------------------------
# Mandatory JD skills
# ---------------------------------------------------------------------------

def extract_mandatory_skills(jd_text: str, all_found_skills):
    mandatory_markers = [
        "must have",
        "must-have",
        "required skill",
        "required skills",
        "mandatory",
        "minimum requirement",
        "minimum requirements",
        "required qualifications",
        "requirements",
    ]

    preferred_markers = [
        "preferred",
        "nice to have",
        "nice-to-have",
        "bonus",
        "good to have",
        "optional",
    ]

    lines = jd_text.splitlines()

    mandatory = set()
    mode = "neutral"
    blank_streak = 0

    for raw_line in lines:
        line = raw_line.strip()
        line_lower = line.lower()

        if not line:
            blank_streak += 1

            if blank_streak >= 2:
                mode = "neutral"

            continue

        blank_streak = 0

        if any(marker in line_lower for marker in preferred_markers):
            mode = "preferred"

        elif any(marker in line_lower for marker in mandatory_markers):
            mode = "mandatory"

        if mode != "mandatory":
            continue

        for canonical in all_found_skills.keys():
            if canonical in MANDATORY_GATE_EXCLUDED:
                continue

            synonyms = [
                phrase
                for phrase, skill in SYNONYM_TO_CANONICAL.items()
                if skill == canonical
            ]

            if any(
                _phrase_pattern(synonym).search(line)
                for synonym in synonyms
            ):
                mandatory.add(canonical)

    return mandatory


# ---------------------------------------------------------------------------
# Public profile builders
# ---------------------------------------------------------------------------

def extract_jd_profile(jd_text: str) -> dict:
    jd_text = _normalize_text(jd_text)

    skills = find_skills(jd_text)
    mandatory = extract_mandatory_skills(jd_text, skills)
    years = extract_years_experience(jd_text)

    edu_details = extract_education_details(jd_text)

    return {
        "skills": skills,
        "mandatory_skills": mandatory,
        "required_years": years,
        "required_education": edu_details["level"],
        "required_education_rank": edu_details["rank"],
        "education_confidence": edu_details["confidence"],
        "education_evidence": edu_details["evidence"],
        "raw_text": jd_text,
    }


def extract_resume_profile(resume_text: str, filename: str = "") -> dict:
    resume_text = _normalize_text(resume_text)

    skills = find_skills(resume_text)

    experience_section = _extract_experience_section(resume_text)
    experience_entries = _extract_experience_entries(
        experience_section
    ) if experience_section.strip() else _extract_experience_entries(resume_text)

    years = extract_years_experience(resume_text)

    edu_details = extract_education_details(resume_text)

    return {
        "filename": filename,
        "skills": skills,

        # Existing public field.
        "years_experience": years,

        # New structured information used internally by matching.
        "experience_entries": experience_entries,
        "experience_confidence": round(
            max(
                [entry["confidence"] for entry in experience_entries],
                default=0.0,
            ),
            2,
        ),

        # Existing public fields.
        "education": edu_details["level"],
        "education_rank": edu_details["rank"],

        # New evidence fields; app.py does not need to display them.
        "education_confidence": edu_details["confidence"],
        "education_evidence": edu_details["evidence"],

        "raw_text": resume_text,
    }