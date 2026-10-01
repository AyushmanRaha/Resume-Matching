"""
A lightweight skills taxonomy used for exact + synonym-normalized matching.
This is intentionally simple (a Python dict), so you can extend it freely —
just add more entries. No database or external service required.

Format:
  CANONICAL_SKILL: [list of synonyms / alternate spellings / related terms]

The canonical name is what gets shown in the UI. Synonyms are lowercased
and matched against resume/JD text.
"""

SKILL_SYNONYMS = {
    # Programming languages
    "Python": ["python", "python3", "python 3"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "es6", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    "Go": ["golang", "go language"],
    "Ruby": ["ruby"],
    "PHP": ["php"],
    "SQL": ["sql", "structured query language"],
    "R": ["r programming", "r language"],

    # Backend
    "FastAPI": ["fastapi", "fast api"],
    "Flask": ["flask"],
    "Django": ["django"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Express.js": ["express.js", "expressjs", "express js"],
    "Spring Boot": ["spring boot", "springboot"],
    "REST API": [
        "rest api",
        "rest apis",
        "restful api",
        "restful apis",
        "restful services",
    ],
    "GraphQL": ["graphql"],
    "Microservices": ["microservices", "microservice architecture"],

    # Frontend
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular", "angularjs"],
    "Vue.js": ["vue.js", "vuejs"],
    "Next.js": ["next.js", "nextjs"],
    "HTML/CSS": ["html", "css", "html5", "css3"],

    # Cloud / DevOps
    "Docker": ["docker", "containerization", "containerisation"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud platform"],
    "CI/CD": [
        "ci/cd",
        "continuous integration",
        "continuous deployment",
        "continuous delivery",
    ],
    "Terraform": ["terraform", "infrastructure as code"],
    "Linux": ["linux", "unix"],

    # Databases
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo db", "mongo"],
    "Redis": ["redis"],
    "Elasticsearch": ["elasticsearch", "elastic search"],
    "Kafka": ["kafka", "apache kafka"],

    # Data / AI
    "Machine Learning": ["machine learning"],
    "Deep Learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn"],

    # Engineering / process
    "Git": ["git", "git version control"],
    "GitHub": ["github"],
    "GitLab": ["gitlab"],
    "Bitbucket": ["bitbucket"],
    "Agile/Scrum": ["agile", "scrum", "kanban"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Project Management": ["project management", "pmp"],
    "Communication": ["communication skills", "stakeholder management"],
}

# Flatten into a lookup: synonym phrase (lowercase) -> canonical skill name
SYNONYM_TO_CANONICAL = {}
for canonical, synonyms in SKILL_SYNONYMS.items():
    for syn in synonyms:
        SYNONYM_TO_CANONICAL[syn.strip().lower()] = canonical
    # also allow matching on the canonical name itself
    SYNONYM_TO_CANONICAL[canonical.strip().lower()] = canonical

# Skills that are "close cousins" of each other — used to decide whether a
# semantic/partial match should be flagged as "related" rather than silently
# treated as a full match. Add pairs both ways.
RELATED_SKILLS = {
    "FastAPI": ["Flask", "Django", "Node.js"],
    "Flask": ["FastAPI", "Django"],
    "Django": ["FastAPI", "Flask"],

    "React": ["Vue.js", "Angular"],
    "Vue.js": ["React", "Angular"],
    "Angular": ["React", "Vue.js"],

    "AWS": ["Azure", "GCP"],
    "Azure": ["AWS", "GCP"],
    "GCP": ["AWS", "Azure"],

    "PostgreSQL": ["MySQL", "SQL"],
    "MySQL": ["PostgreSQL", "SQL"],

    "TensorFlow": ["PyTorch"],
    "PyTorch": ["TensorFlow"],

    "Git": ["GitHub", "GitLab", "Bitbucket"],
    "GitHub": ["Git"],
    "GitLab": ["Git"],
    "Bitbucket": ["Git"],

    "Docker": ["Kubernetes"],
    "Kubernetes": ["Docker"],

    "Express.js": ["Node.js"],
    "Node.js": ["Express.js"],
}

# Soft skills / process terms that should NEVER trigger the hard mandatory
# gate, even if they appear in a JD's "must have" section. These are too
# unreliable to detect from resume text (someone can be a great
# communicator without ever writing the word "communication"), so treating
# a missing one as a score-capping dealbreaker produces false negatives.
# They still count toward the normal skill-match score — just not the gate.
MANDATORY_GATE_EXCLUDED = {
    "Communication",
    "Agile/Scrum",
    "Project Management",
    "Data Analysis",
}

EDUCATION_LEVELS = {
    "phd": 5, "doctorate": 5, "ph.d": 5,
    "master": 4, "masters": 4, "m.s": 4, "msc": 4, "mba": 4, "m.tech": 4,
    "bachelor": 3, "bachelors": 3, "b.s": 3, "bsc": 3, "b.tech": 3, "be ": 3, "undergraduate": 3,
    "diploma": 2, "associate": 2,
    "high school": 1, "secondary school": 1,
}
