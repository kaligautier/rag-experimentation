# Dependency audit

CI audits the installed locked dependencies, including development tools. Any
known finding fails the job; no advisory is currently excluded.

To add an exception, record the reviewed advisory here (affected API, why the
application is not exposed, review date and reassessment deadline) and pass only
that advisory ID to `pip-audit --ignore-vuln` in `.github/workflows/ci.yml`.

The NLTK exception for CVE-2026-81726 was removed on 2026-11-19: NLTK 3.10.4
fixes it.
