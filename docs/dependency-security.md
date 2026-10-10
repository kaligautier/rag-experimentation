# Dependency audit

CI audits the installed locked dependencies, including development tools. Known
findings fail the job except for the following reviewed, narrowly scoped exception.

## NLTK model artifact paths — CVE-2026-81726

Reviewed 2026-10-10. NLTK 3.10.3 is a transitive dependency of LlamaIndex.
The [upstream advisory](https://github.com/nltk/nltk/security/advisories/GHSA-8mgp-746c-j5xp)
reports no patched version. Its affected APIs import/export NLP model artifacts
using caller-selected filesystem paths. This application uses the Markdown parser
and tiktoken tokenization; it neither calls those NLTK model persistence APIs nor
accepts model artifact paths from users. NLTK's path sandbox is not used as an
application security boundary.

CI excludes this one advisory (including its PYSEC/GHSA aliases), not the package
or other NLTK findings. Reassess before adding NLP model loading, updating
LlamaIndex/NLTK, or by 2026-11-10; remove the exception when an upstream fix exists.
