# Verification

Verified locally on Windows with Python 3.14 on 2026-11-08:

- 39 pytest tests pass in both the existing research environment and a fresh environment installed from a Git archive of the maintained project.
- Ruff lint and formatting checks pass.
- The full offline pipeline runs from both the original checkout and the archive: 3,876 historical driver-race rows, expanding season evaluations, three trained models and a 22-driver forecast.
- The fresh-checkout predictions match the bundled dashboard predictions exactly in the tested environment.
- Streamlit interaction tests cover initial loading, exploratory weights, diagnostic opt-in, a custom three-driver field and missing-report recovery.
- The running Streamlit server returned HTTP 200 with `ok` from its health endpoint.
- No model binaries, local environments, caches or credentials are tracked by Git. A credential-pattern scan of source/config/documentation found no matches.

## Regression coverage

Eight regression cases cover three additional input and dashboard edge cases:

1. A chart slider crashed for a custom three-driver field.
2. Historical input validation accepted fractional/negative positions and negative race points.
3. Duplicate race dates under distinct keys and season/date-year disagreement could violate date-only chronology.

No same-date leakage was observed in the bundled valid data. The validation now rejects inconsistent calendars before feature construction.

## Project configuration

- **Python package plus Streamlit:** a command-line pipeline and interactive dashboard share the existing Python stack.
- **Research compatibility:** the original research scripts remain intact. The maintained project is on `main`.
- **Editable checkout installation:** documented commands use `pip install -e .`; config files, history and demo reports live in the repository. A standalone wheel does not bundle these inputs.
- **Canonical text hashes:** normalize CRLF to LF before SHA-256 hashing so Windows/Linux Git checkout conversion does not invalidate provenance.
- **Unverified inputs and predictive limits:** the inherited history/grid do not support claims of official validation. The README/dashboard disclose this. The small retrospective advantage over the pole baseline is not a superiority or calibration claim.

## Remaining limits

The tested pandas/NumPy combination emits four upstream deprecation warnings during target-frame concatenation. Streamlit also logs deprecation notices for `use_container_width`; this was retained for compatibility with the declared minimum Streamlit version. These warnings do not fail the tests.

Live API availability, current FIA documents, predictive superiority and calibration were not verified. Exact numerical results across different dependency versions/platforms are not guaranteed. GitHub Actions checks the supported Linux/Windows installation paths independently; its remote run status is available in the repository's Actions tab.
