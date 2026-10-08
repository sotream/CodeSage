# Security policy

This is a personal project, maintained on a best-effort basis with no support commitment.

## Reporting a vulnerability

Do not open a public issue. Use GitHub's private reporting: **Security** tab, then **Report a
vulnerability**. Include the affected file, steps to reproduce and the impact.

Reports are read when time allows; there is no response-time guarantee. Fixes land on `main`, and only the
latest commit on `main` is supported.

## Scope

In scope: the code in this repository. Out of scope: vulnerabilities in third-party dependencies (report
them upstream).

Note for users: `codesage ask` sends retrieved code excerpts to the Anthropic API. Do not index code you
are not allowed to share with that service. The index file (`.codesage/index.json`) contains your source
code in plain text; keep it out of version control.
