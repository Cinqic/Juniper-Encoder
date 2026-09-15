# Security

> Historical document. Juniper Encoder was retired on 2026-09-15; see [RETIRED.md](../RETIRED.md) for the canonical final status. The text below describes the project as it stood during development.

Do not commit credentials, private conversations, bulk data, caches, checkpoints, expiring signed URLs, or unreviewed third-party material. Load untrusted metadata as validated JSON only; deployment does not deserialize executable objects. Pull-request workflows must verify the exact requested candidate SHA and must not run untrusted code on a privileged training workstation.
