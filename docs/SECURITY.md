# Security

Do not commit credentials, private conversations, bulk data, caches, checkpoints, expiring signed URLs, or unreviewed third-party material. Load untrusted metadata as validated JSON only; deployment does not deserialize executable objects. Pull-request workflows must verify the exact requested candidate SHA and must not run untrusted code on a privileged training workstation.
