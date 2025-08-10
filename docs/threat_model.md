# Threat Modeling (STRIDE)

- Spoofing: Mitigated by JWT auth with strong secrets, HTTPS (assumed by reverse proxy), rate limiting
- Tampering: Database user permissions (app user), SQLAlchemy ORM to avoid injection, input validation via Pydantic
- Repudiation: Audit logs table captures user actions
- Information Disclosure: Minimal data returned, RBAC, CORS restrictions
- Denial of Service: Redis-backed rate limiting and worker isolation
- Elevation of Privilege: Role checks in endpoints, default role=user

Security Hardening:
- Rotate JWT secret; set short token lifetimes
- Enable TLS termination in ingress/proxy
- Regular dependency scans
- Secrets via vault in production