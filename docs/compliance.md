# Compliance

- Data minimization: Only store necessary fields (email, password hash, role). Prediction summaries avoid PII beyond subject_name provided by user.
- Right to deletion: Implement endpoint (admin or self) to delete account; cascades predictions via FK.
- Data export: Provide endpoint to export user data (JSON dump of user + predictions).
- Auditability: All admin operations recorded in audit_logs.
- Privacy levels: Admin has elevated access; users access only own data.