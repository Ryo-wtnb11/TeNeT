# Audit artifacts

Each file here is a census of the workspace at one commit. They are cited as
the current capability authority — by `AUDIT_REPORT.md` and by the roadmap
issues — so a reader must be able to find the current one and see how old it
is.

Two rules, both of which the earlier commit-suffixed filenames broke:

1. **The filename is stable; the commit lives in the file.** Every artifact
   carries an `Audited at: <full sha>` line directly under its title. Refreshing
   an audit is an edit to the same path, so citations do not rot and the
   history of the census is the file's git history.

2. **The `Audited at` line is the only authority statement.** Do not repeat the
   commit in the title. `operation-matrix-c612b4f.md` was previously titled for
   `c612b4f` while its body claimed `d612869`; that ambiguity is what this
   convention removes.

Refreshing an artifact means: rerun the census against the new commit, update
the `Audited at` line, and state in the commit message what changed
classification. A stale artifact is not deleted — it is refreshed or explicitly
demoted in `AUDIT_REPORT.md`.

`public-api.jsonl` is generated from rustdoc JSON. It is reproducible from the
commit in the accompanying `artifact-classification.md`, not hand-edited.
