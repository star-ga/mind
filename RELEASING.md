# Releasing

1. Ensure `main` is green on CI.
2. Update `README.md` examples if needed.
3. Create a tag: `git tag mind-vX.Y.Z && git push --tags`
4. Draft GitHub release (Release Drafter will prefill if configured).
5. Note feature compatibility with `mind-runtime` in the release notes.
