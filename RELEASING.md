# Release checklist (as of v0.9.0)
- [x] CI green on all repos
- [x] Tags pushed for mind, mind-runtime, mind-spec
- [x] Docs published via GitHub Pages
- [x] License and advisory checks clean
- [x] Main branches protected and signed

# Releasing

1. Ensure `main` is green on CI.
2. Update `README.md` examples if needed.
3. Create a tag: `git tag mind-vX.Y.Z && git push --tags`
4. Draft GitHub release (Release Drafter will prefill if configured).
5. Note feature compatibility with `mind-runtime` in the release notes.
