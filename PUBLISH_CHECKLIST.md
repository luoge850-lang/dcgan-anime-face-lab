# GitHub publishing checklist

## Public release status

The repository owner has authorized public visibility. The current public release intentionally excludes the dataset and model weights. Keep the checklist below for future changes and any open-source license decision.

## Before adding new public artifacts

- [x] Confirm that the project may be shown publicly.
- [x] Confirm that the current code, metrics, plots, manifests, and generated samples may be shown publicly.
- [ ] Confirm whether the model weights may be redistributed.
- [x] Review the repository for internal names, paths, logs, and credentials.
- [ ] Choose a license only after ownership is clear.

## Recommended repository settings

- Suggested name: `dcgan-anime-face-lab`
- Short description: `Resource-constrained DCGAN experiments for 64x64 anime-face generation`
- Topics: `pytorch`, `dcgan`, `gan`, `generative-models`, `computer-vision`, `reproducible-research`
- Default branch: `main`
- First tag: `v0.1-snapshot`
- The owner has confirmed that this snapshot may be public; keep datasets, weights, credentials, and unapproved internal material private.

## Local push after authentication

```powershell
gh auth login
cd "$SNAPSHOT_DIR"
gh auth status
git switch main
git pull --ff-only origin main
git push origin main
```

The repository already exists and has been switched to public visibility. Never paste a personal access token into chat; authenticate locally with `gh auth login`. Future updates should use a feature branch and pull request rather than rerunning repository creation.
