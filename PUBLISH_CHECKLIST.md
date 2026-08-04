# GitHub publishing checklist

## Before making the repository public

- [ ] Confirm that the internship employer permits public release.
- [ ] Confirm that the dataset and generated samples may be shown publicly.
- [ ] Confirm whether the model weights may be redistributed.
- [ ] Review the repository for internal names, paths, logs, and credentials.
- [ ] Choose a license only after ownership is clear.

## Recommended repository settings

- Suggested name: `dcgan-anime-face-lab`
- Short description: `Resource-constrained DCGAN experiments for 64x64 anime-face generation`
- Topics: `pytorch`, `dcgan`, `gan`, `generative-models`, `computer-vision`, `reproducible-research`
- Default branch: `main`
- First tag: `v0.1-snapshot`
- Keep the repository private until publication permission is confirmed.

## Local push after authentication

```powershell
gh auth login
cd "C:\Users\32875\OneDrive\Desktop\DCGAN_Interview_GitHub_Snapshot_2026-08-04"
git commit -m "Freeze initial DCGAN interview snapshot"
git branch -M main
gh repo create dcgan-anime-face-lab --private --source=. --remote=origin --push
git tag -a v0.1-snapshot -m "Initial interview snapshot"
git push origin v0.1-snapshot
```

Change `--private` to `--public` only after the IP and dataset review is complete. Never paste a personal access token into chat; authenticate locally with `gh auth login`.
