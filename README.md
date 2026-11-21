# LiftoDroid apps for Home Assistant

| App | What it does |
|---|---|
| [LiftoDroid Barbell Camera](liftodroid_barbell) | Reads the weight on your barbell from an overhead gym camera, for LiftoDroid's set logging |

## Install from GitHub

In Home Assistant: **Settings → Apps → App store → ⋮ (top right) → Repositories**, paste the repository URL, **Add**,
then close the dialog. "LiftoDroid Barbell Camera" appears in the store; open it and **Install**. Home Assistant
builds the app on your machine the first time, which takes a few minutes.

Updates show up like any other app's when the version in `liftodroid_barbell/config.yaml` goes up.

### Which URL

* **The public mirror** (recommended): the repository this folder is published to by
  `.github/workflows/publish-ha-app.yml`, e.g. `https://github.com/Flexingg/liftodroid-ha-apps`. It holds only this
  folder, so Home Assistant can clone it without credentials.
* **This repository directly** works only if it's public: Home Assistant clones app repositories anonymously, and
  LiftoDroid is private. (A token in the URL, `https://<token>@github.com/…`, may be accepted, but Home Assistant
  writes the full URL to its logs on every clone and update, so the token leaks there. Not recommended.)

### Setting up the public mirror (once)

1. On GitHub, create an empty **public** repository, e.g. `Flexingg/liftodroid-ha-apps`.
2. Create a fine-grained personal access token with **Contents: Read and write** on that repository only.
3. In this repository: **Settings → Secrets and variables → Actions**:
   * secret `HA_APPS_TOKEN` = the token;
   * variable `HA_APPS_REPO` = `Flexingg/liftodroid-ha-apps` (that's the default if you leave it out).
4. Run **Actions → Publish Home Assistant app → Run workflow** (it also runs on every push to `main` that touches
   `homeassistant/`).

Only `homeassistant/` is published: no app code, signing keys or other files from this repository.

## Develop

```bash
cd liftodroid_barbell
pip install -r requirements-dev.txt
python -m pytest -q tests
```
