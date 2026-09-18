# Deploying the dashboard to Streamlit Community Cloud

Streamlit Community Cloud is free for public repositories. It runs the app
straight from GitHub, so every push updates the live site.

## Before you start

The deployed app cannot run the pipeline. It only has what is in the
repository, so these files must be committed:

- `streamlit_app.py`
- `requirements.txt`
- `src/`
- `config.yaml`
- `data/processed/analysis_table.csv`
- `data/manual/start_times.csv`
- `outputs/` (the charts and result tables)

The `.gitignore` in this repository is already set up that way. Check it before
you deploy:

```bash
git status
git ls-files data/processed outputs | head
```

If `data/processed/analysis_table.csv` does not appear, run
`python run_pipeline.py` first and commit the result.

## Steps

1. Go to <https://share.streamlit.io> in your browser.

2. Click **Sign in with GitHub** (top right). Streamlit will ask GitHub for
   permission to read your repositories. Click **Authorize streamlit**.

3. Once you are signed in, click the **Create app** button (top right).

4. On the "Deploy an app" screen, choose **Deploy a public app from GitHub**.

5. Fill in the three fields:
   - **Repository**: start typing `school-start-times` and pick
     `YOUR_USERNAME/school-start-times` from the dropdown.
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`

6. Click **Advanced settings** just below those fields. Set **Python version**
   to `3.12`. Leave the secrets box empty, since this app has no API keys.
   Click **Save**.

7. Optionally, edit the **App URL** field to choose the address, for example
   `nevada-school-start-times`. The full address becomes
   `https://nevada-school-start-times.streamlit.app`.

8. Click **Deploy**.

9. A build log opens on the right. Installing the packages takes two to five
   minutes on the first deploy. When it finishes, the dashboard loads in the
   same tab.

10. Copy the URL from the address bar. That link is public and works on a
    phone.

## Updating the live app

Push to `main` and Streamlit redeploys within about a minute:

```bash
git add outputs data/processed data/manual
git commit -m "Update results with new start times"
git push
```

If the app does not pick up the change, open it on
<https://share.streamlit.io>, click the three dot menu beside the app, and
choose **Reboot app**.

## If the build fails

Open the build log and read the last twenty lines. The two common causes:

**A package fails to install.** Streamlit Cloud may not have a wheel for a
pinned version on its Python version. Loosen the pin in `requirements.txt`, for
example change `pandas==3.0.6` to `pandas>=2.2`, then commit and push.

**The app starts but says no analysis table was found.** The processed data was
not committed. Run the pipeline locally, then:

```bash
git add -f data/processed/analysis_table.csv
git commit -m "Add processed analysis table for the deployed dashboard"
git push
```

## Putting the app to sleep

An app with no visitors for several days is put to sleep automatically. Opening
its URL wakes it up, which takes about thirty seconds. Nothing is lost.
