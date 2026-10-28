# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get:

- An **ATS score out of 100** with a breakdown (keywords, formatting, content, structure, readability)
- **Prioritized improvements** (High / Medium / Low) with exact fixes
- **Missing keywords**, optionally matched against a job description you paste
- **Bullet-point rewrites** and quick rule-based checks (email, phone, sections, word count)
- A downloadable text report

Built with **Streamlit** (UI) and **Groq** (AI).

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"      # Windows PowerShell: $env:GROQ_API_KEY="your_key_here"
streamlit run app.py
```

Get a free key at https://console.groq.com/keys. If you skip the environment variable, you can paste the key into the sidebar.

## Models

The default is `openai/gpt-oss-120b`. Groq retires models regularly (the Llama 3.3 70B model was shut down in August 2026), so if you see a "model rejected" error, pick another model in the sidebar or edit `MODEL_OPTIONS` at the top of `app.py`. Current list: https://console.groq.com/docs/models

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (app.py, requirements.txt, README.md).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Click **Deploy**.

Never commit your API key to GitHub. Keep it only in Secrets.

## Notes

- Scanned or image-only resumes cannot be read (ATS systems cannot read them either). Use a text-based PDF or DOCX.
- The score is an AI estimate based on common ATS rules, not the output of a real employer's ATS.
