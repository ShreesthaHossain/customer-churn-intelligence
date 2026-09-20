# Deploy to Render (Free Tier)

Professional **Docker** deployment: **FastAPI API** + **Streamlit dashboard**, both on Render’s free plan. The model is trained inside **GitHub Actions** when the Docker image is built — Render only runs inference (512 MB RAM is enough).

---

## Architecture

```
GitHub push (main)
    → GitHub Actions builds Docker image + trains model
    → Pushes to ghcr.io/shreesthahossain/customer-churn-intelligence:latest
    → Render pulls image for two free web services:
         churn-api        (FastAPI, /health, /docs)
         churn-streamlit  (Streamlit UI, links to API)
```

**Free-tier notes:** Services sleep after ~15 minutes of inactivity. First request after sleep may take 30–60 seconds (cold start).

---

## One-time setup (~10 minutes)

### Step 1 — Push this repo to GitHub

Ensure `main` is on GitHub. The **Publish Docker Image** workflow runs automatically and publishes the public container image.

Check: **GitHub → Actions → Publish Docker Image → green checkmark**

Verify image: `https://github.com/ShreesthaHossain/customer-churn-intelligence/pkgs/container/customer-churn-intelligence`

### Step 2 — Create a Render account

1. Go to [render.com](https://render.com) and sign up (GitHub login is easiest).
2. Dashboard → **New** → **Blueprint**.
3. Connect repository `ShreesthaHossain/customer-churn-intelligence`.
4. Render reads `render.yaml` and proposes two services:
   - `churn-api`
   - `churn-streamlit`
5. Click **Apply**.

Wait for both deploys to finish (first pull can take a few minutes).

### Step 3 — Copy your live URLs

| Service | URL pattern | Use |
|---------|-------------|-----|
| **API** | `https://churn-api-xxxx.onrender.com` | `/docs`, `/health`, integrations |
| **Streamlit** | `https://churn-streamlit-xxxx.onrender.com` | Interactive demo |

Render auto-generates `CHURN_API_KEY` for the API. Streamlit receives `CHURN_API_BASE_URL` from the API service automatically.

### Step 4 — Smoke test

```bash
# Health (no auth)
curl https://YOUR-API-URL.onrender.com/health

# Predict (use key from Render → churn-api → Environment)
curl -X POST https://YOUR-API-URL.onrender.com/predict_churn \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":75.0,"TotalCharges":900.0}'
```

Open the Streamlit URL and score one customer.

### Step 5 — Add links to README (optional polish)

Add your live URLs near the top of `README.md` so recruiters can click through immediately.

---

## Updating after code changes

1. Push to `main` → GitHub Actions rebuilds and pushes `:latest`.
2. Render Dashboard → each service → **Manual Deploy** → **Deploy latest image**.

(Prebuilt-image services do not auto-redeploy on git push.)

---

## Local Docker (same image behavior)

```bash
docker compose up --build
```

- API: http://localhost:8000  
- Streamlit: http://localhost:8501  

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Render deploy fails “image not found” | Wait for GitHub Actions to finish; confirm package is **public** on GHCR |
| API returns 401 | Send `X-API-Key` header; copy key from Render env vars |
| Streamlit sidebar API links wrong | Confirm `CHURN_API_BASE_URL` on `churn-streamlit` points to API `RENDER_EXTERNAL_URL` |
| Slow first load | Free tier cold start — normal for portfolio demos |
| Streamlit OOM | Rare on 512 MB; redeploy; if persistent, upgrade API to free and Streamlit to Starter |
