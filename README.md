# NovaRetail Customer Intelligence Dashboard

An executive customer analytics dashboard built with **Streamlit**, **Plotly**, and **Google Gemini 2.5 Flash** for **Sophia Martinez, Director of Customer Intelligence** at NovaRetail.

The dashboard empowers decision-makers to:
1. **Analyze Revenue Drivers**: Identify which customer segments, regions, and product categories drive omnichannel sales.
2. **Detect Risk & Decline Early**: Monitor customer dissatisfaction and decline indicators before churn occurs.
3. **Formulate Actionable Strategies**: Prioritize data-backed commercial interventions across retail channels.
4. **AI-Powered Customer Intelligence**: Query an embedded Gemini AI advisor grounded directly in the active dashboard view.

---

## 🚀 Running Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed. Clone the repository and navigate to the project directory:

```bash
git clone <your-repo-url>
cd novaretail-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key (Optional for AI Chatbot)
Create a `.streamlit/secrets.toml` file in the project root:

```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
```

Add your Gemini API key:
```toml
GEMINI_API_KEY = "your-actual-gemini-api-key"
```

> **Note:** If no API key is provided, the dashboard and all visualizations remain 100% functional; the AI Chatbot tab will display a friendly setup notification.

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## ☁️ Deploying on Streamlit Community Cloud

1. **Push to GitHub**:
   Ensure your code is committed to a GitHub repository (note that `.gitignore` prevents `.streamlit/secrets.toml` from being committed).
   ```bash
   git add .
   git commit -m "Deploy NovaRetail Customer Intelligence Dashboard"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
   - Click **New app**.
   - Select your repository, branch (`main`), and set **Main file path** to `app.py`.

3. **Configure Secrets in Streamlit Cloud**:
   - In your app settings on Streamlit Cloud, go to **Settings > Secrets**.
   - Paste your configuration:
     ```toml
     GEMINI_API_KEY = "your-actual-gemini-api-key"
     ```
   - Click **Save**. The app will automatically rebuild and deploy with live AI capabilities!

---

## 📊 Dataset & Architecture

- **Data Source**: Relative path `NR_dataset.xlsx` (`data` sheet).
- **Core Behavioral Segments**: *Promising*, *Growth*, *Stable*, and *Decline*.
- **Standardized Product Groups**: 35 raw categories categorized into 9 structured groups (*Electronics*, *Clothing & Fashion*, *Groceries*, *Home & Furniture*, *Sports & Outdoors*, *Health & Beauty*, *Books*, *Toys*, and *Other*).
- **Customer Counting**: Strictly computed using `CustomerID.nunique()` to handle repeat transactions accurately.
