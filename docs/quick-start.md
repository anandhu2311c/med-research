# Quick Start Guide

Get your Medical Literature Assistant running in 10 minutes!

## Step 1: Get API Keys (5 minutes)

### Groq (LLM) - FREE
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with email
3. Click "API Keys" → "Create API Key"
4. Copy key (starts with `gsk_`)

### Pinecone (Vector DB) - FREE
1. Go to [pinecone.io](https://pinecone.io) → "Start for Free"
2. Sign up with email
3. Create Index:
   - Name: `medlit-embeddings`
   - Dimensions: `384`
   - Metric: `cosine`
4. Go to "API Keys" → Copy your key
5. Note your environment (e.g., `us-east-1-aws`)

### Langfuse (Observability) - FREE
1. Go to [cloud.langfuse.com](https://cloud.langfuse.com)
2. Sign up with email
3. Create project: "Medical Literature Assistant"
4. Go to "API Keys" → Copy public & secret keys

## Step 2: Setup Project (3 minutes)

```bash
# Clone or download the project
# Navigate to project directory

# Copy environment template
copy .env.example .env

# Edit .env with your API keys
notepad .env
```

Your `.env` should look like:
```env
GROQ_API_KEY=gsk_your_actual_key_here
PINECONE_API_KEY=your_actual_pinecone_key
PINECONE_ENV=us-east-1-aws
LANGFUSE_PUBLIC_KEY=pk-lf-your_actual_public_key
LANGFUSE_SECRET_KEY=sk-lf-your_actual_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Step 3: Install & Run (2 minutes)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..

# Verify setup
python scripts/verify_setup.py

# Start backend (terminal 1)
uvicorn app.main:app --reload

# Start frontend (terminal 2)
cd frontend && npm run dev
```

## Step 4: Use the App

1. Open http://localhost:5173
2. Enter a research query like "AI in medical imaging"
3. Select sources (arXiv, PubMed)
4. Click "Search Literature"
5. Watch real-time progress
6. View generated report with citations

## Troubleshooting

**"Module not found" errors**:
```bash
pip install -r requirements.txt
```

**Pinecone connection fails**:
- Check your index name is exactly `medlit-embeddings`
- Verify dimensions are `384`
- Confirm your environment matches your region

**Frontend won't start**:
```bash
cd frontend
npm install
npm run dev
```

**API key errors**:
- Double-check keys are copied correctly
- Ensure no extra spaces in .env file
- Verify keys are active in respective consoles

## What's Next?

- Try different research queries
- Explore the generated reports
- Check observability in Langfuse dashboard
- Customize filters and sources

Need help? Check the full documentation in `docs/api-setup.md`