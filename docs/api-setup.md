# API Setup Guide

This guide walks you through getting free API keys for all required services.

## 1. Groq API (LLM)

**Free Tier**: 14,400 requests/day with Llama models

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with email or GitHub
3. Navigate to "API Keys" in the left sidebar
4. Click "Create API Key"
5. Copy the key (starts with `gsk_...`)

**Add to .env**:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

## 2. Pinecone (Vector Database)

**Free Tier**: 1 index, 100K vectors, 2GB storage

### Step-by-Step Setup:

1. **Sign Up**:
   - Go to [pinecone.io](https://www.pinecone.io)
   - Click "Start for Free"
   - Sign up with email

2. **Create Index**:
   - In the Pinecone console, click "Create Index"
   - **Index Name**: `medlit-embeddings`
   - **Dimensions**: `384` (for sentence-transformers model)
   - **Metric**: `cosine`
   - **Cloud**: Choose your preferred region
   - Click "Create Index"

3. **Get API Key**:
   - Go to "API Keys" in left sidebar
   - Copy your API key
   - Note your environment (e.g., `us-east-1-aws`)

**Add to .env**:
```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENV=us-east-1-aws
```

## 3. Langfuse (Observability)

**Free Tier**: 50K observations/month

### Option A: Cloud (Recommended)

1. **Sign Up**:
   - Go to [cloud.langfuse.com](https://cloud.langfuse.com)
   - Sign up with email or GitHub

2. **Create Project**:
   - Click "New Project"
   - Name: "Medical Literature Assistant"
   - Click "Create"

3. **Get Keys**:
   - In project settings, go to "API Keys"
   - Copy the **Public Key** and **Secret Key**

**Add to .env**:
```env
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Option B: Self-Host (Advanced)

If you prefer self-hosting:

1. **Deploy on Railway**:
   - Fork [langfuse/langfuse](https://github.com/langfuse/langfuse)
   - Deploy to Railway with PostgreSQL addon
   - Set environment variables as per Langfuse docs

2. **Use Docker**:
   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker-compose up -d
   ```

**Add to .env**:
```env
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here
LANGFUSE_HOST=http://localhost:3000
```

## 4. Complete .env File

Your final `.env` should look like:

```env
# Groq LLM API
GROQ_API_KEY=gsk_your_groq_key_here

# Pinecone Vector Database
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENV=us-east-1-aws

# Langfuse Observability
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
```

## 5. Verify Setup

Run the verification script:

```bash
python scripts/verify_setup.py
```

This will test all API connections and confirm your setup is working.

## Troubleshooting

### Pinecone Issues
- **Index creation fails**: Check if you already have an index (free tier allows only 1)
- **Connection errors**: Verify your environment matches your index region

### Langfuse Issues
- **Authentication errors**: Double-check public/secret key pairs
- **Self-hosted connection**: Ensure your host URL is correct and accessible

### Groq Issues
- **Rate limits**: Free tier has daily limits, wait 24h if exceeded
- **Model access**: Ensure you're using supported models (llama-3.1-70b-versatile)

## Cost Monitoring

All services offer generous free tiers:
- **Groq**: Monitor usage in console dashboard
- **Pinecone**: Check vector count and storage in console
- **Langfuse**: View observation count in project settings

Set up alerts in each service to avoid unexpected charges.