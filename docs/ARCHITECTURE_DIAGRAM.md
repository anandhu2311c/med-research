# 🏗️ Smart Medical Literature Assistant - Architecture Diagrams

## System Architecture Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        WEB[Web Browser]
        MOBILE[Mobile App]
    end
    
    subgraph "Frontend Application"
        REACT[React + TypeScript]
        VITE[Vite Build Tool]
        TAILWIND[Tailwind CSS]
        QUERY[React Query]
    end
    
    subgraph "API Gateway Layer"
        FASTAPI[FastAPI Server]
        LANGSERVE[LangServe Router]
        CORS[CORS Middleware]
        AUTH[Auth Middleware]
        RATE[Rate Limiter]
    end
    
    subgraph "Workflow Engine"
        LANGGRAPH[LangGraph State Machine]
        PLANNER[Query Planner]
        ORCHESTRATOR[Node Orchestrator]
        STATE[State Manager]
    end
    
    subgraph "Processing Nodes"
        FETCH_ARXIV[arXiv Fetcher]
        FETCH_PUBMED[PubMed Fetcher]
        FETCH_CTGOV[ClinicalTrials Fetcher]
        NORMALIZE[Data Normalizer]
        DEDUPE[Deduplicator]
        EMBED[Embedding Generator]
        RETRIEVE[RAG Retriever]
        SYNTHESIZE[AI Synthesizer]
        PERSIST[Report Persister]
    end
    
    subgraph "AI/ML Services"
        GROQ[Groq LLM API]
        HF[HuggingFace Transformers]
        SENTENCE[Sentence Transformers]
    end
    
    subgraph "Data Storage"
        PINECONE[Pinecone Vector DB]
        FILES[File System]
        CACHE[Redis Cache]
    end
    
    subgraph "External APIs"
        ARXIV_API[arXiv API]
        PUBMED_API[PubMed API]
        CTGOV_API[ClinicalTrials API]
    end
    
    subgraph "Observability"
        LANGFUSE[Langfuse Tracing]
        PROMETHEUS[Prometheus Metrics]
        GRAFANA[Grafana Dashboard]
        LOGS[Centralized Logging]
    end
    
    %% User Flow
    WEB --> REACT
    MOBILE --> REACT
    REACT --> FASTAPI
    
    %% API Layer
    FASTAPI --> LANGSERVE
    LANGSERVE --> LANGGRAPH
    
    %% Workflow Execution
    LANGGRAPH --> PLANNER
    PLANNER --> FETCH_ARXIV
    PLANNER --> FETCH_PUBMED
    PLANNER --> FETCH_CTGOV
    
    FETCH_ARXIV --> NORMALIZE
    FETCH_PUBMED --> NORMALIZE
    FETCH_CTGOV --> NORMALIZE
    
    NORMALIZE --> DEDUPE
    DEDUPE --> EMBED
    EMBED --> RETRIEVE
    RETRIEVE --> SYNTHESIZE
    SYNTHESIZE --> PERSIST
    
    %% External Integrations
    FETCH_ARXIV --> ARXIV_API
    FETCH_PUBMED --> PUBMED_API
    FETCH_CTGOV --> CTGOV_API
    
    EMBED --> HF
    SYNTHESIZE --> GROQ
    
    %% Data Storage
    EMBED --> PINECONE
    RETRIEVE --> PINECONE
    PERSIST --> FILES
    
    %% Observability
    LANGGRAPH -.-> LANGFUSE
    FASTAPI -.-> PROMETHEUS
    PROMETHEUS -.-> GRAFANA
    
    %% Styling
    classDef frontend fill:#e1f5fe
    classDef api fill:#f3e5f5
    classDef workflow fill:#e8f5e8
    classDef processing fill:#fff3e0
    classDef ai fill:#fce4ec
    classDef storage fill:#f1f8e9
    classDef external fill:#fff8e1
    classDef observability fill:#f3e5f5
    
    class WEB,MOBILE,REACT,VITE,TAILWIND,QUERY frontend
    class FASTAPI,LANGSERVE,CORS,AUTH,RATE api
    class LANGGRAPH,PLANNER,ORCHESTRATOR,STATE workflow
    class FETCH_ARXIV,FETCH_PUBMED,FETCH_CTGOV,NORMALIZE,DEDUPE,EMBED,RETRIEVE,SYNTHESIZE,PERSIST processing
    class GROQ,HF,SENTENCE ai
    class PINECONE,FILES,CACHE storage
    class ARXIV_API,PUBMED_API,CTGOV_API external
    class LANGFUSE,PROMETHEUS,GRAFANA,LOGS observability
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant LangGraph
    participant Fetchers
    participant AI_Services
    participant Storage
    participant External_APIs
    
    User->>Frontend: Submit Query
    Frontend->>API: POST /api/query/invoke
    API->>LangGraph: Initialize Workflow
    
    LangGraph->>LangGraph: Plan Execution
    
    par Parallel Fetching
        LangGraph->>Fetchers: Fetch arXiv Papers
        Fetchers->>External_APIs: Query arXiv API
        External_APIs-->>Fetchers: Return Papers
        
        LangGraph->>Fetchers: Fetch PubMed Papers
        Fetchers->>External_APIs: Query PubMed API
        External_APIs-->>Fetchers: Return Papers
        
        LangGraph->>Fetchers: Fetch ClinicalTrials
        Fetchers->>External_APIs: Query CT.gov API
        External_APIs-->>Fetchers: Return Trials
    end
    
    LangGraph->>LangGraph: Normalize & Deduplicate
    
    LangGraph->>AI_Services: Generate Embeddings
    AI_Services->>Storage: Store Vectors
    
    LangGraph->>Storage: Retrieve Similar Papers
    Storage-->>LangGraph: Return Relevant Papers
    
    LangGraph->>AI_Services: Synthesize Summary
    AI_Services-->>LangGraph: Return Analysis
    
    LangGraph->>Storage: Persist Report
    
    LangGraph-->>API: Return Report ID
    API-->>Frontend: Stream Progress & Result
    Frontend-->>User: Display Report
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "Frontend Components"
        QF[QueryForm]
        PS[ProgressStream]
        RV[ReportViewer]
        EX[ExportTools]
    end
    
    subgraph "API Endpoints"
        QE[/api/query]
        RE[/api/report]
        HE[/health]
        FE[/feedback]
    end
    
    subgraph "LangGraph Nodes"
        PN[Planner Node]
        FN[Fetch Nodes]
        NN[Normalize Node]
        DN[Dedupe Node]
        EN[Embed Node]
        RN[Retrieve Node]
        SN[Synthesize Node]
        CN[Cite Check Node]
        PRN[Persist Node]
    end
    
    subgraph "Service Layer"
        AC[arXiv Client]
        PC[PubMed Client]
        CC[ClinicalTrials Client]
        ES[Embedding Service]
        LS[LLM Service]
        VS[Vector Service]
        RS[Report Service]
    end
    
    subgraph "Data Layer"
        VDB[(Vector Database)]
        FS[(File System)]
        CH[(Cache)]
    end
    
    %% Frontend to API
    QF --> QE
    PS --> QE
    RV --> RE
    EX --> RE
    
    %% API to LangGraph
    QE --> PN
    RE --> PRN
    
    %% LangGraph Flow
    PN --> FN
    FN --> NN
    NN --> DN
    DN --> EN
    EN --> RN
    RN --> SN
    SN --> CN
    CN --> PRN
    
    %% Nodes to Services
    FN --> AC
    FN --> PC
    FN --> CC
    EN --> ES
    RN --> VS
    SN --> LS
    PRN --> RS
    
    %% Services to Data
    ES --> VDB
    VS --> VDB
    RS --> FS
    AC --> CH
    PC --> CH
    CC --> CH
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Application Load Balancer]
        SSL[SSL Termination]
    end
    
    subgraph "Frontend Tier"
        CDN[CloudFront CDN]
        S3[S3 Static Hosting]
    end
    
    subgraph "Application Tier"
        subgraph "Auto Scaling Group"
            API1[FastAPI Instance 1]
            API2[FastAPI Instance 2]
            API3[FastAPI Instance 3]
        end
    end
    
    subgraph "Processing Tier"
        subgraph "Container Orchestration"
            K8S[Kubernetes Cluster]
            WORKER1[Worker Pod 1]
            WORKER2[Worker Pod 2]
            WORKER3[Worker Pod 3]
        end
    end
    
    subgraph "Data Tier"
        PINECONE_PROD[Pinecone Production]
        RDS[RDS PostgreSQL]
        ELASTICACHE[ElastiCache Redis]
        EFS[EFS Shared Storage]
    end
    
    subgraph "External Services"
        GROQ_API[Groq API]
        ARXIV_EXT[arXiv API]
        PUBMED_EXT[PubMed API]
        LANGFUSE_CLOUD[Langfuse Cloud]
    end
    
    subgraph "Monitoring"
        CLOUDWATCH[CloudWatch]
        PROMETHEUS_PROD[Prometheus]
        GRAFANA_PROD[Grafana]
        ALERTMANAGER[AlertManager]
    end
    
    %% User Traffic Flow
    Users --> CDN
    CDN --> S3
    Users --> LB
    LB --> API1
    LB --> API2
    LB --> API3
    
    %% API to Processing
    API1 --> K8S
    API2 --> K8S
    API3 --> K8S
    
    K8S --> WORKER1
    K8S --> WORKER2
    K8S --> WORKER3
    
    %% Data Connections
    API1 --> PINECONE_PROD
    API2 --> PINECONE_PROD
    API3 --> PINECONE_PROD
    
    WORKER1 --> RDS
    WORKER2 --> RDS
    WORKER3 --> RDS
    
    API1 --> ELASTICACHE
    API2 --> ELASTICACHE
    API3 --> ELASTICACHE
    
    WORKER1 --> EFS
    WORKER2 --> EFS
    WORKER3 --> EFS
    
    %% External API Calls
    WORKER1 --> GROQ_API
    WORKER2 --> GROQ_API
    WORKER3 --> GROQ_API
    
    WORKER1 --> ARXIV_EXT
    WORKER2 --> PUBMED_EXT
    WORKER3 --> LANGFUSE_CLOUD
    
    %% Monitoring Connections
    API1 -.-> CLOUDWATCH
    API2 -.-> CLOUDWATCH
    API3 -.-> CLOUDWATCH
    
    K8S -.-> PROMETHEUS_PROD
    PROMETHEUS_PROD -.-> GRAFANA_PROD
    PROMETHEUS_PROD -.-> ALERTMANAGER
```

## Security Architecture

```mermaid
graph TB
    subgraph "Security Perimeter"
        WAF[Web Application Firewall]
        DDOS[DDoS Protection]
    end
    
    subgraph "Authentication & Authorization"
        AUTH0[Auth0 / Cognito]
        JWT[JWT Tokens]
        RBAC[Role-Based Access Control]
    end
    
    subgraph "API Security"
        RATE_LIMIT[Rate Limiting]
        API_KEY[API Key Management]
        CORS_SEC[CORS Policy]
        INPUT_VAL[Input Validation]
    end
    
    subgraph "Data Security"
        ENCRYPT_TRANSIT[Encryption in Transit]
        ENCRYPT_REST[Encryption at Rest]
        KEY_MGMT[Key Management Service]
        DATA_MASK[Data Masking]
    end
    
    subgraph "Network Security"
        VPC[Virtual Private Cloud]
        SUBNETS[Private Subnets]
        NAT[NAT Gateway]
        SG[Security Groups]
    end
    
    subgraph "Compliance & Auditing"
        AUDIT_LOG[Audit Logging]
        COMPLIANCE[HIPAA/GDPR Compliance]
        BACKUP[Encrypted Backups]
        RETENTION[Data Retention Policy]
    end
    
    %% Security Flow
    Internet --> WAF
    WAF --> DDOS
    DDOS --> AUTH0
    AUTH0 --> JWT
    JWT --> RBAC
    
    RBAC --> RATE_LIMIT
    RATE_LIMIT --> API_KEY
    API_KEY --> CORS_SEC
    CORS_SEC --> INPUT_VAL
    
    INPUT_VAL --> ENCRYPT_TRANSIT
    ENCRYPT_TRANSIT --> VPC
    VPC --> SUBNETS
    SUBNETS --> SG
    
    SG --> ENCRYPT_REST
    ENCRYPT_REST --> KEY_MGMT
    KEY_MGMT --> DATA_MASK
    
    DATA_MASK --> AUDIT_LOG
    AUDIT_LOG --> COMPLIANCE
    COMPLIANCE --> BACKUP
    BACKUP --> RETENTION
```

## Monitoring & Observability Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        APP[Application Code]
        MIDDLEWARE[Monitoring Middleware]
    end
    
    subgraph "Metrics Collection"
        PROMETHEUS[Prometheus Server]
        NODE_EXPORTER[Node Exporter]
        APP_METRICS[Application Metrics]
        CUSTOM_METRICS[Custom Metrics]
    end
    
    subgraph "Logging Pipeline"
        FLUENTD[Fluentd/Fluent Bit]
        ELASTICSEARCH[Elasticsearch]
        KIBANA[Kibana Dashboard]
    end
    
    subgraph "Tracing System"
        LANGFUSE_TRACE[Langfuse Tracing]
        JAEGER[Jaeger (Optional)]
        TRACE_COLLECTOR[Trace Collector]
    end
    
    subgraph "Alerting System"
        ALERTMANAGER[Alert Manager]
        SLACK[Slack Notifications]
        EMAIL[Email Alerts]
        PAGERDUTY[PagerDuty]
    end
    
    subgraph "Visualization"
        GRAFANA_VIZ[Grafana Dashboards]
        LANGFUSE_UI[Langfuse UI]
        CUSTOM_DASH[Custom Dashboards]
    end
    
    subgraph "Health Checks"
        HEALTH_ENDPOINT[/health Endpoint]
        READINESS[Readiness Probes]
        LIVENESS[Liveness Probes]
    end
    
    %% Data Flow
    APP --> MIDDLEWARE
    MIDDLEWARE --> APP_METRICS
    APP_METRICS --> PROMETHEUS
    
    APP --> FLUENTD
    FLUENTD --> ELASTICSEARCH
    ELASTICSEARCH --> KIBANA
    
    APP --> LANGFUSE_TRACE
    LANGFUSE_TRACE --> TRACE_COLLECTOR
    
    PROMETHEUS --> ALERTMANAGER
    ALERTMANAGER --> SLACK
    ALERTMANAGER --> EMAIL
    ALERTMANAGER --> PAGERDUTY
    
    PROMETHEUS --> GRAFANA_VIZ
    LANGFUSE_TRACE --> LANGFUSE_UI
    
    APP --> HEALTH_ENDPOINT
    HEALTH_ENDPOINT --> READINESS
    HEALTH_ENDPOINT --> LIVENESS
```

## Data Processing Pipeline

```mermaid
graph LR
    subgraph "Input Stage"
        QUERY[User Query]
        FILTERS[Search Filters]
    end
    
    subgraph "Planning Stage"
        PARSE[Query Parser]
        STRATEGY[Search Strategy]
        PARALLEL[Parallel Planner]
    end
    
    subgraph "Collection Stage"
        ARXIV_FETCH[arXiv Fetcher]
        PUBMED_FETCH[PubMed Fetcher]
        CTGOV_FETCH[CT.gov Fetcher]
    end
    
    subgraph "Processing Stage"
        NORMALIZE[Data Normalizer]
        DEDUPE[Deduplicator]
        RANK[Ranking Algorithm]
    end
    
    subgraph "Embedding Stage"
        CHUNK[Text Chunker]
        EMBED_GEN[Embedding Generator]
        VECTOR_STORE[Vector Storage]
    end
    
    subgraph "Retrieval Stage"
        QUERY_EMBED[Query Embedding]
        SIMILARITY[Similarity Search]
        RERANK[Re-ranking]
    end
    
    subgraph "Analysis Stage"
        CONTEXT[Context Assembly]
        LLM_CALL[LLM Generation]
        POST_PROCESS[Post Processing]
    end
    
    subgraph "Output Stage"
        CITE_CHECK[Citation Check]
        FORMAT[Report Formatting]
        EXPORT[Export Generation]
    end
    
    %% Pipeline Flow
    QUERY --> PARSE
    FILTERS --> PARSE
    PARSE --> STRATEGY
    STRATEGY --> PARALLEL
    
    PARALLEL --> ARXIV_FETCH
    PARALLEL --> PUBMED_FETCH
    PARALLEL --> CTGOV_FETCH
    
    ARXIV_FETCH --> NORMALIZE
    PUBMED_FETCH --> NORMALIZE
    CTGOV_FETCH --> NORMALIZE
    
    NORMALIZE --> DEDUPE
    DEDUPE --> RANK
    
    RANK --> CHUNK
    CHUNK --> EMBED_GEN
    EMBED_GEN --> VECTOR_STORE
    
    QUERY --> QUERY_EMBED
    QUERY_EMBED --> SIMILARITY
    VECTOR_STORE --> SIMILARITY
    SIMILARITY --> RERANK
    
    RERANK --> CONTEXT
    CONTEXT --> LLM_CALL
    LLM_CALL --> POST_PROCESS
    
    POST_PROCESS --> CITE_CHECK
    CITE_CHECK --> FORMAT
    FORMAT --> EXPORT
```

## Technology Stack Diagram

```mermaid
graph TB
    subgraph "Frontend Stack"
        REACT_TECH[React 18]
        TS[TypeScript]
        VITE_TECH[Vite]
        TAILWIND_TECH[Tailwind CSS]
        REACT_QUERY[React Query]
    end
    
    subgraph "Backend Stack"
        FASTAPI_TECH[FastAPI]
        PYTHON[Python 3.11+]
        PYDANTIC[Pydantic]
        ASYNCIO[AsyncIO]
    end
    
    subgraph "AI/ML Stack"
        LANGCHAIN_TECH[LangChain]
        LANGGRAPH_TECH[LangGraph]
        LANGSERVE_TECH[LangServe]
        GROQ_TECH[Groq API]
        HUGGINGFACE[HuggingFace]
        SENTENCE_TRANS[Sentence Transformers]
    end
    
    subgraph "Data Stack"
        PINECONE_TECH[Pinecone]
        REDIS_TECH[Redis]
        POSTGRESQL[PostgreSQL]
        S3_TECH[AWS S3]
    end
    
    subgraph "DevOps Stack"
        DOCKER_TECH[Docker]
        KUBERNETES_TECH[Kubernetes]
        GITHUB_ACTIONS[GitHub Actions]
        TERRAFORM_TECH[Terraform]
    end
    
    subgraph "Monitoring Stack"
        LANGFUSE_TECH[Langfuse]
        PROMETHEUS_TECH[Prometheus]
        GRAFANA_TECH[Grafana]
        ELASTICSEARCH_TECH[Elasticsearch]
    end
    
    %% Technology Relationships
    REACT_TECH --> FASTAPI_TECH
    FASTAPI_TECH --> LANGCHAIN_TECH
    LANGCHAIN_TECH --> GROQ_TECH
    LANGCHAIN_TECH --> HUGGINGFACE
    
    FASTAPI_TECH --> PINECONE_TECH
    FASTAPI_TECH --> REDIS_TECH
    FASTAPI_TECH --> POSTGRESQL
    
    DOCKER_TECH --> KUBERNETES_TECH
    GITHUB_ACTIONS --> DOCKER_TECH
    TERRAFORM_TECH --> KUBERNETES_TECH
    
    LANGFUSE_TECH --> PROMETHEUS_TECH
    PROMETHEUS_TECH --> GRAFANA_TECH
```

## API Architecture

```mermaid
graph TB
    subgraph "API Gateway"
        GATEWAY[FastAPI Gateway]
        MIDDLEWARE_STACK[Middleware Stack]
        ROUTING[Request Routing]
    end
    
    subgraph "Core APIs"
        QUERY_API[Query API]
        REPORT_API[Report API]
        HEALTH_API[Health API]
        FEEDBACK_API[Feedback API]
    end
    
    subgraph "LangServe Integration"
        LANGSERVE_ROUTER[LangServe Router]
        WORKFLOW_ENDPOINT[Workflow Endpoint]
        STREAMING[Streaming Support]
        PLAYGROUND[Interactive Playground]
    end
    
    subgraph "Authentication APIs"
        AUTH_API[Authentication API]
        TOKEN_API[Token Management]
        USER_API[User Management]
    end
    
    subgraph "Admin APIs"
        METRICS_API[Metrics API]
        CONFIG_API[Configuration API]
        LOGS_API[Logs API]
    end
    
    subgraph "External Integrations"
        WEBHOOK_API[Webhook API]
        EXPORT_API[Export API]
        INTEGRATION_API[Integration API]
    end
    
    %% API Flow
    GATEWAY --> MIDDLEWARE_STACK
    MIDDLEWARE_STACK --> ROUTING
    
    ROUTING --> QUERY_API
    ROUTING --> REPORT_API
    ROUTING --> HEALTH_API
    ROUTING --> FEEDBACK_API
    
    QUERY_API --> LANGSERVE_ROUTER
    LANGSERVE_ROUTER --> WORKFLOW_ENDPOINT
    WORKFLOW_ENDPOINT --> STREAMING
    LANGSERVE_ROUTER --> PLAYGROUND
    
    ROUTING --> AUTH_API
    AUTH_API --> TOKEN_API
    AUTH_API --> USER_API
    
    ROUTING --> METRICS_API
    ROUTING --> CONFIG_API
    ROUTING --> LOGS_API
    
    ROUTING --> WEBHOOK_API
    ROUTING --> EXPORT_API
    ROUTING --> INTEGRATION_API
```

---

## Architecture Decision Records (ADRs)

### ADR-001: Choice of LangGraph for Workflow Orchestration

**Status**: Accepted

**Context**: Need for robust workflow orchestration with state management, error handling, and observability.

**Decision**: Use LangGraph as the primary workflow engine.

**Consequences**:
- ✅ Built-in state management
- ✅ Native LangChain integration
- ✅ Excellent observability
- ❌ Learning curve for team
- ❌ Relatively new technology

### ADR-002: Pinecone for Vector Database

**Status**: Accepted

**Context**: Need for high-performance vector similarity search with minimal operational overhead.

**Decision**: Use Pinecone as the managed vector database solution.

**Consequences**:
- ✅ Managed service (no ops overhead)
- ✅ Excellent performance
- ✅ Good SDK and documentation
- ❌ Vendor lock-in
- ❌ Cost at scale

### ADR-003: React + TypeScript for Frontend

**Status**: Accepted

**Context**: Need for modern, maintainable frontend with good developer experience.

**Decision**: Use React 18 with TypeScript and Vite.

**Consequences**:
- ✅ Strong ecosystem
- ✅ Type safety
- ✅ Fast development
- ✅ Good performance
- ❌ Bundle size considerations

---

This architecture documentation provides a comprehensive view of the Smart Medical Literature Assistant system design, covering all major components, their interactions, and the rationale behind key technology choices.