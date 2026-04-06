# System Design Interview Framework

Use this as your mental checklist for every system design question. A 45-minute round should roughly follow these time splits.

---

## Step 1: Requirements Clarification (3–5 min)

**This is where most candidates lose points by skipping ahead.**

### Functional Requirements (What does it do?)
Ask: "What are the core features we need to support?"

Template questions:
- Who are the users? (end users, internal, B2B?)
- What are the primary actions? (create, read, update, delete?)
- What are the inputs and outputs?
- Do we need real-time behavior? (notifications, live updates)
- Any special features? (search, analytics, recommendations)

**Tip:** List 3–5 core features. If the interviewer says "design Twitter," don't design ALL of Twitter. Clarify: "Should I focus on the feed, posting, following, or search?"

### Non-Functional Requirements (How well does it do it?)
Ask: "What qualities matter most for this system?"

Always address these four:

| Property | Question to Ask | Common Answer |
|----------|----------------|---------------|
| **Scale** | How many users? How many requests/sec? | Helps decide DB, caching, sharding |
| **Latency** | What's the acceptable response time? | Read-heavy = cache, real-time = WebSockets |
| **Availability vs Consistency** | Can we tolerate stale data? | Most systems: AP > CP (except payments) |
| **Durability** | Can we afford to lose data? | Almost always no — need replication |

---

## Step 2: Back-of-Envelope Estimation (3–5 min)

**Interviewers want to see you can think in numbers, not just boxes.**

### Traffic Estimation
```
Daily Active Users (DAU): ___
Writes per day:           DAU × actions_per_user = ___
Writes per second:        writes_per_day / 86400 = ___
Read:Write ratio:         ___ : 1 (usually 5:1 to 100:1)
Reads per second:         writes_per_second × ratio = ___
```

### Storage Estimation
```
Size per record:          ___ bytes
New records per day:      ___
Storage per year:         records_per_day × 365 × size = ___
Storage for 5 years:      ___
```

### Bandwidth
```
Incoming (write):         writes_per_sec × record_size = ___ KB/s
Outgoing (read):          reads_per_sec × response_size = ___ KB/s
```

### Memory (Cache)
```
If we cache top 20% of daily reads:
  daily_reads × 0.2 × response_size = ___ GB
```

**Tip:** Round aggressively. 86400 ≈ 100,000. The goal is order of magnitude, not precision.

---

## Step 3: API Design (2–3 min)

Define the interface before the internals. REST is the safe default.

```
POST   /api/v1/resource          → Create
GET    /api/v1/resource/{id}     → Read
PUT    /api/v1/resource/{id}     → Update
DELETE /api/v1/resource/{id}     → Delete
GET    /api/v1/resource?query=x  → Search/List
```

For each endpoint, briefly state:
- Request parameters
- Response format
- Authentication (API key, OAuth token)
- Rate limiting (mention it, shows awareness)

**Tip:** Mention pagination for any list endpoint: `?page=1&limit=20` or cursor-based.

---

## Step 4: Data Model + Database Choice (3–5 min)

### Define Your Entities
Draw the core tables/documents and their relationships.

Example format:
```
Table: users
  - user_id (PK)
  - name
  - email
  - created_at

Table: posts
  - post_id (PK)
  - user_id (FK)
  - content
  - created_at
```

### Choose Your Database

| Use Case | Pick | Why |
|----------|------|-----|
| Simple key-value lookups | **DynamoDB / Redis** | Fast, scales horizontally |
| Relationships + transactions | **PostgreSQL / MySQL** | ACID, joins, integrity |
| High write throughput, time-series | **Cassandra** | Write-optimized, distributed |
| Full-text search | **Elasticsearch** | Inverted index, fast queries |
| Graph relationships | **Neo4j** | Friend-of-friend, recommendations |
| File/blob storage | **S3** | Cheap, durable, scalable |

**Decision Framework:**
1. What's the access pattern? (key-value? joins? range queries?)
2. Read-heavy or write-heavy?
3. Need strong consistency or eventual is fine?
4. How much data? Need horizontal scaling?

---

## Step 5: High-Level Architecture (5–8 min)

**This is the core of your answer. Draw the boxes and arrows.**

### Standard Components (pick what applies)

```
Clients (Web/Mobile)
    │
Load Balancer
    │
App Servers (stateless, horizontally scaled)
    │
    ├── Cache Layer (Redis/Memcached)
    │       │
    │   Database (Primary + Replicas)
    │
    ├── Message Queue (Kafka/SQS) ← for async work
    │       │
    │   Worker Servers (background jobs)
    │
    ├── CDN ← for static content / media
    │
    ├── Object Storage (S3) ← for files, images, videos
    │
    └── Search Index (Elasticsearch) ← if search is needed
```

### Walk Through the Flows
Always explain two flows:
1. **Write path:** User creates something → what happens step by step?
2. **Read path:** User requests something → where does it come from?

**Tip:** Mention what's synchronous vs asynchronous. E.g., "The redirect is sync, but analytics logging is async via a message queue."

---

## Step 6: Deep Dives (10–15 min)

**This is where you win or lose. The interviewer picks 2–3 areas to probe.**

### Common Deep Dive Topics

**Scaling:**
- How to shard/partition the database? What's the partition key?
- Consistent hashing for even distribution
- Read replicas for read-heavy workloads
- Horizontal vs vertical scaling

**Caching:**
- What to cache? (hot data, computed results)
- Cache eviction policy? (LRU, LFU, TTL)
- Cache-aside vs write-through vs write-behind?
- Cache invalidation — how do you keep cache in sync with DB?
- Cold start problem — how to warm the cache?

**Consistency:**
- Strong vs eventual consistency — what's acceptable here?
- How do read replicas cause staleness?
- Conflict resolution for concurrent writes

**Availability + Fault Tolerance:**
- What happens if [component X] goes down?
- Single points of failure — how to eliminate them?
- Data replication strategy (sync vs async)
- Failover mechanisms

**Data Flow:**
- Sync vs async processing
- Message queues for decoupling (Kafka, SQS, RabbitMQ)
- Event-driven architecture
- Idempotency for retry safety

**Security:**
- Authentication / Authorization
- Rate limiting (token bucket, sliding window)
- Data encryption (at rest, in transit)

### How to Handle Deep Dives
1. State the problem clearly: "The challenge here is..."
2. Present 2–3 options with trade-offs
3. Pick one and justify: "I'd go with X because..."
4. Acknowledge the weakness: "The downside is... but we can mitigate by..."

---

## Step 7: Wrap Up (1–2 min)

Briefly mention things you'd add with more time:
- Monitoring and alerting (Grafana, PagerDuty)
- Logging and tracing (distributed tracing with Jaeger)
- CI/CD pipeline
- A/B testing infrastructure
- Geographic distribution (multi-region)

---

## Quick Reference: Patterns That Repeat Across Designs

| Pattern | When to Use | Example Systems |
|---------|-------------|-----------------|
| **Cache-aside** | Read-heavy, tolerance for slight staleness | URL shortener, news feed |
| **Write-behind queue** | Async writes, decouple producers/consumers | Analytics, notifications |
| **Fan-out on write** | Pre-compute for fast reads | News feed (push model) |
| **Fan-out on read** | Write is simple, read computes on the fly | News feed (pull model) |
| **CQRS** | Reads and writes have very different patterns | E-commerce, search |
| **Event sourcing** | Need full history of changes | Banking, collaborative editing |
| **Consistent hashing** | Distribute data across nodes evenly | Any sharded system |
| **Pub/Sub** | Multiple consumers for same event | Notifications, real-time updates |
| **Leader-follower** | One writer, many readers | Database replication |
| **Circuit breaker** | Prevent cascade failures | Microservice communication |

---

## Common Mistakes to Avoid

1. **Jumping into boxes and arrows without clarifying requirements** — always ask questions first
2. **Ignoring the numbers** — estimation shows you think about real-world scale
3. **Designing for Google scale from minute one** — start simple, then say "to scale this, I'd..."
4. **Not discussing trade-offs** — never say "use X." Say "I'd use X because Y, though the trade-off is Z"
5. **Forgetting the read path** — most candidates explain writes well but hand-wave reads
6. **Single points of failure** — if any one component dying kills the system, call it out
7. **Over-engineering** — don't add Kafka, Redis, Elasticsearch, and a service mesh unless each one is justified

---

## The One-Line Version

**Requirements → Estimation → API → Data Model → Architecture → Deep Dives → Wrap Up**

Practice this flow enough that it becomes automatic. The framework is the skeleton — your knowledge of trade-offs is the muscle.
