# URL Shortener — Revision Notes

## Requirements
- **Functional:** Shorten URL, redirect (core), custom aliases, expiration, analytics (optional)
- **Non-Functional:** Read-heavy (100:1), low latency (<50ms redirects), high availability (AP over CP), non-guessable codes

## Estimation (for 100M URLs/month)
- Writes: ~38/sec (manageable for any modern DB)
- Reads: ~3,900/sec (needs caching)
- 7-char code (base62) = 62⁷ ≈ 3.5 trillion unique URLs

## Short Code Generation — 3 Approaches

| Approach | How | Pros | Cons |
|----------|-----|------|------|
| **Hash (MD5/SHA)** | Hash long URL → take first 7 chars | Simple | Collisions likely, need retry loops |
| **Counter-based** | Auto-increment ID → base62 encode | Zero collisions | Predictable, coordination needed |
| **Pre-generated keys (KGS)** ✅ | Separate service generates random codes in advance | No collisions, no coordination, not predictable | Need to manage key pool |

**Winner: KGS** — production systems prefer this.

## Database
- **Choice: NoSQL (DynamoDB)** — access pattern is pure key-value lookup, no joins needed
- **Schema:** `short_code (PK) → long_url, created_at, expires_at`
- **Why not SQL:** We don't need relations/joins. Horizontal scaling is painful with SQL.
- **Partition key:** `short_code` — random codes distribute evenly via consistent hashing

**Rule of thumb:** Simple access pattern + horizontal scaling needed → NoSQL. Complex queries + relationships → SQL.

## Architecture

```
Client → Load Balancer → App Servers (stateless)
                              │
                         Redis Cache (LRU + TTL)
                              │ (miss only)
                         DynamoDB

         KGS (pre-generates keys, hands out in batches)
```

**Write Flow:** User → LB → App Server → grab key from KGS → save to DB → return short URL

**Read Flow:** User clicks → LB → App Server → check Redis → HIT: redirect / MISS: query DB → cache → redirect

## Key Design Decisions

**302 vs 301 Redirect:**
- **302 (Temporary)** ✅ — every click passes through us, enables analytics + mutable mappings
- **301 (Permanent)** ✗ — browser caches it, we lose tracking and can't update the mapping

**Cache Strategy:**
- **LRU + TTL** — LRU evicts least recently used (handles viral-then-dead URLs), TTL prevents stale data
- **Why not LFU:** A once-viral URL accumulates high count and never gets evicted even when dead
- **Cache warm-up:** Pre-load top N URLs on Redis restart to avoid cold start

**DB Scaling:**
- **Hash-based partitioning** over range-based — guarantees uniform distribution regardless of key patterns
- DynamoDB handles this automatically with `short_code` as partition key + consistent hashing

**KGS Resilience:**
- **Batch allocation:** Each app server grabs ~1,000-10,000 keys into local memory
- **Buffer:** 10K keys at ~12 writes/sec/server = ~15 min buffer if KGS goes down
- **Standby replica:** Primary + standby KGS with separate key ranges, no duplicate risk
- **Lost keys on crash:** Negligible out of 3.5 trillion — not worth recovering

## Quick Recall Checklist
- [ ] Can I explain the full write + read flow in 2 minutes?
- [ ] Can I justify NoSQL over SQL for this use case?
- [ ] Can I explain why KGS > Hashing > Counter?
- [ ] Can I explain 302 vs 301 trade-off?
- [ ] Can I discuss cache eviction (LRU vs LFU vs TTL)?
- [ ] Can I explain how KGS stays resilient?
- [ ] Can I do back-of-envelope estimation?
