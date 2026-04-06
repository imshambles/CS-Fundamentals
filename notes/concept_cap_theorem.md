# CAP Theorem — Availability vs Consistency

## The CAP Theorem
In a distributed system, when a **network partition** (P) happens, you can only guarantee TWO of three:
- **C — Consistency:** Every read gets the most recent write
- **A — Availability:** Every request gets a response (even if slightly stale)
- **P — Partition Tolerance:** System works even if messages between nodes are lost

Partitions are unavoidable → real choice is **CP or AP**.

## CP vs AP

**CP:** "I'd rather give you an error than wrong data."
- Nodes that can't confirm latest data **refuse requests**
- Use for: money, inventory, bookings

**AP:** "I'd rather give you stale data than nothing."
- Every node **keeps serving** with whatever it has
- Use for: social feeds, URL shorteners, DNS

## Real-World Analogy
Two bank branches, shared account ($1000), phone line goes down:
- **CP:** Both stop withdrawals → frustrating but safe
- **AP:** Both keep serving → customers happy but account might overdraw

## Eventual Consistency
Most AP systems sync up within milliseconds to seconds. Not "wrong forever," just "briefly stale." Acceptable for a tweet, unacceptable for a bank transfer.

## Decision Framework

| System | Pick | Why |
|--------|------|-----|
| URL Shortener | AP | Redirects must always work |
| Chat App | AP | Delivery > perfect ordering |
| Payment System | CP | Wrong balance = real money lost |
| E-commerce Cart | AP | Cart can sync later |
| E-commerce Checkout | CP | Don't sell what you don't have |
| Social Media Feed | AP | Stale feed is fine |
| Bank Transfer | CP | Cannot duplicate money |

**Key insight:** Different parts of the SAME system can make different choices (e.g., e-commerce cart = AP, checkout = CP). Saying this in interviews shows depth.
