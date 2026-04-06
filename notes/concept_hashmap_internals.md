# HashMap Internals — How It Actually Works

## Core Idea
A hashmap is an **array** where the index is computed from the key using a **hash function**.

```
put("apple", 5)
  → hash("apple") = 2198734
  → index = 2198734 % array_size → index 3
  → array[3] = ("apple", 5)

get("apple")
  → same hash → same index → return value
```

This is why lookups are O(1) — index is computed directly, no searching.

## Collision Handling

When two keys hash to the same index, two strategies:

### 1. Chaining (Java HashMap)
Each slot holds a **linked list**. Colliding entries are appended.

```
slot 3: ("apple", 5) → ("banana", 8) → null
```

- Lookup: hash → go to slot → walk list → compare keys
- Easy deletion (just remove node from list)
- Worse cache performance (pointer chasing across memory)
- Java 8+: chains longer than 8 nodes convert to **red-black tree** (O(log n) instead of O(n))

### 2. Open Addressing (Python dict)
No linked lists. If slot is taken, **probe** for next empty slot.

```
put("apple", 5)  → hash = 3 → slot 3 empty → place it
put("banana", 8) → hash = 3 → slot 3 taken → try slot 4 → place it
```

Probing strategies:
- **Linear probing:** try 3, 4, 5, 6... (simple but causes clustering)
- **Quadratic probing:** try 3, 4, 7, 12... (less clustering)
- **Double hashing:** use second hash function for step size (best distribution)

- Better cache performance (data is contiguous in memory)
- Tricky deletion (need tombstone markers to not break probe chains)

### Chaining vs Open Addressing

| | Chaining | Open Addressing |
|---|----------|----------------|
| Structure | Array of linked lists | Flat array, probe for next empty |
| Deletion | Easy | Tricky (tombstones) |
| Cache performance | Worse | Better |
| Used by | Java HashMap | Python dict |

## Load Factor and Rehashing

**Load factor** = number of entries / array size

When load factor exceeds **~0.75**, the hashmap:
1. Creates a new array (usually **2x** the size)
2. Rehashes every entry into the new array

This is why hashmap is **amortized O(1)** — most operations are O(1), but occasionally one insert triggers an O(n) resize. Averaged across all operations, it's still O(1).

## What Makes a Good Hash Function

1. **Deterministic** — same input → same output, always
2. **Uniform distribution** — spreads keys evenly (fewer collisions)
3. **Fast to compute** — runs on every get/put

Python uses **SipHash** for strings (also resistant to hash collision attacks — intentionally crafted keys that all map to same slot).

## Internal Structure Is Hidden

Both Python and Java hide the internal array from you. No `map.getSlot(3)` API exists. You only interact via `get(key)` and `put(key, value)`. The collision handling is completely transparent.

## 30-Second Interview Answer

> "A hashmap is an array where the index is computed from the key using a hash function. Collisions are handled by chaining (linked lists at each slot) or open addressing (probing for the next empty slot). When the load factor exceeds ~0.75, the array doubles and all entries are rehashed. This gives amortized O(1) for get, put, and delete."

## Complexity

| Operation | Average | Worst Case (all keys collide) |
|-----------|---------|-------------------------------|
| Get | O(1) | O(n) |
| Put | O(1) amortized | O(n) |
| Delete | O(1) | O(n) |
| Space | O(n) | O(n) |
