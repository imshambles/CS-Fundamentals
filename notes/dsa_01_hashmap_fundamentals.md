# DSA Day 1: HashMap Fundamentals (Apr 6, 2026)

## Pattern: HashMap / Frequency Counting
**When to use:** "frequency", "unique", "duplicate", "count", "anagram", "two sum"

## Problem 1: Two Sum (#1)

**Approach:** One-pass hashmap — store each number's index, check if complement exists.

```python
def two_sum(nums, target):
    hashmap = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hashmap:
            return [hashmap[complement], i]
        hashmap[num] = i
    return []
```

**Key insight:** Complement check BEFORE insertion avoids using the same element twice.

**Complexity:** Time O(n), Space O(n)

## Problem 2: Valid Anagram (#242)

**Approach:** Count character frequencies in first string, decrement with second, check all zeros.

```python
def valid_anagram(s, t):
    hashmap = {}
    for c in s:
        if c not in hashmap:
            hashmap[c] = 1
        else:
            hashmap[c] += 1
    for c in t:
        if c not in hashmap:
            return False
        else:
            hashmap[c] -= 1
    return all(v == 0 for v in hashmap.values())
```

**Cleaner version with defaultdict:**
```python
from collections import defaultdict

def valid_anagram(s, t):
    hashmap = defaultdict(int)
    for c in s:
        hashmap[c] += 1
    for c in t:
        hashmap[c] -= 1
    return all(v == 0 for v in hashmap.values())
```

**One-liner (mention in interview, then implement manually):**
```python
from collections import Counter
def valid_anagram(s, t):
    return Counter(s) == Counter(t)
```

**Gotcha:** Just checking that all chars in t exist in s isn't enough — must verify counts match (e.g., "aab" vs "ab"). Either check `all values == 0` or verify `len(s) == len(t)` upfront.

**Complexity:** Time O(n), Space O(n)

## Takeaways
- HashMap turns O(n²) brute force into O(n) by trading space for time
- `enumerate()` > `range(len())` in Python
- `defaultdict(int)` or `.get(key, 0)` avoids if/else for initialization
- Always state time + space complexity after solving
