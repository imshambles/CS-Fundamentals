# Interview Prep Roadmap — 12 Weeks
**Start Date:** April 7, 2026 (Monday)
**Target:** Interview-ready by July 1, 2026
**Daily commitment:** 1–2 hours (more on weekends if possible)
**Target companies:** Mix of FAANG/Big Tech + high-growth startups

---

## Strategy

You can already solve some mediums, so we're skipping pure fundamentals. The plan focuses on:

1. **Pattern mastery** — not grinding 500 problems, but owning ~15 core patterns
2. **System design from Week 1** — your 5 YOE means interviewers expect depth here
3. **Realistic pacing** — 1–2 hours/day means ~1 DSA problem + review OR 1 system design session per day

### Weekly Split
- **Weekdays (Mon–Fri):** DSA patterns (1–2 problems/day, focused on one pattern per week)
- **Weekends (Sat–Sun):** System design (1 full design per weekend) + weekly DSA review
- **Every 3rd weekend:** Mock interview simulation

### How to Use Each Session with Claude
1. **"Give me today's problem"** — I'll match it to your current week/pattern
2. **"Review my code"** — Paste your solution, I'll give interview-grade feedback
3. **"Explain [pattern]"** — I'll teach the concept, then quiz you
4. **"Mock interview me"** — Timed problem, hints only if stuck 10+ min
5. **"System design: [topic]"** — We'll walk through it like a real 45-min round
6. **"What's my progress?"** — I'll check where you stand

---

## PHASE 1: Core Patterns (Weeks 1–4)

### Week 1 (Apr 7–13): Hash Maps + Two Pointers
**Why together:** These two patterns solve ~30% of interview questions combined.

| Day | Focus | Problems | Time |
|-----|-------|----------|------|
| Mon | HashMap fundamentals | #1 Two Sum, #242 Valid Anagram | 1h |
| Tue | HashMap — frequency counting | #347 Top K Frequent Elements, #49 Group Anagrams | 1h |
| Wed | Two pointers — sorted arrays | #167 Two Sum II, #15 3Sum | 1h |
| Thu | Two pointers — opposite ends | #11 Container With Most Water, #42 Trapping Rain Water | 1.5h |
| Fri | Fast & slow pointers | #141 Linked List Cycle, #287 Find Duplicate | 1h |
| **Sat** | **System Design #1: URL Shortener** | Design TinyURL (hashing, DB choice, read-heavy) | **1.5h** |
| **Sun** | **Week review** | Redo any problem you struggled with | **1h** |

### Week 2 (Apr 14–20): Sliding Window + Binary Search
| Day | Focus | Problems | Time |
|-----|-------|----------|------|
| Mon | Fixed-size sliding window | #643 Max Avg Subarray, #1456 Max Vowels in Substring | 1h |
| Tue | Variable-size window | #3 Longest Substring Without Repeating, #76 Minimum Window Substring | 1.5h |
| Wed | Binary search — classic | #704 Binary Search, #33 Search in Rotated Sorted Array | 1h |
| Thu | Binary search on answer space | #875 Koko Eating Bananas, #1011 Capacity to Ship Packages | 1.5h |
| Fri | Mixed practice | #209 Min Size Subarray Sum, #74 Search a 2D Matrix | 1h |
| **Sat** | **System Design #2: Rate Limiter** | Token bucket, sliding window, distributed rate limiting | **1.5h** |
| **Sun** | **Week review + mock** | 2 timed problems (25 min each) | **1h** |

### Week 3 (Apr 21–27): Stacks/Queues + Intervals
| Day | Focus | Problems | Time |
|-----|-------|----------|------|
| Mon | Monotonic stack | #739 Daily Temperatures, #496 Next Greater Element | 1h |
| Tue | Stack applications | #20 Valid Parentheses, #155 Min Stack, #84 Largest Rectangle in Histogram | 1.5h |
| Wed | Interval problems | #56 Merge Intervals, #57 Insert Interval | 1h |
| Thu | Interval scheduling | #435 Non-overlapping Intervals, #253 Meeting Rooms II | 1h |
| Fri | Mixed stack + interval | #150 Evaluate RPN, #986 Interval List Intersections | 1h |
| **Sat** | **System Design #3: Chat System (WhatsApp)** | WebSockets, message queues, presence, delivery receipts | **1.5h** |
| **Sun** | **Review + weak areas** | Focus on whatever felt hardest | **1h** |

### Week 4 (Apr 28–May 4): Trees + BFS/DFS
| Day | Focus | Problems | Time |
|-----|-------|----------|------|
| Mon | Tree traversals + basics | #102 Level Order, #104 Max Depth, #226 Invert Tree | 1h |
| Tue | BST properties | #98 Validate BST, #230 Kth Smallest, #235 LCA of BST | 1.5h |
| Wed | Tree construction + serialization | #105 From Preorder+Inorder, #297 Serialize/Deserialize | 1.5h |
| Thu | Path problems | #112 Path Sum, #124 Binary Tree Max Path Sum | 1h |
| Fri | BFS on trees | #199 Right Side View, #103 Zigzag Level Order | 1h |
| **Sat** | **System Design #4: News Feed (Twitter/X)** | Fan-out, ranking, caching, push vs pull | **1.5h** |
| **Sun** | **Phase 1 mock** | 3 problems in 60 min (timed, no hints) | **1.5h** |

---

## PHASE 2: Advanced Patterns (Weeks 5–8)

### Week 5 (May 5–11): Graphs
| Day | Focus | Problems | Time |
|-----|-------|----------|------|
| Mon | Graph BFS + DFS | #200 Number of Islands, #733 Flood Fill | 1h |
| Tue | Connected components | #323 Number of Connected Components, #547 Number of Provinces | 1h |
| Wed | Topological sort | #207 Course Schedule, #210 Course Schedule II | 1.5h |
| Thu | Shortest path (Dijkstra) | #743 Network Delay Time, #787 Cheapest Flights K Stops | 1.5h |
| Fri | Union-Find | #684 Redundant Connection, #721 Accounts Merge | 1h |
| **Sat** | **System Design #5: Notification System** | Push vs pull, priority, multi-channel delivery | **1.5h** |
| **Sun** | **Review** | Redo graph problems, focus on templates | **1h** |

### Week 6 (May 12–18): Dynamic Programming — Part 1
| Day | Focus | Problems | Time |
|-----|-------|----------|------|
| Mon | DP framework (state → recurrence → base) | #70 Climbing Stairs, #198 House Robber | 1h |
| Tue | Knapsack patterns | #322 Coin Change, #416 Partition Equal Subset Sum | 1.5h |
| Wed | Subsequence DP | #300 LIS, #1143 Longest Common Subsequence | 1.5h |
| Thu | String DP | #72 Edit Distance, #5 Longest Palindromic Substring | 1.5h |
| Fri | Grid DP | #62 Unique Paths, #64 Minimum Path Sum | 1h |
| **Sat** | **System Design #6: Distributed Cache (Redis)** | Eviction, sharding, consistency, replication | **1.5h** |
| **Sun** | **DP review** | Re-derive recurrences from scratch (no code, just logic) | **1h** |

### Week 7 (May 19–25): Dynamic Programming — Part 2 + Backtracking
| Day | Focus | Problems | Time |
|-----|-------|----------|------|
| Mon | Stock buy/sell series (state machine DP) | #121, #122, #309 With Cooldown | 1.5h |
| Tue | DP on trees | #337 House Robber III, #124 Max Path Sum | 1h |
| Wed | Backtracking — subsets/permutations | #78 Subsets, #46 Permutations | 1h |
| Thu | Backtracking — combinations | #39 Combination Sum, #79 Word Search | 1.5h |
| Fri | Hard backtracking | #51 N-Queens, #37 Sudoku Solver | 1.5h |
| **Sat** | **System Design #7: YouTube / Video Streaming** | Upload pipeline, CDN, adaptive bitrate, recommendations | **1.5h** |
| **Sun** | **Phase 2 mock** | 3 problems in 60 min + 1 system design in 35 min | **2h** |

### Week 8 (May 26–Jun 1): Heaps + Tries + Greedy
| Day | Focus | Problems | Time |
|-----|-------|----------|------|
| Mon | Heap / Top-K pattern | #215 Kth Largest, #347 Top K Frequent (heap approach) | 1h |
| Tue | Merge K pattern | #23 Merge K Sorted Lists, #373 Find K Pairs | 1.5h |
| Wed | Trie | #208 Implement Trie, #211 Add and Search Word | 1h |
| Thu | Greedy fundamentals | #55 Jump Game, #45 Jump Game II | 1h |
| Fri | Greedy + intervals | #452 Min Arrows to Burst Balloons, #621 Task Scheduler | 1.5h |
| **Sat** | **System Design #8: Web Crawler** | Politeness, dedup, URL frontier, distributed crawling | **1.5h** |
| **Sun** | **Review** | Pattern flashcards — can you name the pattern in <30 sec? | **1h** |

---

## PHASE 3: Hard Problems + System Design Depth (Weeks 9–10)

### Week 9 (Jun 2–8): Hard LeetCode Patterns
| Day | Focus | Problems | Time |
|-----|-------|----------|------|
| Mon | Monotonic stack (hard) | #84 Largest Rectangle, #85 Maximal Rectangle | 1.5h |
| Tue | Hard sliding window | #239 Sliding Window Maximum, #480 Sliding Window Median | 1.5h |
| Wed | Hard graph | #269 Alien Dictionary, #332 Reconstruct Itinerary | 1.5h |
| Thu | Hard DP | #312 Burst Balloons, #1235 Max Profit Job Scheduling | 1.5h |
| Fri | Hard tree/design | #295 Find Median from Data Stream, #380 Insert Delete GetRandom O(1) | 1.5h |
| **Sat** | **System Design #9: Search Autocomplete** | Trie + ranking, caching, distributed aggregation | **1.5h** |
| **Sun** | **System Design #10: Uber/Ride Sharing** | Geo-spatial indexing, matching, ETA, surge pricing | **1.5h** |

### Week 10 (Jun 9–15): System Design Deep Dives
| Day | Focus | Topic | Time |
|-----|-------|-------|------|
| Mon | DSA: Weak pattern review | Redo 2-3 problems from your weakest pattern | 1h |
| Tue | Design: Distributed systems concepts | CAP theorem, consistent hashing, leader election | 1.5h |
| Wed | Design: Database design | SQL vs NoSQL, sharding, indexing, replication | 1.5h |
| Thu | DSA: Design-oriented coding | #146 LRU Cache, #460 LFU Cache, #355 Design Twitter | 1.5h |
| Fri | Design: API design + estimation | RESTful principles, back-of-envelope calculations | 1h |
| **Sat** | **System Design #11: Google Docs (Collaborative Editing)** | OT/CRDT, conflict resolution, real-time sync | **1.5h** |
| **Sun** | **Phase 3 mock** | Full mock: 2 DSA (45 min) + 1 system design (45 min) | **2h** |

---

## PHASE 4: Interview Simulation (Weeks 11–12)

### Week 11 (Jun 16–22): Mock Interviews + Behavioral
| Day | Focus | Format | Time |
|-----|-------|--------|------|
| Mon | Behavioral prep | STAR method — prepare 5 stories (leadership, conflict, failure, impact, ambiguity) | 1h |
| Tue | Full mock #1 | 1 medium + 1 hard, 50 min, talk out loud | 1.5h |
| Wed | Behavioral practice | Practice telling stories concisely (2 min each) | 1h |
| Thu | Full mock #2 | System design round (45 min, no hints) | 1.5h |
| Fri | Review mock feedback | Fix patterns that broke under pressure | 1h |
| **Sat** | **Full mock #3** | Complete loop: behavioral (30m) + DSA (45m) + design (45m) | **2h** |
| **Sun** | **Weak area blitz** | 3-4 problems from your weakest patterns | **1.5h** |

### Week 12 (Jun 23–Jul 1): Final Polish
| Day | Focus | Format | Time |
|-----|-------|--------|------|
| Mon | Company-specific prep | Research target companies' interview styles | 1h |
| Tue | Full mock #4 | 2 mediums + 1 hard, 60 min | 1.5h |
| Wed | System design review | Rapid-fire: explain any of the 11 designs in 5 min | 1h |
| Thu | Full mock #5 | System design + follow-up deep dive | 1.5h |
| Fri | Pattern speed drill | Can you identify pattern + write skeleton in 5 min? | 1h |
| Sat–Sun | **Buffer** | Catch up, rest, or do one final mock | **Flexible** |

---

## System Design Topics Summary (11 Designs)

| # | System | Key Concepts |
|---|--------|-------------|
| 1 | URL Shortener | Hashing, base62, read-heavy, caching |
| 2 | Rate Limiter | Token bucket, sliding window, distributed |
| 3 | Chat System (WhatsApp) | WebSockets, message queue, presence, delivery |
| 4 | News Feed (Twitter) | Fan-out, ranking, push vs pull, caching |
| 5 | Notification System | Multi-channel, priority, templating |
| 6 | Distributed Cache (Redis) | Eviction, sharding, consistency |
| 7 | Video Streaming (YouTube) | CDN, transcoding, adaptive bitrate |
| 8 | Web Crawler | URL frontier, politeness, dedup |
| 9 | Search Autocomplete | Trie, ranking, distributed aggregation |
| 10 | Ride Sharing (Uber) | Geo-indexing, matching, ETA |
| 11 | Collaborative Editor (Google Docs) | CRDT/OT, conflict resolution, real-time sync |

---

## DSA Patterns Checklist (15 Patterns)

Track your confidence (1–5) as you go:

| # | Pattern | Key Signal | Confidence |
|---|---------|-----------|------------|
| 1 | HashMap / Counting | "frequency", "unique", "duplicate" | ☐ |
| 2 | Two Pointers | sorted array, pair finding | ☐ |
| 3 | Sliding Window | subarray/substring with constraint | ☐ |
| 4 | Binary Search | sorted data, "minimum maximum" | ☐ |
| 5 | Monotonic Stack | "next greater/smaller" | ☐ |
| 6 | BFS / Level-order | shortest path (unweighted), level processing | ☐ |
| 7 | DFS / Backtracking | generate all, permutations, subsets | ☐ |
| 8 | Tree Traversal | any tree problem | ☐ |
| 9 | Graph (BFS/DFS/Topo) | relationships, dependencies, grids | ☐ |
| 10 | Union-Find | connected components, grouping | ☐ |
| 11 | Heap / Top-K | "k largest/smallest", merge sorted | ☐ |
| 12 | Trie | prefix matching, autocomplete | ☐ |
| 13 | DP (1D) | optimization, counting ways | ☐ |
| 14 | DP (2D / Knapsack) | grid paths, subset sum, LCS | ☐ |
| 15 | Greedy | local optimum → global, intervals | ☐ |

---

## Key Metrics

- **Problems per week:** 8–12 (quality over quantity)
- **Target for mediums:** < 25 min by Week 8
- **Target for hards:** < 40 min by Week 12
- **System designs:** Be able to explain any of the 11 designs in a 35-min round
- **Total problems by end:** ~120–140 well-understood problems

---

*The goal isn't to solve every problem on LeetCode. It's to see a new problem and think "this is just [pattern] with a twist" within 2 minutes.*
