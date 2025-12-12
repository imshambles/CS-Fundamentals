# 🧭 DSA Decision Tree & Cheat Sheet

> A simple, visual guide to select the right **Data Structure** and **Algorithm** with step-by-step implementation rules.

---

## 📋 Table of Contents
1. [Problem Analysis Checklist](#-step-1-problem-analysis-checklist)
2. [Data Structure Decision Tree](#-data-structure-decision-tree)
3. [Data Structure Cheat Sheets](#-data-structure-cheat-sheets)
4. [Algorithm Decision Tree](#-algorithm-decision-tree)
5. [Algorithm Cheat Sheets](#-algorithm-cheat-sheets)

---

## 🔍 Step 1: Problem Analysis Checklist

Before choosing anything, answer these:

| # | Question | What It Tells You |
|---|----------|-------------------|
| 1 | What's the **input size (n)**? | Time complexity you can afford |
| 2 | What **operations** are needed? | Which data structure fits |
| 3 | What's asked: **single value, list, boolean**? | Output format |
| 4 | Is input **sorted**? | Binary search possible? |
| 5 | Are there **duplicates** to handle? | Need hash set/map? |

### Constraint → Complexity Quick Map

| If n ≤ | You can use | Example Algorithms |
|--------|-------------|-------------------|
| 10 | O(n!) | Brute force, permutations |
| 20 | O(2ⁿ) | Bitmask, subsets |
| 500 | O(n³) | Floyd-Warshall |
| 5,000 | O(n²) | Nested loops, simple DP |
| 100,000 | O(n log n) | Sorting, heap, binary search |
| 1,000,000 | O(n) | Linear scan, hash map |
| 10⁹+ | O(log n) | Binary search, math |

---

## 🌳 Data Structure Decision Tree

```
START: What do you need to do?
│
├─▶ Need ORDERING? (process in specific order)
│   │
│   ├─▶ Last In, First Out? ──────▶ 📚 STACK
│   │
│   ├─▶ First In, First Out? ─────▶ 📬 QUEUE
│   │
│   └─▶ By Priority (min/max)? ───▶ ⛰️ HEAP
│
├─▶ Need FAST LOOKUP? (O(1) search)
│   │
│   ├─▶ Just check existence? ────▶ 🔵 HASH SET
│   │
│   └─▶ Store key-value pairs? ───▶ 🗂️ HASH MAP
│
├─▶ Need SORTED DATA + Search?
│   │
│   ├─▶ Static data? ─────────────▶ 📊 SORTED ARRAY + Binary Search
│   │
│   └─▶ Dynamic insert/delete? ───▶ 🌲 BST / TreeMap
│
├─▶ Need PREFIX operations?
│   │
│   └─▶ Prefix search/autocomplete? ▶ 🔤 TRIE
│
├─▶ Need HIERARCHY?
│   │
│   └─▶ Parent-child relationship? ─▶ 🌳 TREE
│
├─▶ Need RELATIONSHIPS/CONNECTIONS?
│   │
│   ├─▶ Sparse connections? ──────▶ 📋 ADJACENCY LIST
│   │
│   └─▶ Dense connections? ───────▶ 📐 ADJACENCY MATRIX
│
├─▶ Need RANGE QUERIES with updates?
│   │
│   └─▶ Sum/Min/Max in ranges? ───▶ 🌲 SEGMENT TREE / BIT
│
└─▶ Need GROUPING / CONNECTIVITY?
    │
    └─▶ Merge groups, find if connected? ▶ 🔗 UNION-FIND
```

---

## 📘 Data Structure Cheat Sheets

### 📚 STACK
**When to Use:** Parentheses matching, undo operations, DFS, monotonic problems, expression evaluation

| Cheat Sheet |
|-------------|
| 1. Create empty stack |
| 2. **Push** elements onto top |
| 3. **Pop** returns and removes top |
| 4. **Peek** checks top without removing |
| 5. Process until stack is empty |

**Key Patterns:**
| Pattern | Rule |
|---------|------|
| Matching brackets | Push open, pop on close, check match |
| Monotonic stack | Pop while condition fails, then push |
| Previous greater | Maintain decreasing stack of indices |
| Next greater | Process from right or use result array |

---

### 📬 QUEUE
**When to Use:** BFS, level-order traversal, scheduling, sliding window max

| Cheat Sheet |
|-------------|
| 1. Create empty queue |
| 2. **Enqueue** adds to back |
| 3. **Dequeue** removes from front |
| 4. Process level by level (for BFS) |
| 5. Track size at each level if needed |

**Key Patterns:**
| Pattern | Rule |
|---------|------|
| BFS | Add start, process level by level |
| Level order | Save queue size, process that many |
| Monotonic deque | Maintain sorted order in deque |

---

### ⛰️ HEAP (Priority Queue)
**When to Use:** K-th element, merge K lists, scheduling, top K problems

| Cheat Sheet |
|-------------|
| 1. Choose **Min Heap** or **Max Heap** |
| 2. Insert all elements (or first k) |
| 3. Pop to get smallest/largest |
| 4. For K-largest: use min heap of size K |
| 5. For K-smallest: use max heap of size K |

**Key Patterns:**
| Pattern | Rule |
|---------|------|
| K-th largest | Min heap of size K, answer = top |
| K-th smallest | Max heap of size K, answer = top |
| Merge K sorted | Heap of (value, list_idx, elem_idx) |
| Median stream | Max heap (left) + Min heap (right) |
| Top K frequent | Count first, then heap by frequency |

---

### 🗂️ HASH MAP / HASH SET
**When to Use:** Two sum, counting frequency, checking duplicates, O(1) lookup

| Cheat Sheet |
|-------------|
| 1. Create empty map/set |
| 2. **Set**: Add item for existence check |
| 3. **Map**: Store key→value pairs |
| 4. Lookup is O(1) average |
| 5. Watch out for hash collisions (rare) |

**Key Patterns:**
| Pattern | Rule |
|---------|------|
| Two Sum | Store value→index, check target-current |
| Frequency count | map[item] += 1 |
| Check duplicate | If item in set: found, else add |
| Group by key | map[key].append(item) |
| Subarray sum = K | Store prefix_sum→count |

---

### 🌲 TREE / BST
**When to Use:** Hierarchical data, sorted dynamic data, range queries

| Cheat Sheet for Binary Tree Traversal |
|---------------------------------------|
| 1. **Preorder**: Root → Left → Right (copy tree) |
| 2. **Inorder**: Left → Root → Right (sorted order for BST) |
| 3. **Postorder**: Left → Right → Root (delete tree) |
| 4. **Level order**: Use queue, process by levels |

**Key Patterns:**
| Pattern | Rule |
|---------|------|
| Find height | max(left_height, right_height) + 1 |
| Check BST | Pass min/max bounds down recursively |
| LCA | If both in left, go left; both in right, go right; else current is LCA |
| Path sum | Subtract from target as you go down |

---

### 🔤 TRIE (Prefix Tree)
**When to Use:** Autocomplete, prefix search, word dictionary, word search in grid

| Cheat Sheet |
|-------------|
| 1. Create root node (empty) |
| 2. Each node has children map + `is_end` flag |
| 3. **Insert**: Walk down, create nodes, mark end |
| 4. **Search**: Walk down, check `is_end` |
| 5. **StartsWith**: Walk down, return if path exists |

**Key Patterns:**
| Pattern | Rule |
|---------|------|
| Insert word | For each char: create child if missing, move down, mark last as end |
| Search word | Walk path, return true only if path ends AND is_end=true |
| Prefix exists | Walk path, return true if path exists (ignore is_end) |
| Autocomplete | Walk to prefix, then DFS to collect all words |

---

### 🔗 UNION-FIND (Disjoint Set)
**When to Use:** Connected components, cycle detection, grouping, Kruskal's MST

| Cheat Sheet |
|-------------|
| 1. Initialize: each element is its own parent |
| 2. **Find(x)**: Follow parents to root (use path compression) |
| 3. **Union(x, y)**: Connect roots (use union by rank) |
| 4. Same component? → find(x) == find(y) |
| 5. Cycle exists? → If find(u) == find(v) before union |

**Key Patterns:**
| Pattern | Rule |
|---------|------|
| Count components | N - (number of successful unions) |
| Detect cycle | Before adding edge, check if already connected |
| Group items | Union all related items, count unique roots |

---

### 📊 SEGMENT TREE / BIT
**When to Use:** Range sum/min/max queries with point updates

| Cheat Sheet |
|-------------|
| 1. Build tree from array (O(n)) |
| 2. **Query(l, r)**: Get sum/min/max in range (O(log n)) |
| 3. **Update(i, val)**: Change one element (O(log n)) |
| 4. Tree size = 4 × n (for safety) |
| 5. Left child = 2i+1, Right child = 2i+2 |

---

## 🎯 Algorithm Decision Tree

```
START: What type of problem?
│
├─▶ SEARCHING for something?
│   │
│   ├─▶ Data is sorted? ──────────────▶ 🔍 BINARY SEARCH
│   │
│   ├─▶ Search in graph/tree? ────────▶ 🌊 BFS or DFS
│   │
│   └─▶ Find all combinations? ───────▶ 🔙 BACKTRACKING
│
├─▶ SHORTEST PATH?
│   │
│   ├─▶ Unweighted graph? ────────────▶ 🌊 BFS
│   │
│   ├─▶ Weighted (non-negative)? ─────▶ 🛣️ DIJKSTRA
│   │
│   ├─▶ Weighted (can be negative)? ──▶ 🔔 BELLMAN-FORD
│   │
│   └─▶ All pairs shortest path? ─────▶ 🌐 FLOYD-WARSHALL
│
├─▶ OPTIMIZATION (min/max)?
│   │
│   ├─▶ Overlapping subproblems? ─────▶ 📊 DYNAMIC PROGRAMMING
│   │
│   └─▶ Local optimal = Global optimal? ▶ 💰 GREEDY
│
├─▶ Working with SUBARRAYS?
│   │
│   ├─▶ Fixed size window? ───────────▶ 🪟 SLIDING WINDOW (fixed)
│   │
│   ├─▶ Variable size window? ────────▶ 🪟 SLIDING WINDOW (expand/shrink)
│   │
│   └─▶ Need pairs from sorted? ──────▶ 👆 TWO POINTERS
│
├─▶ GRAPH problems?
│   │
│   ├─▶ Traversal/explore all? ───────▶ 🌊 DFS
│   │
│   ├─▶ Level by level? ──────────────▶ 🌊 BFS
│   │
│   ├─▶ Dependencies/ordering? ───────▶ 📋 TOPOLOGICAL SORT
│   │
│   ├─▶ Connectivity/grouping? ───────▶ 🔗 UNION-FIND
│   │
│   └─▶ Minimum spanning tree? ───────▶ 🌲 KRUSKAL / PRIM
│
├─▶ Need PREVIOUS/NEXT greater/smaller?
│   │
│   └─▶ Monotonic relationship? ──────▶ 📚 MONOTONIC STACK
│
└─▶ Working with INTERVALS?
    │
    └─▶ Merge, insert, schedule? ─────▶ 📏 SORT BY START + GREEDY
```

---

## 📗 Algorithm Cheat Sheets

### 🔍 BINARY SEARCH
**When to Use:** Sorted array, search space can be halved, "minimum X that satisfies Y"

| Step | Action |
|------|--------|
| 1 | Set `left = 0`, `right = n-1` (or search space bounds) |
| 2 | While `left <= right`: |
| 3 | → Calculate `mid = left + (right - left) // 2` |
| 4 | → If `arr[mid] == target`: return mid |
| 5 | → If `arr[mid] < target`: `left = mid + 1` |
| 6 | → If `arr[mid] > target`: `right = mid - 1` |
| 7 | Return -1 (or `left` for insertion point) |

**Variants:**
| Variant | Modification |
|---------|--------------|
| First occurrence | When found, save answer, `right = mid - 1` |
| Last occurrence | When found, save answer, `left = mid + 1` |
| Search on answer | Binary search on result value, check if possible |

---

### 👆 TWO POINTERS
**When to Use:** Sorted array pairs, palindrome check, remove duplicates

| Pattern | Steps |
|---------|-------|
| **Opposite ends** | 1. left=0, right=n-1 |
| | 2. While left < right: |
| | 3. → Process arr[left] and arr[right] |
| | 4. → Move left++ or right-- based on condition |
| **Same direction** | 1. slow=0, fast=0 |
| | 2. Move fast through array |
| | 3. Update slow when condition met |
| | 4. slow marks the answer position |

**Key Rules:**
| Problem Type | Rule |
|--------------|------|
| Two sum (sorted) | Sum too big → right--, too small → left++ |
| Remove duplicates | slow=write position, fast=read position |
| Cycle detection | slow moves 1, fast moves 2, if meet → cycle |

---

### 🪟 SLIDING WINDOW
**When to Use:** Contiguous subarray/substring, window of size K, "longest/shortest with condition"

| Type | Steps |
|------|-------|
| **Fixed Window** | 1. Compute first window of size K |
| | 2. Slide: add new element, remove old element |
| | 3. Update answer at each position |
| **Variable Window** | 1. Initialize left=0, right=0 |
| | 2. Expand right (add element to window) |
| | 3. While window invalid: shrink left |
| | 4. Update answer when window is valid |

**Template Rules:**
| Condition | Action |
|-----------|--------|
| Window too big/invalid | Shrink from left |
| Window valid | Update answer, may continue expanding |
| Need exactly K distinct | Track count, shrink when exceeds K |

---

### 🌊 BFS (Breadth-First Search)
**When to Use:** Shortest path (unweighted), level-order traversal, spreading problems

| Step | Action |
|------|--------|
| 1 | Create queue, add starting node(s) |
| 2 | Create visited set, mark start as visited |
| 3 | While queue not empty: |
| 4 | → Dequeue front node |
| 5 | → Process node (check if goal) |
| 6 | → For each unvisited neighbor: mark visited, enqueue |
| 7 | Track distance/level if needed |

**Key Rules:**
| Scenario | Rule |
|----------|------|
| Shortest path | First time you reach target = shortest |
| Level tracking | Process all nodes in queue before moving to next level |
| Grid traversal | Use 4 directions: (0,1), (0,-1), (1,0), (-1,0) |
| Multi-source | Add all sources to queue initially |

---

### 🌲 DFS (Depth-First Search)
**When to Use:** Path finding, cycle detection, connected components, backtracking

| Step | Action |
|------|--------|
| 1 | Start at source node |
| 2 | Mark current node as visited |
| 3 | Process current node |
| 4 | For each unvisited neighbor: |
| 5 | → Recursively DFS on neighbor |
| 6 | (Optional) Unmark visited for backtracking |

**Key Rules:**
| Scenario | Rule |
|----------|------|
| Cycle detection (undirected) | Track parent, if neighbor visited AND not parent → cycle |
| Cycle detection (directed) | Use 3 states: unvisited, visiting, visited |
| Path exists | Return true when target found |
| All paths | Backtrack (unmark visited after recursion) |

---

### 📊 DYNAMIC PROGRAMMING
**When to Use:** Overlapping subproblems + optimal substructure, count ways, min/max optimization

| Step | Action |
|------|--------|
| 1 | **Define state**: What does dp[i] or dp[i][j] represent? |
| 2 | **Base case**: What's dp[0] or smallest subproblem? |
| 3 | **Recurrence**: How does dp[i] depend on smaller states? |
| 4 | **Order**: Fill table in correct order (usually small → large) |
| 5 | **Answer**: Which cell contains the final answer? |

**Common Patterns:**
| Pattern | State Definition | Recurrence |
|---------|-----------------|------------|
| **Fibonacci/Climbing** | dp[i] = ways to reach i | dp[i] = dp[i-1] + dp[i-2] |
| **Knapsack 0/1** | dp[i][w] = max value with first i items, capacity w | dp[i][w] = max(skip, take) |
| **LCS** | dp[i][j] = LCS of first i chars and first j chars | Match: dp[i-1][j-1]+1, else max(dp[i-1][j], dp[i][j-1]) |
| **LIS** | dp[i] = LIS ending at i | dp[i] = max(dp[j]+1) for all j < i where arr[j] < arr[i] |
| **Coin change** | dp[i] = min coins for amount i | dp[i] = min(dp[i-coin]+1) for all coins |

---

### 🔙 BACKTRACKING
**When to Use:** Generate all possibilities, permutations, combinations, constraint satisfaction

| Step | Action |
|------|--------|
| 1 | **Choose**: Make a choice (add element to path) |
| 2 | **Explore**: Recurse with remaining choices |
| 3 | **Unchoose**: Remove element (backtrack) |
| 4 | **Base case**: When to stop and save result |
| 5 | **Pruning**: Skip invalid branches early |

**Template:**
```
backtrack(choices, path):
    if path is complete:
        save path to results
        return
    
    for each choice in choices:
        if choice is valid:
            add choice to path
            backtrack(remaining choices, path)
            remove choice from path  ← BACKTRACK
```

**Key Rules:**
| Problem | Rule |
|---------|------|
| Subsets | For each element: include or exclude |
| Permutations | Try each unused element at current position |
| Combinations | Only consider elements after current index |
| No duplicates | Sort first, skip if same as previous |

---

### 📋 TOPOLOGICAL SORT
**When to Use:** Task scheduling, course prerequisites, build order (DAG only)

| Method | Steps |
|--------|-------|
| **Kahn's (BFS)** | 1. Calculate in-degree for all nodes |
| | 2. Add all nodes with in-degree 0 to queue |
| | 3. While queue not empty: |
| | 4. → Dequeue node, add to result |
| | 5. → Decrease in-degree of neighbors |
| | 6. → If neighbor's in-degree becomes 0, enqueue it |
| | 7. If result size ≠ n → cycle exists |
| **DFS-based** | 1. DFS on each unvisited node |
| | 2. After all children processed, add to stack |
| | 3. Reverse stack = topological order |

---

### 🛣️ DIJKSTRA
**When to Use:** Shortest path with non-negative weights

| Step | Action |
|------|--------|
| 1 | Set dist[start] = 0, all others = ∞ |
| 2 | Add (0, start) to min heap |
| 3 | While heap not empty: |
| 4 | → Pop (d, node) with smallest distance |
| 5 | → If d > dist[node]: skip (outdated) |
| 6 | → For each neighbor: |
| 7 | →→ new_dist = dist[node] + weight |
| 8 | →→ If new_dist < dist[neighbor]: update and push |

---

### 📚 MONOTONIC STACK
**When to Use:** Next/previous greater/smaller element

| Type | Steps |
|------|-------|
| **Next Greater** | 1. Traverse right to left |
| | 2. Pop while stack top ≤ current |
| | 3. Answer = stack top (or -1 if empty) |
| | 4. Push current to stack |
| **Previous Greater** | 1. Traverse left to right |
| | 2. Pop while stack top ≤ current |
| | 3. Answer = stack top (or -1 if empty) |
| | 4. Push current to stack |

**Key Rules:**
| Looking for | Stack maintains | Pop condition |
|-------------|-----------------|---------------|
| Next greater | Decreasing | top ≤ current |
| Next smaller | Increasing | top ≥ current |
| Previous greater | Decreasing | top ≤ current |
| Previous smaller | Increasing | top ≥ current |

---

### 💰 GREEDY
**When to Use:** Local optimal choice leads to global optimal (activity selection, intervals, Huffman)

| Step | Action |
|------|--------|
| 1 | Sort by appropriate criteria |
| 2 | At each step, make locally optimal choice |
| 3 | Never reconsider past choices |
| 4 | Prove correctness by exchange argument |

**Common Greedy Strategies:**
| Problem Type | Sort By | Greedy Choice |
|--------------|---------|---------------|
| Activity selection | End time | Pick earliest ending |
| Interval scheduling | Start time | Merge overlapping |
| Min platforms | Events | Process arrivals/departures |
| Fractional knapsack | Value/weight | Take highest ratio |
| Min coins (if works) | Largest coin | Take as many as possible |

---

## 🎓 Quick Pattern Recognition

| You See This... | Try This... |
|-----------------|-------------|
| "Sorted array" | Binary Search |
| "Minimum/maximum subarray" | Sliding Window or Kadane |
| "Shortest path" | BFS (unweighted) or Dijkstra |
| "All possibilities" | Backtracking |
| "Longest/shortest subsequence" | DP |
| "Contiguous" | Sliding Window |
| "K-th element" | Heap |
| "Parentheses/brackets" | Stack |
| "Level by level" | BFS |
| "Connected components" | Union-Find or DFS |
| "Dependencies" | Topological Sort |
| "Next greater" | Monotonic Stack |
| "Prefix sum" | Precompute prefix array |

---

## ✅ Final Problem-Solving Checklist

- [ ] Read problem twice
- [ ] Note constraints (what complexity needed?)
- [ ] Identify input/output types
- [ ] Look for keywords (sorted, minimum, all, etc.)
- [ ] Pick data structure from decision tree
- [ ] Pick algorithm from decision tree
- [ ] Write steps in comments first
- [ ] Code the solution
- [ ] Test with examples
- [ ] Check edge cases (empty, single element, duplicates)

---

*Last Updated: December 2024*
