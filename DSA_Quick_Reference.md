# 🚀 DSA Quick Reference (1-Pager)

> **Print this!** Quick lookup during practice. For details, see `DSA_Problem_Solving_Roadmap.md`

---

## ⏱️ Constraints → Complexity

| n ≤ | Use | n ≤ | Use |
|-----|-----|-----|-----|
| 10 | O(n!) | 10⁵ | O(n log n) |
| 20 | O(2ⁿ) | 10⁶ | O(n) |
| 500 | O(n³) | 10⁹ | O(log n) |
| 5000 | O(n²) | | |

---

## 🗂️ Data Structure → When to Use

| DS | When |
|----|------|
| **Stack** | LIFO, parentheses, undo, monotonic |
| **Queue** | FIFO, BFS, level-order |
| **Heap** | K-th element, top K, merge K sorted |
| **HashMap** | O(1) lookup, counting, two sum |
| **HashSet** | Duplicates, existence check |
| **BST/TreeMap** | Sorted + dynamic insert/delete |
| **Trie** | Prefix search, autocomplete |
| **Union-Find** | Connectivity, grouping, cycle detection |
| **Segment Tree** | Range queries + updates |

---

## 🎯 Problem Pattern → Algorithm

| You See... | Use... |
|------------|--------|
| Sorted array | Binary Search |
| Shortest path (unweighted) | BFS |
| Shortest path (weighted) | Dijkstra |
| All possibilities | Backtracking |
| Min/max + overlapping | DP |
| Contiguous subarray | Sliding Window |
| Pairs in sorted | Two Pointers |
| Next/prev greater | Monotonic Stack |
| Dependencies/ordering | Topological Sort |
| Connected components | Union-Find / DFS |
| K-th element | Heap |
| Prefix/range sum | Prefix Sum |

---

## 🔧 Algorithm Templates (Condensed)

### Binary Search
```
left, right = 0, n-1
while left <= right:
    mid = (left + right) // 2
    if found: return
    if too_small: left = mid + 1
    else: right = mid - 1
```

### Two Pointers
```
Opposite: left=0, right=n-1, move based on condition
Same dir: slow=0, fast loops, slow marks valid position
```

### Sliding Window
```
Fixed: compute first K, then slide (add new, remove old)
Variable: expand right, shrink left when invalid
```

### BFS
```
queue = [start], visited = {start}
while queue:
    node = queue.pop(0)
    for neighbor: if not visited → add to both
```

### DFS
```
def dfs(node):
    visited.add(node)
    for neighbor: if not visited → dfs(neighbor)
```

### DP
```
1. Define dp[i] meaning
2. Base case
3. Recurrence relation
4. Fill order
5. Return answer
```

### Backtracking
```
def backtrack(path):
    if complete: save result
    for choice:
        add to path → recurse → remove from path
```

---

## 🆘 Interview Quick Tips

| Phase | Do This |
|-------|---------|
| Start | Clarify: constraints, edge cases, output format |
| Approach | "Brute force is O(n²), optimized is O(n) because..." |
| Stuck | "Let me think of brute force first" |
| Done | Test: given example → empty → single → edge case |

---

## ⚡ Common Edge Cases

```
Empty: [], "", null     Single: [x], "a"
Duplicates: all same    Sorted: already sorted
Negative: negative nums  Zero: contains 0
Max: n at constraint    Overflow: large numbers
```

---

*Full guide: `DSA_Problem_Solving_Roadmap.md`*
