# 🧭 DSA Problem-Solving Roadmap

> A systematic approach to identify the right **Data Structure** and **Algorithm** for any problem.

---

## 📋 Table of Contents

1. [Phase 1: Problem Analysis](#phase-1-problem-analysis)
2. [Phase 2: Data Structure Decision Tree](#phase-2-data-structure-decision-tree)
3. [Phase 3: Algorithm Decision Tree](#phase-3-algorithm-decision-tree)
4. [Phase 4: Implementation Rulebook](#phase-4-implementation-rulebook)
5. [Quick Reference Cheat Sheets](#quick-reference-cheat-sheets)

---

## Phase 1: Problem Analysis

### 🔍 Step 1: Understand the Problem

Ask yourself these **5 Key Questions**:

| # | Question | Why It Matters |
|---|----------|----------------|
| 1 | What is the **input format**? (array, string, graph, tree?) | Determines base data structure |
| 2 | What is the **output format**? (single value, list, boolean?) | Defines what to return |
| 3 | What are the **constraints**? (n ≤ 10⁵, 10⁶, 10⁹?) | Determines time complexity needed |
| 4 | What **operations** are required? (search, insert, delete, sort?) | Narrows down DS choices |
| 5 | Are there any **patterns** in examples? (sorted, unique, pairs?) | Hints at the approach |

### 📊 Constraint Analysis → Time Complexity Guide

| n (Input Size) | Max Acceptable Complexity | Typical Approaches |
|----------------|---------------------------|-------------------|
| n ≤ 10 | O(n!) or O(2ⁿ) | Brute Force, Backtracking |
| n ≤ 20 | O(2ⁿ) | Bitmask DP, Subset Generation |
| n ≤ 100 | O(n³) | Triple nested loops, Floyd-Warshall |
| n ≤ 1,000 | O(n²) | Nested loops, Simple DP |
| n ≤ 10⁵ | O(n log n) | Sorting, Binary Search, Heap |
| n ≤ 10⁶ | O(n) | Linear scan, Two Pointers, Hashing |
| n ≤ 10⁹ | O(log n) or O(1) | Math, Binary Search on Answer |

---

## Phase 2: Data Structure Decision Tree

```
                           START HERE
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
            Need ORDERING?          Need FAST LOOKUP?
                    │                     │
           ┌────────┼────────┐            ▼
           ▼        ▼        ▼      ┌────────────┐
        LIFO?    FIFO?   PRIORITY?  │ Hash Table │
           │        │        │      │   (O(1))   │
           ▼        ▼        ▼      └────────────┘
        ┌────-┐  ┌─────┐  ┌──────┐
        │Stack│  │Queue│  │ Heap │
        └────-┘  └─────┘  └──────┘
                               │
        ┌──────────────────────┴───────────────────────┐
        ▼                                              ▼
   Need HIERARCHY?                              Need RELATIONSHIPS?
        │                                              │
   ┌────┴────┐                                   ┌─────┴─────┐
   ▼         ▼                                   ▼           ▼
Tree/BST   Trie                            Adjacency     Adjacency
(parent-   (prefix                          List          Matrix
 child)    search)                        (sparse)       (dense)
```

### 🔑 Data Structure Selection Guide

| Pattern/Requirement | Data Structure | When to Use |
|---------------------|----------------|-------------|
| Last In, First Out operations | **Stack** | Parentheses matching, DFS, Undo operations, Monotonic problems |
| First In, First Out operations | **Queue** | BFS, Level-order traversal, Scheduling |
| Need min/max quickly | **Heap (Priority Queue)** | K-th element, Merge K lists, Top K problems |
| Fast search, insert, delete | **Hash Map / Set** | Two Sum, Duplicates, Frequency counting |
| Sorted data + search | **Binary Search Tree / TreeMap** | Range queries, Ordered statistics |
| Prefix/substring operations | **Trie** | Autocomplete, Word search, Prefix matching |
| Range queries + updates | **Segment Tree / BIT** | Sum/min/max in ranges with updates |
| Union-Find operations | **Disjoint Set Union (DSU)** | Connected components, Cycle detection |
| Bidirectional iteration | **Doubly Linked List** | LRU Cache, Browser history |
| Hierarchy with single parent | **Tree** | File systems, Organizational charts |
| Complex relationships | **Graph** | Networks, Dependencies, Paths |

---

## Phase 3: Algorithm Decision Tree

### 🌳 Master Decision Flowchart

```
                              PROBLEM TYPE
                                   │
       ┌───────────┬───────────┬───┴───┬───────────┬───────────┐
       ▼           ▼           ▼       ▼           ▼           ▼
   SEARCHING   OPTIMIZATION  GRAPH  SEQUENCE   PATTERN     COUNTING
       │           │           │       │           │           │
       ▼           ▼           ▼       ▼           ▼           │
  ┌────────┐  ┌────────┐   ┌──────┐ ┌─────┐   ┌───────┐        │
  │Sorted? │  │Optimal │   │BFS/  │ │Array│   │String │        │
  │        │  │Substruc│   │DFS?  │ │/List│   │Match? │        │
  └───┬────┘  └───┬────┘   └──┬───┘ └──┬──┘   └───┬───┘        │
      │           │           │        │          │            │
   YES│NO      YES│NO     LEVEL│PATH   │          │            ▼
      ▼ ▼         ▼ ▼        ▼  ▼      ▼          ▼       ┌─────────┐
   Binary│    DP/Greedy│   BFS DFS  Sliding/   KMP/       │DP/Combo │
   Search│      │      │        │   TwoPtr   Rabin-Karp   │ Math    │
         ▼      │      ▼        ▼      │                  └─────────┘
      Linear    │   Backtrack          ▼
      Search    ▼                Monotonic
             Greedy              Stack/Deque
```

### 📌 Algorithm Pattern Recognition

#### 🔍 SEARCHING Problems

| Signal/Pattern | Algorithm | Example Problems |
|----------------|-----------|------------------|
| Sorted array/matrix | **Binary Search** | Search in rotated array, First/Last position |
| Find k-th element | **QuickSelect / Heap** | K-th largest, Median of stream |
| Search in ranges | **Binary Search on Answer** | Capacity to ship, Koko eating bananas |
| Need all combinations | **Backtracking** | Subsets, Permutations |

#### ⚡ OPTIMIZATION Problems

| Signal/Pattern | Algorithm | Example Problems |
|----------------|-----------|------------------|
| "Minimum/Maximum" + Overlapping subproblems | **Dynamic Programming** | Longest subsequence, Knapsack |
| Local choice leads to global optimal | **Greedy** | Activity selection, Huffman coding |
| "Can we achieve X?" (Yes/No) | **Binary Search on Answer** | Minimum max distance, Allocate books |
| Minimize/Maximize with constraints | **DP or Greedy** | Coin change, Job scheduling |

#### 🕸️ GRAPH Problems

| Signal/Pattern | Algorithm | Example Problems |
|----------------|-----------|------------------|
| Shortest path (unweighted) | **BFS** | Word ladder, 01 Matrix |
| Shortest path (weighted, non-negative) | **Dijkstra** | Network delay, Cheapest flights |
| Shortest path (negative weights) | **Bellman-Ford** | Arbitrage detection |
| All pairs shortest path | **Floyd-Warshall** | City travel times |
| Connectivity / Components | **DFS / Union-Find** | Number of islands, Accounts merge |
| Cycle detection | **DFS / Union-Find** | Course schedule, Detect cycle |
| Topological ordering | **Kahn's BFS / DFS** | Course order, Build order |
| Minimum spanning tree | **Kruskal / Prim** | Min cost to connect all points |

#### 📊 SEQUENCE/ARRAY Problems

| Signal/Pattern | Algorithm | Example Problems |
|----------------|-----------|------------------|
| Contiguous subarray | **Sliding Window** | Max sum subarray of size k |
| Variable-size window | **Two Pointers + Expand/Shrink** | Minimum window substring |
| Sorted array, find pair | **Two Pointers** | Two sum (sorted), Container with most water |
| Previous/Next greater/smaller | **Monotonic Stack** | Next greater element, Trapping rain water |
| Range min/max | **Monotonic Deque** | Sliding window maximum |
| Subsequence problems | **DP** | LCS, LIS, Edit distance |

#### 📝 STRING Problems

| Signal/Pattern | Algorithm | Example Problems |
|----------------|-----------|------------------|
| Pattern matching | **KMP / Rabin-Karp / Z-Algorithm** | Search pattern in text |
| Prefix-based operations | **Trie** | Autocomplete, Word search II |
| Palindrome queries | **Manacher / DP** | Longest palindrome substring |
| Anagram / Permutation | **Frequency Map + Sliding Window** | Find all anagrams |

---

## Phase 4: Implementation Rulebook

### 📘 Rule 1: Binary Search Template

```python
# Use when: Sorted data, minimize/maximize, search space can be halved

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # or left for insertion point

# Binary Search on Answer Template
def binary_search_on_answer(low, high, is_valid):
    """Find minimum value where is_valid returns True"""
    while low < high:
        mid = low + (high - low) // 2
        if is_valid(mid):
            high = mid  # answer could be mid or lower
        else:
            low = mid + 1
    return low
```

### 📘 Rule 2: Two Pointers Template

```python
# Use when: Sorted array, pair/triplet sum, palindrome check

# Opposite direction (start from both ends)
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

# Same direction (fast-slow pointer)
def remove_duplicates(arr):
    if not arr:
        return 0
    slow = 0
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]
    return slow + 1
```

### 📘 Rule 3: Sliding Window Template

```python
# Use when: Contiguous subarray/substring, fixed or variable size

# Fixed size window
def fixed_sliding_window(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]  # Slide: add new, remove old
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Variable size window (Expand + Shrink pattern)
def variable_sliding_window(s, target):
    """Find minimum window containing target"""
    from collections import Counter
    
    need = Counter(target)
    have = {}
    left = 0
    min_len = float('inf')
    formed = 0
    
    for right, char in enumerate(s):
        # Expand window
        have[char] = have.get(char, 0) + 1
        if char in need and have[char] == need[char]:
            formed += 1
        
        # Shrink window when valid
        while formed == len(need):
            min_len = min(min_len, right - left + 1)
            left_char = s[left]
            have[left_char] -= 1
            if left_char in need and have[left_char] < need[left_char]:
                formed -= 1
            left += 1
    
    return min_len if min_len != float('inf') else 0
```

### 📘 Rule 4: BFS Template (Shortest Path in Unweighted Graph)

```python
# Use when: Shortest path, level-order traversal, spreading problems

from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([(start, 0)])  # (node, distance)
    
    while queue:
        node, dist = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return visited

# BFS for Grid (4 directions)
def bfs_grid(grid, start_row, start_col):
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    visited = {(start_row, start_col)}
    queue = deque([(start_row, start_col, 0)])
    
    while queue:
        row, col, dist = queue.popleft()
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < rows and 0 <= new_col < cols 
                and (new_row, new_col) not in visited
                and grid[new_row][new_col] != '#'):  # Assuming '#' is obstacle
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, dist + 1))
```

### 📘 Rule 5: DFS Template (Explore All Paths)

```python
# Use when: Path finding, cycle detection, connected components

# Recursive DFS
def dfs_recursive(graph, node, visited):
    visited.add(node)
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

# Iterative DFS (using Stack)
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                stack.append(neighbor)
    
    return visited

# DFS with Path Tracking
def dfs_path(graph, start, end, path, all_paths):
    path.append(start)
    
    if start == end:
        all_paths.append(path.copy())
    else:
        for neighbor in graph[start]:
            if neighbor not in path:  # Avoid cycles
                dfs_path(graph, neighbor, end, path, all_paths)
    
    path.pop()  # Backtrack
```

### 📘 Rule 6: Dynamic Programming Template

```python
# Use when: Overlapping subproblems + Optimal substructure

# DP Framework: 5 Steps
# 1. Define state: dp[i] = meaning
# 2. Base case
# 3. Recurrence relation
# 4. Order of computation
# 5. Return answer

# 1D DP Example: Fibonacci / Climbing Stairs
def climb_stairs(n):
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# 2D DP Example: Longest Common Subsequence
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# Space Optimized DP (using 1D array for 2D problems)
def knapsack_optimized(weights, values, capacity):
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):  # Reverse to avoid reuse
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]
```

### 📘 Rule 7: Backtracking Template

```python
# Use when: Generate all possibilities, constraint satisfaction

def backtrack(candidates, path, result, start):
    # Base case: found valid solution
    if is_valid_solution(path):
        result.append(path.copy())
        return
    
    for i in range(start, len(candidates)):
        # Skip duplicates (if needed)
        if i > start and candidates[i] == candidates[i-1]:
            continue
        
        # Make choice
        path.append(candidates[i])
        
        # Recurse
        backtrack(candidates, path, result, i + 1)  # i+1 if no reuse, i if reuse allowed
        
        # Undo choice (backtrack)
        path.pop()

# Example: Subsets
def subsets(nums):
    result = []
    
    def backtrack(start, path):
        result.append(path.copy())  # Every path is a valid subset
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

### 📘 Rule 8: Monotonic Stack Template

```python
# Use when: Next/Previous greater/smaller element

def next_greater_element(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # Store indices
    
    for i in range(n):
        # Pop elements smaller than current
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    
    return result

def previous_smaller_element(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    
    for i in range(n):
        while stack and nums[stack[-1]] >= nums[i]:
            stack.pop()
        if stack:
            result[i] = nums[stack[-1]]
        stack.append(i)
    
    return result
```

### 📘 Rule 9: Union-Find (DSU) Template

```python
# Use when: Dynamic connectivity, grouping, cycle detection in undirected graph

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already connected
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

### 📘 Rule 10: Heap/Priority Queue Template

```python
import heapq

# Use when: K-th element, merge sorted lists, scheduling

# Min Heap (default in Python)
def k_smallest(nums, k):
    heapq.heapify(nums)  # O(n)
    return [heapq.heappop(nums) for _ in range(k)]

# Max Heap (negate values)
def k_largest(nums, k):
    max_heap = [-x for x in nums]
    heapq.heapify(max_heap)
    return [-heapq.heappop(max_heap) for _ in range(k)]

# Heap with custom objects
def merge_k_sorted_lists(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))  # (value, list_index, element_index)
    
    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result
```

---

## Quick Reference Cheat Sheets

### 🎯 Problem Keyword → Algorithm Mapping

| Keywords in Problem | Likely Algorithm |
|---------------------|------------------|
| "sorted array" | Binary Search |
| "minimum/maximum subarray" | Sliding Window / Kadane |
| "shortest path" | BFS (unweighted) / Dijkstra (weighted) |
| "all possibilities" / "combinations" | Backtracking |
| "longest/shortest subsequence" | Dynamic Programming |
| "connected components" | DFS / BFS / Union-Find |
| "top K" / "K-th largest" | Heap / QuickSelect |
| "anagram" / "permutation" | Hash Map + Sorting/Sliding Window |
| "parentheses matching" | Stack |
| "level by level" | BFS |
| "number of ways" | DP / Combinatorics |
| "detect cycle" | DFS / Union-Find |
| "prefix sum" | Prefix Sum Array |
| "in-place" | Two Pointers |

### ⏱️ Time Complexity Quick Reference

| Data Structure | Access | Search | Insert | Delete |
|----------------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1) | O(1) |
| Stack | O(n) | O(n) | O(1) | O(1) |
| Queue | O(n) | O(n) | O(1) | O(1) |
| Hash Table | — | O(1)* | O(1)* | O(1)* |
| BST (balanced) | — | O(log n) | O(log n) | O(log n) |
| Heap | — | O(n) | O(log n) | O(log n) |

*Average case, worst case O(n)

### 🧠 Common DP State Definitions

| Problem Type | State Definition |
|--------------|------------------|
| Linear sequence | `dp[i]` = answer for first i elements |
| Two sequences | `dp[i][j]` = answer for first i of seq1 and j of seq2 |
| Substring | `dp[i][j]` = answer for substring from i to j |
| Knapsack | `dp[i][w]` = answer using first i items with capacity w |
| Grid | `dp[i][j]` = answer to reach cell (i, j) |

---

## 🏁 The Complete Problem-Solving Checklist

- [ ] **Read** the problem twice
- [ ] **Identify** input/output format
- [ ] **Note** constraints (time complexity hint)
- [ ] **Draw** examples on paper
- [ ] **Recognize** the pattern (use decision trees above)
- [ ] **Choose** data structure
- [ ] **Choose** algorithm
- [ ] **Write** pseudocode first
- [ ] **Code** the solution
- [ ] **Test** with examples
- [ ] **Edge cases** (empty, single element, duplicates, large input)
- [ ] **Optimize** if needed

---

> 💡 **Pro Tip**: When stuck, ask yourself:  
> *"What would brute force look like? What repeated work am I doing that I can cache or skip?"*

---

## 🔧 Advanced Algorithm Templates

### 📘 Rule 11: Dijkstra's Algorithm (Weighted Shortest Path)

```python
import heapq

# Use when: Shortest path in weighted graph with non-negative edges

def dijkstra(graph, start, n):
    """
    graph: adjacency list {node: [(neighbor, weight), ...]}
    Returns: distances from start to all nodes
    """
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]  # (distance, node)
    
    while heap:
        d, node = heapq.heappop(heap)
        
        if d > dist[node]:  # Skip outdated entries
            continue
        
        for neighbor, weight in graph[node]:
            new_dist = dist[node] + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    
    return dist

# Example usage:
# graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}
# dijkstra(graph, 0, 4) -> [0, 3, 1, 4]
```

### 📘 Rule 12: Topological Sort (DAG Ordering)

```python
from collections import deque, defaultdict

# Use when: Task scheduling, course prerequisites, build order

# Kahn's Algorithm (BFS-based)
def topological_sort_bfs(n, edges):
    """
    n: number of nodes
    edges: list of (from, to) pairs
    Returns: topological order or empty if cycle exists
    """
    graph = defaultdict(list)
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else []  # Empty = cycle detected

# DFS-based Topological Sort
def topological_sort_dfs(n, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    
    visited = [0] * n  # 0: unvisited, 1: visiting, 2: visited
    result = []
    
    def dfs(node):
        if visited[node] == 1:  # Cycle detected
            return False
        if visited[node] == 2:
            return True
        
        visited[node] = 1
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        
        visited[node] = 2
        result.append(node)
        return True
    
    for i in range(n):
        if visited[i] == 0:
            if not dfs(i):
                return []  # Cycle detected
    
    return result[::-1]
```

### 📘 Rule 13: Trie (Prefix Tree)

```python
# Use when: Prefix search, autocomplete, word dictionary, word search

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # Optional: count words with this prefix

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True
    
    def search(self, word):
        """Returns True if word exists in trie"""
        node = self._find_node(word)
        return node is not None and node.is_end
    
    def starts_with(self, prefix):
        """Returns True if any word starts with prefix"""
        return self._find_node(prefix) is not None
    
    def count_prefix(self, prefix):
        """Returns count of words starting with prefix"""
        node = self._find_node(prefix)
        return node.count if node else 0
    
    def _find_node(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def get_all_words(self, prefix=""):
        """Returns all words starting with prefix"""
        words = []
        node = self._find_node(prefix)
        if node:
            self._dfs(node, prefix, words)
        return words
    
    def _dfs(self, node, path, words):
        if node.is_end:
            words.append(path)
        for char, child in node.children.items():
            self._dfs(child, path + char, words)
```

### 📘 Rule 14: Segment Tree (Range Queries)

```python
# Use when: Range sum/min/max queries with point updates

class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self._build(arr, 0, 0, self.n - 1)
    
    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2*node+1, start, mid)
            self._build(arr, 2*node+2, mid+1, end)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
    def update(self, idx, val):
        """Update arr[idx] = val"""
        self._update(0, 0, self.n-1, idx, val)
    
    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2*node+1, start, mid, idx, val)
            else:
                self._update(2*node+2, mid+1, end, idx, val)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
    def query(self, left, right):
        """Query sum in range [left, right]"""
        return self._query(0, 0, self.n-1, left, right)
    
    def _query(self, node, start, end, left, right):
        if right < start or left > end:
            return 0  # Out of range
        if left <= start and end <= right:
            return self.tree[node]  # Fully covered
        
        mid = (start + end) // 2
        left_sum = self._query(2*node+1, start, mid, left, right)
        right_sum = self._query(2*node+2, mid+1, end, left, right)
        return left_sum + right_sum
```

### 📘 Rule 15: Binary Indexed Tree / Fenwick Tree

```python
# Use when: Prefix sums with updates, simpler than Segment Tree

class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, idx, delta):
        """Add delta to arr[idx]"""
        idx += 1  # 1-indexed
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & (-idx)  # Add LSB
    
    def prefix_sum(self, idx):
        """Sum of arr[0..idx]"""
        idx += 1
        total = 0
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & (-idx)  # Remove LSB
        return total
    
    def range_sum(self, left, right):
        """Sum of arr[left..right]"""
        return self.prefix_sum(right) - (self.prefix_sum(left-1) if left > 0 else 0)
```

### 📘 Rule 16: Prefix Sum (1D and 2D)

```python
# Use when: Multiple range sum queries, subarray sum problems

# 1D Prefix Sum
def build_prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

def range_sum_1d(prefix, left, right):
    """Sum of arr[left..right] inclusive"""
    return prefix[right + 1] - prefix[left]

# 2D Prefix Sum
def build_prefix_sum_2d(matrix):
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            prefix[i][j] = (matrix[i-1][j-1] + prefix[i-1][j] 
                          + prefix[i][j-1] - prefix[i-1][j-1])
    return prefix

def range_sum_2d(prefix, r1, c1, r2, c2):
    """Sum of submatrix from (r1,c1) to (r2,c2) inclusive"""
    return (prefix[r2+1][c2+1] - prefix[r1][c2+1] 
          - prefix[r2+1][c1] + prefix[r1][c1])
```

---

## 📝 Practice Problems by Category

### 🔍 Binary Search

| Difficulty | Problem | Key Insight |
|------------|---------|-------------|
| Easy | Binary Search | Standard template |
| Easy | Search Insert Position | Return left pointer |
| Medium | Search in Rotated Sorted Array | Find pivot, then search |
| Medium | Find First and Last Position | Two binary searches |
| Medium | Koko Eating Bananas | Binary search on answer |
| Hard | Median of Two Sorted Arrays | Binary search on partition |

### 🎯 Two Pointers

| Difficulty | Problem | Key Insight |
|------------|---------|-------------|
| Easy | Two Sum II (Sorted) | Opposite pointers |
| Easy | Valid Palindrome | Compare from ends |
| Medium | 3Sum | Sort + fix one + two pointers |
| Medium | Container With Most Water | Move shorter height pointer |
| Medium | Remove Duplicates from Sorted Array II | Fast-slow pointers |

### 🪟 Sliding Window

| Difficulty | Problem | Key Insight |
|------------|---------|-------------|
| Easy | Maximum Average Subarray I | Fixed window |
| Medium | Longest Substring Without Repeating | Variable window + hashset |
| Medium | Minimum Size Subarray Sum | Shrink when valid |
| Medium | Permutation in String | Fixed window + frequency |
| Hard | Minimum Window Substring | Expand then shrink |
| Hard | Sliding Window Maximum | Monotonic deque |

### 🌳 Trees

| Difficulty | Problem | Key Insight |
|------------|---------|-------------|
| Easy | Maximum Depth of Binary Tree | Recursion or BFS |
| Easy | Invert Binary Tree | Swap left and right |
| Medium | Validate BST | Track valid range |
| Medium | Lowest Common Ancestor | Recursive path finding |
| Medium | Binary Tree Level Order Traversal | BFS with level tracking |
| Hard | Serialize and Deserialize Binary Tree | Preorder + null markers |

### 🕸️ Graphs

| Difficulty | Problem | Key Insight |
|------------|---------|-------------|
| Medium | Number of Islands | DFS/BFS flood fill |
| Medium | Clone Graph | BFS/DFS with hashmap |
| Medium | Course Schedule | Topological sort / cycle detection |
| Medium | Pacific Atlantic Water Flow | BFS from edges |
| Medium | Network Delay Time | Dijkstra |
| Hard | Word Ladder | BFS + preprocessing |
| Hard | Alien Dictionary | Topological sort |

### 📊 Dynamic Programming

| Difficulty | Problem | Key Insight |
|------------|---------|-------------|
| Easy | Climbing Stairs | dp[i] = dp[i-1] + dp[i-2] |
| Easy | House Robber | Take or skip pattern |
| Medium | Coin Change | Unbounded knapsack |
| Medium | Longest Increasing Subsequence | dp[i] = max(dp[j]+1) or Binary Search |
| Medium | Longest Common Subsequence | 2D DP |
| Medium | Word Break | dp[i] = can reach position i |
| Hard | Edit Distance | 2D DP with 3 operations |
| Hard | Regular Expression Matching | 2D DP with * handling |

### 📚 Stack

| Difficulty | Problem | Key Insight |
|------------|---------|-------------|
| Easy | Valid Parentheses | Match opening with closing |
| Medium | Daily Temperatures | Monotonic decreasing stack |
| Medium | Evaluate Reverse Polish Notation | Operands on stack |
| Hard | Largest Rectangle in Histogram | Monotonic stack + area |
| Hard | Trapping Rain Water | Monotonic stack or two pointers |

### ⚡ Heap / Priority Queue

| Difficulty | Problem | Key Insight |
|------------|---------|-------------|
| Easy | Kth Largest Element in Stream | Min heap of size k |
| Medium | Top K Frequent Elements | Heap or bucket sort |
| Medium | Kth Largest Element in Array | QuickSelect or heap |
| Medium | Task Scheduler | Max heap + cooldown |
| Hard | Merge K Sorted Lists | Min heap with (val, list_idx) |
| Hard | Find Median from Data Stream | Two heaps (max + min) |

### 🔗 Linked List

| Difficulty | Problem | Key Insight |
|------------|---------|-------------|
| Easy | Reverse Linked List | Iterative or recursive |
| Easy | Merge Two Sorted Lists | Dummy head technique |
| Medium | Add Two Numbers | Carry handling |
| Medium | Remove Nth Node From End | Two pointers with gap |
| Medium | Linked List Cycle II | Floyd's algorithm |
| Hard | Reverse Nodes in k-Group | Reverse k nodes at a time |

### 🔄 Backtracking

| Difficulty | Problem | Key Insight |
|------------|---------|-------------|
| Medium | Subsets | Include or exclude each element |
| Medium | Permutations | Swap elements |
| Medium | Combination Sum | Allow reuse with same index |
| Medium | Word Search | DFS with visited tracking |
| Hard | N-Queens | Column, diagonal checks |
| Hard | Sudoku Solver | Try 1-9 in empty cells |

---

## 🧩 Common Patterns & Tricks

### 1️⃣ Dummy Node Trick (Linked Lists)
```python
dummy = ListNode(0)
dummy.next = head
# Now you don't need special handling for head
return dummy.next
```

### 2️⃣ In-place Array Modification
```python
# Keep a "write pointer" for valid positions
write = 0
for read in range(len(arr)):
    if is_valid(arr[read]):
        arr[write] = arr[read]
        write += 1
```

### 3️⃣ Fast Power (Exponentiation by Squaring)
```python
def power(base, exp, mod=None):
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result = result * base
            if mod: result %= mod
        base = base * base
        if mod: base %= mod
        exp //= 2
    return result
```

### 4️⃣ GCD and LCM
```python
import math
gcd = math.gcd(a, b)
lcm = (a * b) // gcd
```

### 5️⃣ Bit Manipulation Tricks
```python
# Check if n is power of 2
is_power_of_2 = n > 0 and (n & (n - 1)) == 0

# Get lowest set bit
lowest_bit = n & (-n)

# Count set bits
count = bin(n).count('1')  # or n.bit_count() in Python 3.10+

# Clear lowest set bit
n = n & (n - 1)

# Check if i-th bit is set
is_set = (n >> i) & 1
```

### 6️⃣ Coordinate Compression
```python
def compress(arr):
    sorted_unique = sorted(set(arr))
    mapping = {v: i for i, v in enumerate(sorted_unique)}
    return [mapping[x] for x in arr]
```

### 7️⃣ Graph Representation
```python
# Adjacency List (most common)
from collections import defaultdict
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)  # For undirected

# Adjacency Matrix (dense graphs)
matrix = [[0] * n for _ in range(n)]
for u, v in edges:
    matrix[u][v] = 1
```

### 8️⃣ Sorting with Custom Key
```python
# Sort by multiple criteria
arr.sort(key=lambda x: (x[0], -x[1]))  # Ascending first, descending second

# Sort indices
indices = sorted(range(len(arr)), key=lambda i: arr[i])
```

---

## 🎓 Problem-Solving Mindset

### When You're Stuck:

1. **Simplify the problem** - Can you solve it for n=1, n=2?
2. **Draw it out** - Visualize with small examples
3. **Think backwards** - Start from the answer
4. **Look for patterns** - Is there repetition you can exploit?
5. **Brute force first** - Then optimize

### Red Flags That Suggest Specific Approaches:

| Red Flag | Likely Approach |
|----------|-----------------|
| "Minimum number of steps" | BFS |
| "All possible ways" | Backtracking or DP |
| "Maximum/Minimum with choices" | DP or Greedy |
| "Subarray" | Sliding Window or Prefix Sum |
| "Subsequence" | DP |
| "Sorted + search" | Binary Search |
| "Graph with cycles" | DFS with visited states |
| "Shortest path" | BFS (unweighted) or Dijkstra |
| "Connected components" | Union-Find or DFS |
| "K-th something" | Heap or QuickSelect |

---

## � Interview Success Guide

> **Master the process, not just the solution.** How you approach problems matters as much as solving them.

### 🎯 The REACTO Framework (Use for EVERY Problem)

| Step | What to Do | Time | What to Say |
|------|-----------|------|-------------|
| **R**epeat | Restate problem in your own words | 1 min | *"So we need to find... and return..."* |
| **E**xamples | Walk through examples, ask edge cases | 2-3 min | *"Let me trace through this example..."* |
| **A**pproach | Discuss 2-3 approaches with trade-offs | 3-5 min | *"I can think of two approaches..."* |
| **C**ode | Write clean, modular code | 15-20 min | Think aloud while coding |
| **T**est | Test with examples AND edge cases | 3-5 min | *"Let me verify with the example..."* |
| **O**ptimize | Discuss possible optimizations | 2-3 min | *"We could optimize by..."* |

---

### ❓ Clarifying Questions to ALWAYS Ask

**Ask these BEFORE writing any code:**

```
📋 INPUT QUESTIONS:
├── What are the constraints? (size, range?)
├── Can input be empty or null?
├── Are there duplicates? How to handle them?
├── Is the input sorted?
├── Can there be negative numbers?
└── What's the data type? (int, float, string?)

📋 OUTPUT QUESTIONS:
├── What to return if no answer exists? (-1, null, []?)
├── If multiple valid answers, return any or specific one?
├── Should I modify in-place or return new structure?
└── What format should the output be?

📋 CONSTRAINT QUESTIONS:
├── Optimize for time or space?
├── Can I use extra space?
└── Are there memory limitations?
```

---

### 💬 How to Communicate During the Interview

| Phase | What to Do | Example Phrases |
|-------|------------|-----------------|
| **Understanding** | Restate + Clarify | *"Let me make sure I understand..."* |
| **Thinking** | Think aloud | *"I'm considering using X because..."* |
| **Choosing** | Explain trade-offs | *"Approach A is O(n²) but O(1) space..."* |
| **Coding** | Narrate actions | *"I'll create a hashmap to store..."* |
| **Stuck** | Be transparent | *"I'm thinking about how to handle..."* |
| **Testing** | Walk through | *"Let me trace: input [1,2,3], first we..."* |

---

### 🔄 The Trade-off Discussion Interviewers Love

**Always present multiple approaches:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│ APPROACH 1: Brute Force                                                 │
│ • Time: O(n²)  Space: O(1)                                              │
│ • ✅ Simple to implement                                                │
│ • ❌ Too slow for large inputs                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ APPROACH 2: Hash Map                                                    │
│ • Time: O(n)   Space: O(n)                                              │
│ • ✅ Fast lookup                                                        │
│ • ❌ Uses extra space                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ DECISION: "Given n can be up to 10⁵, I'll use Approach 2 because       │
│ O(n²) would be too slow. The O(n) space is acceptable."                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 🆘 How to Handle Being Stuck

| Situation | What to Say | What to Do |
|-----------|------------|------------|
| **No idea** | *"Let me start with brute force"* | Think of the O(n²) or O(2ⁿ) approach first |
| **Partially stuck** | *"I see this is similar to X pattern"* | Connect to known patterns |
| **Hint given** | *"Ah, that helps! So I should..."* | Build on the hint gracefully |
| **Wrong path** | *"This won't work because... Let me try..."* | Pivot confidently, don't panic |
| **Completely lost** | *"Can you give me a hint on the approach?"* | It's OK to ask! Better than silence |

**Remember:** Interviewers want to see how you THINK, not just the answer.

---

### 🧪 Testing Strategy (Interviewers LOVE This)

**Always test with these in order:**

| Test Type | What to Check | Example |
|-----------|--------------|---------|
| 1. **Given Example** | Does your solution work? | The example from the problem |
| 2. **Empty/Null** | Edge case handling | `[]`, `null`, `""` |
| 3. **Single Element** | Boundary condition | `[5]`, `"a"` |
| 4. **Two Elements** | Simple case | `[1, 2]`, `"ab"` |
| 5. **All Same** | Duplicate handling | `[1, 1, 1, 1]` |
| 6. **Already Solved** | Already sorted, etc. | `[1, 2, 3, 4, 5]` |
| 7. **Worst Case** | Stress your solution | Maximum size, worst order |

**Say this:** *"Let me test with the given example first, then try an empty input and a single element..."*

---

### 🔄 Common Follow-up Questions to Expect

| Original Problem | Follow-up Variations |
|------------------|---------------------|
| Two Sum | → Three Sum, K Sum, Two Sum in BST |
| Reverse Linked List | → Reverse in groups of K, Reverse between positions |
| Binary Search | → Rotated array, First/Last position, Peak element |
| BFS (unweighted) | → Weighted edges (Dijkstra), Bi-directional BFS |
| Valid Parentheses | → Multiple bracket types, Wildcards, Longest valid |
| Merge Intervals | → Insert interval, Meeting rooms, Min platforms |
| LRU Cache | → LFU Cache, TTL expiration |
| Tree Traversal | → Iterative version, Without recursion |
| O(n) solution | → *"Can you do it in O(1) space?"* |
| O(n log n) solution | → *"Can you do it in O(n)?"* |

**Pro tip:** Think about these BEFORE the interview!

---

### 💻 Code Quality That Impresses

| ❌ Don't Write | ✅ Write Instead |
|----------------|------------------|
| `l, r, i, j` | `left, right, row, col` |
| `res, ans, tmp` | `result, maxSum, currentNode` |
| One giant function | Helper functions with clear names |
| No comments | Brief comments for key logic |
| Magic numbers `7`, `1000` | Constants `DAYS_IN_WEEK`, `MAX_SIZE` |
| Deeply nested code | Early returns, extracted functions |

**Example of clean code structure:**
```python
def solve(nums, target):
    # 1. Handle edge cases
    if not nums:
        return -1
    
    # 2. Initialize data structures
    seen = {}
    
    # 3. Main logic (with clear purpose)
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    # 4. Handle no solution case
    return -1
```

---

### 🏢 Company-Specific Patterns

| Company | Favorite Topics | Interview Style |
|---------|-----------------|-----------------|
| **Google** | Graphs, DP, sliding window, system design | Want multiple approaches discussed |
| **Meta (Facebook)** | Trees, graphs, strings, recursion | Expect optimal solution first try |
| **Amazon** | Arrays, hashmaps, trees, design patterns | Heavy on Leadership Principles |
| **Microsoft** | Trees, linked lists, classic problems | Step-by-step thinking valued |
| **Apple** | Clean code, edge cases, testing | Perfectionism matters |
| **Netflix** | Practical problems, system design | Senior-focused, ship mentality |
| **Startups** | Practical, debugging, end-to-end | Move fast, show ownership |

---

### ⏰ Time Management in 45-minute Interview

```
┌──────────────────────────────────────────────────────────────┐
│  0:00 - 0:03  │  Introduction                                │
├──────────────────────────────────────────────────────────────┤
│  0:03 - 0:08  │  Read problem, ask clarifying questions      │
├──────────────────────────────────────────────────────────────┤
│  0:08 - 0:13  │  Discuss approach, get approval              │
├──────────────────────────────────────────────────────────────┤
│  0:13 - 0:35  │  CODE! (22 minutes)                          │
├──────────────────────────────────────────────────────────────┤
│  0:35 - 0:40  │  Test and debug                              │
├──────────────────────────────────────────────────────────────┤
│  0:40 - 0:45  │  Complexity analysis, follow-ups, questions  │
└──────────────────────────────────────────────────────────────┘
```

---

### 🚫 Red Flags to Avoid

| ❌ Red Flag | ✅ What to Do Instead |
|-------------|----------------------|
| Jump straight to coding | Ask 2-3 clarifying questions first |
| Code in complete silence | Think aloud constantly |
| Say "I don't know" and freeze | Say "Let me think of brute force first" |
| Argue when given hints | Accept gracefully, build on them |
| Ignore edge cases | Proactively mention and handle them |
| Submit without testing | Dry run with at least one example |
| Panic when stuck | Take a breath, break down the problem |
| Over-engineer | Start simple, optimize if asked |

---

### ✨ Phrases That Impress Interviewers

```
BEFORE CODING:
✓ "Before I start coding, let me clarify a few things..."
✓ "Let me make sure I understand the problem correctly..."
✓ "What should I return if the input is empty?"

DURING APPROACH:
✓ "I can think of two approaches here..."
✓ "The trade-off between these approaches is..."
✓ "This reminds me of the [pattern name] pattern..."
✓ "Let me start with brute force, then optimize..."

WHILE CODING:
✓ "I'm creating a hashmap to store..."
✓ "This helper function will handle..."
✓ "Let me add a comment here to clarify..."

AFTER CODING:
✓ "Let me trace through with the example..."
✓ "Let me also test with an edge case..."
✓ "The time complexity is O(n) because..."
✓ "We could optimize space by..."
```

---

### 🎯 The Interview Day Checklist

**Before the Interview:**
- [ ] Review your resume projects (be ready to discuss)
- [ ] Practice explaining your thought process aloud
- [ ] Prepare 2-3 questions to ask the interviewer
- [ ] Test your setup (camera, mic, IDE) if remote
- [ ] Have water nearby

**During the Interview:**
- [ ] Smile and be personable
- [ ] Ask clarifying questions
- [ ] Think aloud throughout
- [ ] Test your solution
- [ ] Ask thoughtful questions at the end

**If Things Go Wrong:**
- [ ] Stay calm, don't panic
- [ ] It's OK to ask for hints
- [ ] One bad problem ≠ failed interview
- [ ] Learn from it for next time

---

## �📌 Final Tips

> 🔥 **Practice Pattern Recognition**: The more problems you solve, the faster you'll recognize patterns.

> ⏰ **Time Management**: In interviews, spend 5-10 mins understanding + planning before coding.

> 🧪 **Test Your Code**: Always trace through with the given examples before submitting.

> 📝 **Communicate**: In interviews, explain your thought process out loud.

> 🎯 **Start Simple**: Begin with brute force, then optimize. A working O(n²) is better than a broken O(n).

> 💡 **It's a Conversation**: The interview is a collaboration, not an interrogation. Show how you work with others.

---

*Last Updated: December 2024*

