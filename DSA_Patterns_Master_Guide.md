# The Anatomy of Algorithmic Problem Solving: A Master Guide

## 1. Introduction: The Structural Shift
The era of "brain teasers" (e.g., "How many piano tuners in Chicago?") is over. Modern technical interviews at FAANG and high-growth tech companies focus on **Data Structures and Algorithms (DSA)** to evaluate language-agnostic problem-solving skills.

Success isn't about memorizing solutions to thousands of LeetCode problems. It's about recognizing and applying **Algorithmic Patterns**. Approximately **87% of interview questions** are variations of roughly **15-20 core patterns**.

This guide provides a comprehensive analysis of these patterns, their mechanics, and the "physics" of algorithm design.

---

## 2. The Physics of Algorithms: Complexity Analysis

A frequent failure mode is selecting an algorithm without considering input constraints. The constraints are not just boundary conditions; they are **direct hints** about the required time complexity.

### 2.1 The $10^8$ Operations Rule
A standard single-core processor can execute roughly **$10^8$ operations per second**.
- If your algorithm requires $10^{10}$ operations, it will **Time Limit Exceeded (TLE)** (usually > 1 sec).
- analyzing $N$ (input size) tells you the maximum allowable Time Complexity.

### 2.2 Complexity Mapping Framework

| Input Size ($N$) | Target Complexity | Feasible Algorithms | Common Patterns |
| :--- | :--- | :--- | :--- |
| $N \le 10$ | $O(N!)$ | Factorial | Permutations, Brute Force |
| $N \le 20$ | $O(2^N)$ | Exponential | Subsets, Bitmask DP |
| $N \le 100$ | $O(N^4)$ | Quartic | Dense DP |
| $N \le 500$ | $O(N^3)$ | Cubic | Floyd-Warshall, Matrix Mult |
| $N \le 2,000$ | $O(N^2)$ | Quadratic | Nested Loops, Bellman-Ford |
| **$N \le 10^5$** | **$O(N \log N)$** | **Linearithmic** | **Sorting, Heaps, Merge Sort** |
| $N \le 10^6$ | $O(N)$ | Linear | Hash Maps, Two Pointers, BFS/DFS |
| $N > 10^8$ | $O(\log N)$ | Logarithmic | Binary Search |
| Large | $O(1)$ | Constant | Math, Hash Map Lookup |

> [!IMPORTANT]
> **The $10^5$ Cliff**: This is the most common constraint on LeetCode. It **strictly forbids** $O(N^2)$ solutions. You must find an $O(N)$ or $O(N \log N)$ approach (e.g., Sorting, Two Pointers, Hash Maps).

---

## 3. Core Algorithmic Patterns

### 3.1 Sliding Window
**Mechanism**: maintain a dynamic "window" over a sequence to process data in $O(N)$ instead of nested loops.
- **Invariant**: $WindowSum_{new} = WindowSum_{old} - LeftElement + RightElement$

#### Variations
1.  **Fixed Size**: Window size $K$ is constant. Slide one by one.
2.  **Variable Size**: Expand `right` pointer to meet condition; shrink `left` pointer to optimize.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟢 Easy | [Max Sum Subarray of Size K](https://leetcode.com/problems/maximum-average-subarray-i/) | "Contiguous subarray of length k" |
| 🟡 Medium | [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | "Longest substring", "Unique" |
| 🟡 Medium | [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/) | "Replace k characters", "Longest" |
| 🔴 Hard | [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) | "Smallest substring containing all chars" |

---

### 3.2 Two Pointers
**Mechanism**: Use two pointers to iterate through the data, typically from different ends or at different speeds, to satisfy constraints in linear time.
- **Precondition**: Often requires the array to be **Sorted**.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟢 Easy | [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/) | "Compare ends moving inward" |
| 🟡 Medium | [Two Sum II (Sorted)](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) | "Sorted array", "Find pair" |
| 🟡 Medium | [3Sum](https://leetcode.com/problems/3sum/) | "Unique triplets sum to zero" |
| 🟡 Medium | [Container With Most Water](https://leetcode.com/problems/container-with-most-water/) | "Max area", "Lines" |
| 🔴 Hard | [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) | "Elevation map", "Water trapped" |

---

### 3.3 Fast & Slow Pointers (Tortoise and Hare)
**Mechanism**: Move one pointer at speed 1 (`slow`) and another at speed 2 (`fast`).
- **Use Case**: Cycle detection, finding the middle of a list, finding cycle start.
- **Math**: If they meet, a cycle exists. Distance from starts behaves predictably.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟢 Easy | [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) | "Detect cycle in LL" |
| 🟢 Easy | [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/) | "Find middle node" |
| 🟡 Medium | [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/) | "Return start of cycle" |
| 🟡 Medium | [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) | "Array as LL", "One duplicate" |
| 🟢 Easy | [Happy Number](https://leetcode.com/problems/happy-number/) | "Sum of squares cycle" |

---

### 3.4 Merge Intervals
**Mechanism**: Sort intervals by start time, then iterate to merge overlapping intervals.
- **Key Logic**: If `current.start < previous.end`, they overlap. Merge by taking `max(previous.end, current.end)`.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟡 Medium | [Merge Intervals](https://leetcode.com/problems/merge-intervals/) | "Overlapping intervals" |
| 🟡 Medium | [Insert Interval](https://leetcode.com/problems/insert-interval/) | "Insert and merge" |
| 🟡 Medium | [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) | "Min removal to make valid" |
| 🟡 Medium | [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) (Premium/LintCode) | "Min conference rooms" |

---

### 3.5 Cyclic Sort
**Mechanism**: When numbers are in range `1 to N` (or `0 to N`), use values as indices to place numbers in their correct spots ($O(N)$ time, $O(1)$ space).
- **Swap Condition**: `nums[i] != nums[nums[i] - 1]`

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟢 Easy | [Missing Number](https://leetcode.com/problems/missing-number/) | "Range [0, n]", "Missing one" |
| 🟡 Medium | [Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/) | "Range [1, n]", "Extract duplicates" |
| 🔴 Hard | [First Missing Positive](https://leetcode.com/problems/first-missing-positive/) | "Smallest missing positive integer" |

---

### 3.6 In-place Reversal of Linked List
**Mechanism**: Manipulate `next` pointers to change direction without using extra memory. common for reversing sub-parts of a list.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟢 Easy | [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) | "Reverse whole list" |
| 🟡 Medium | [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/) | "Reverse from m to n" |
| 🔴 Hard | [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/) | "Reverse every k nodes" |

---

### 3.7 Tree Width First Search (BFS)
**Mechanism**: Level-by-level traversal using a **Queue**.
- **Template**: `while queue: count = len(queue); for _ in range(count): process()`

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟡 Medium | [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/) | "Level by level list" |
| 🟡 Medium | [Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/) | "Left to right then right to left" |
| 🟢 Easy | [Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/) | "Nearest leaf node" |
| 🟡 Medium | [Populating Next Right Pointers](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/) | "Connect nodes at same level" |

---

### 3.8 Tree Depth First Search (DFS)
**Mechanism**: Go deep before going wide. Uses Recursion or Stack.
- **Use Case**: Path finding, exhaustive types, "Root to Leaf" problems.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟢 Easy | [Path Sum](https://leetcode.com/problems/path-sum/) | "Root to leaf sum" |
| 🟡 Medium | [Path Sum II](https://leetcode.com/problems/path-sum-ii/) | "All paths with sum" |
| 🟡 Medium | [Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) | "Common ancestor" |
| 🔴 Hard | [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | "Max path sum anywhere" |

---

### 3.9 Two Heaps
**Mechanism**: Maintain two heaps (Min-Heap and Max-Heap) to track the **Median** or separate data into two halves.
- **Balance**: Ensure size difference allows clear identification of median.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🔴 Hard | [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) | "Median", "Stream" |
| 🔴 Hard | [Sliding Window Median](https://leetcode.com/problems/sliding-window-median/) | "Median in window" |
| 🔴 Hard | [IPO](https://leetcode.com/problems/ipo/) | "Max capital", "Projects" |

---

### 3.10 Top 'K' Elements
**Mechanism**: Use a **Min-Heap** of size $K$ to keep the $K$ largest elements (pop smallest when size > K). Or **Max-Heap** for smallest.
- **Alt**: QuickSelect (Hoare's) for $O(N)$ average.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟡 Medium | [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) | "Kth largest" |
| 🟡 Medium | [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) | "Frequency map", "Top K" |
| 🟡 Medium | [K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/) | "Distance", "Closest K" |

---

### 3.11 Subsets & Permutations (Backtracking)
**Mechanism**: Exhaustive search using state-space tree.
- **Template**: `if goal: add; for choice: choose -> backtrack -> unchoose`.
- **Pruning**: Return early if current path is invalid.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟡 Medium | [Subsets](https://leetcode.com/problems/subsets/) | "All subsets/combinations" |
| 🟡 Medium | [Permutations](https://leetcode.com/problems/permutations/) | "All orderings" |
| 🟡 Medium | [Combination Sum](https://leetcode.com/problems/combination-sum/) | "Sum to target", "Reuse allowed" |
| 🔴 Hard | [N-Queens](https://leetcode.com/problems/n-queens/) | "Non-attacking placement" |

---

### 3.12 Modified Binary Search
**Mechanism**: Applying Binary Search logic to non-traditional sorted structures (e.g., rotated arrays).
- **Key**: Determine which half is sorted to decide where to move pointers.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟡 Medium | [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) | "Rotated", "Sorted", "O(log N)" |
| 🟡 Medium | [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) | "Analysis of pivot" |
| 🟡 Medium | [Find Peak Element](https://leetcode.com/problems/find-peak-element/) | "Local maximum", "Log N" |

---

### 3.13 Bitwise XOR
**Mechanism**: Use properties of XOR: `A ^ A = 0`, `A ^ 0 = A`.
- **Use Case**: Finding missing unique numbers, cancelling duplicates.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟢 Easy | [Single Number](https://leetcode.com/problems/single-number/) | "One appears once, others twice" |
| 🟡 Medium | [Single Number III](https://leetcode.com/problems/single-number-iii/) | "Two numbers appear once" |
| 🟢 Easy | [Adding Two Negabinary Numbers](https://leetcode.com/problems/adding-two-negabinary-numbers/) (Niche) or [Counting Bits](https://leetcode.com/problems/counting-bits/) | "Bit manipulation" |

---

### 3.14 Topological Sort (Graph)
**Mechanism**: Ordering nodes in a Directed Acyclic Graph (DAG) such that for every edge $u \to v$, $u$ comes before $v$.
- **Algos**: Kahn's (Indegree BFS) or DFS Post-Order Reverse.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟡 Medium | [Course Schedule](https://leetcode.com/problems/course-schedule/) | "Prerequisites", "Can finish?" |
| 🟡 Medium | [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) | "Ordering of courses" |
| 🔴 Hard | [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/) (Premium) | "Order of characters", "Dependency" |

---

### 3.15 Monotonic Stack
**Mechanism**: Stack keeps elements in strictly increasing/decreasing order.
- **Use Case**: "Next Greater Element", "Previous Smaller Element", Histogram areas.

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟢 Easy | [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/) | "Next larger value" |
| 🟡 Medium | [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/) | "Days until warmer" |
| 🔴 Hard | [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) | "Area", "Bars" |
| 🔴 Hard | [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) | "Trapped water" (Stack approach) |

---

### 3.16 Union Find (Disjoint Set)
**Mechanism**: Efficiently manage disjoint sets.
- **Ops**: `find(x)` (with path compression), `union(x, y)` (by rank/size).
- **Time**: $O(\alpha(N))$ (nearly constant).

#### Practice Problems
| Difficulty | Problem | Signal |
| :--- | :--- | :--- |
| 🟡 Medium | [Number of Provinces](https://leetcode.com/problems/number-of-provinces/) | "Connected cities" |
| 🟡 Medium | [Redundant Connection](https://leetcode.com/problems/redundant-connection/) | "Cycle in tree" |
| 🟡 Medium | [Accounts Merge](https://leetcode.com/problems/accounts-merge/) | "Merge emails" |
| 🔴 Hard | [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) | "Consecutive elements", "Unsorted" |

---

### 3.17 Dynamic Programming (DP)
**Mechanism**: Break down problem -> Subproblems -> Store results (Memoization/Tabulation).
- **Patterns**:
    - **0/1 Knapsack**: Include or Exclude.
    - **Unbounded Knapsack**: Include multiple times.
    - **LCS**: Longest Common Subsequence (2 strings).
    - **Palindromes**: Expand from center or DP table.

#### Practice Problems
| Difficulty | Problem | Pattern |
| :--- | :--- | :--- |
| 🟡 Medium | [Coin Change](https://leetcode.com/problems/coin-change/) | Unbounded Knapsack |
| 🟡 Medium | [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/) | LCS |
| 🟡 Medium | [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/) | Palindrome |
| 🟡 Medium | [Word Break](https://leetcode.com/problems/word-break/) | Decision/Split |
| 🔴 Hard | [Edit Distance](https://leetcode.com/problems/edit-distance/) | 2D DP String |

---

## 4. Conclusion

Mastering these patterns allows you to see the "Matrix" of interview questions. When you see a new problem:
1.  **Check Constraints** ($N \le 10^5 \implies O(N)$ or $O(N \log N)$).
2.  **Identify Keywords** ("substring", "shortest path", "top K").
3.  **Map to Pattern**.
4.  **Adapt Template**.

*This guide consolidates the "Physics of Algorithms" and structural patterns for elite technical interview preparation.*
