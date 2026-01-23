# LeetCode DSA Question Types & Approaches Reference Guide

A comprehensive reference guide organized by DSA pattern, question types, and solution approaches with complexity analysis. Designed for Python implementation.

---

## 1. ARRAY & STRING PROBLEMS

### Question Types
- Subarray/substring operations (max, min, sum, product)
- Array manipulation (rotation, partition, reversal)
- String operations (anagrams, palindromes, pattern matching)
- Duplicate detection and removal
- Element search and sorting-related problems

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **HashMap/HashSet Lookup** | O(n) | O(n) | `seen = set(); for x in nums: seen.add(x)` | Duplicate detection, unique elements |
| **Two Pointers (Sorted)** | O(n) | O(1) | `l, r = 0, len-1; while l < r: ...` | Pairs, palindromes, containers |
| **Sliding Window (Fixed)** | O(n) | O(1) or O(k) | `for i in range(len - k + 1): window = nums[i:i+k]` | Max/min subarray of size k |
| **Sliding Window (Dynamic)** | O(n) | O(1) or O(m) | `while right < n: ...; while condition: left += 1` | Longest substring without repeats |
| **Prefix Sum** | O(n) | O(n) | `prefix = [0]; for x in nums: prefix.append(prefix[-1]+x)` | Range sum queries |
| **Sorting** | O(n log n) | O(1) or O(n) | `nums.sort()` | Sorted array requirements |
| **Brute Force (Nested Loops)** | O(n²) | O(1) | `for i in range(n): for j in range(i+1, n): ...` | Small inputs, interview edge cases |

---

## 2. TWO POINTERS PATTERN

### Question Types
- Array pair problems (sum, product, differences)
- Linked list operations (cycle detection, middle finding)
- Container/volume problems
- String validation (palindromes, reversals)
- Collision/meeting point problems

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **Two Pointers (Opposite Ends)** | O(n) | O(1) | `l, r = 0, n-1; while l < r: if arr[l] + arr[r] == target: ...` | Sorted arrays, pairs, palindromes |
| **Fast & Slow Pointers** | O(n) | O(1) | `slow = fast = head; while fast: slow = slow.next; fast = fast.next.next` | Linked lists, cycle detection |
| **Sliding Window with 2 Pointers** | O(n) | O(1) | `for j in range(n): while condition: i += 1` | Variable window problems |
| **Nested Two Pointers** | O(n²) | O(1) | `for i in range(n): l, r = i+1, n-1; while l < r: ...` | 3sum, 4sum, complex constraints |

---

## 3. SLIDING WINDOW PATTERN

### Question Types
- Longest/shortest substring/subarray problems
- Contains duplicate problems
- Variable-size window problems
- Character/element frequency problems
- Anagram and pattern matching problems

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **Sliding Window (Fixed Size)** | O(n) | O(1) or O(k) | `max_sum = sum(nums[:k]); for i in range(1, n-k+1): ...` | Fixed window size |
| **Sliding Window (Dynamic)** | O(n) | O(m) | `while right < n: seen[nums[right]] += 1; while len(seen) > k: ...` | Longest substring, k distinct chars |
| **Sliding Window (Shrink)** | O(n) | O(1) | `while left < right and condition: left += 1` | Minimum length substring |
| **Brute Force (All Subarrays)** | O(n³) | O(1) | `for i in range(n): for j in range(i, n): for k in range(i, j): ...` | Verification only |

---

## 4. BINARY SEARCH PATTERN

### Question Types
- Sorted array search
- Boundary finding (leftmost, rightmost)
- Search in rotated arrays
- Guess and validation problems
- Peak finding
- Minimization/maximization in sorted space

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **Binary Search (Standard)** | O(log n) | O(1) | `l, r = 0, n-1; while l <= r: mid = (l+r)//2; ...` | Direct search in sorted array |
| **Binary Search (Left Boundary)** | O(log n) | O(1) | `while l < r: mid = l + (r-l)//2; if arr[mid] < target: l = mid+1 else: r = mid` | First occurrence |
| **Binary Search (Right Boundary)** | O(log n) | O(1) | `while l < r: mid = r - (r-l)//2; if arr[mid] <= target: l = mid else: r = mid-1` | Last occurrence |
| **Binary Search on Answer** | O(log n * f) | O(1) | `def valid(x): ...; l, r = 0, max_val; while l < r: ...` | Minimize/maximize with constraint |
| **Linear Search** | O(n) | O(1) | `for i in range(n): if arr[i] == target: return i` | Unsorted array |

---

## 5. FAST & SLOW POINTERS (LINKED LIST)

### Question Types
- Cycle detection and finding cycle start
- Find middle of linked list
- Linked list reversal in place
- Remove duplicates
- Palindrome checking in linked lists

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **Fast & Slow (Floyd's Cycle)** | O(n) | O(1) | `slow = fast = head; while fast and fast.next: slow = slow.next; fast = fast.next.next` | Cycle detection |
| **Fast & Slow (Find Middle)** | O(n) | O(1) | `slow = fast = head; while fast and fast.next: slow = slow.next; fast = fast.next.next` | List middle, palindrome |
| **Reverse In-Place** | O(n) | O(1) | `prev = None; while node: next = node.next; node.next = prev; prev, node = node, next` | Linked list reversal |
| **HashSet for Cycles** | O(n) | O(n) | `seen = set(); curr = head; while curr: if curr in seen: return True; seen.add(curr); curr = curr.next` | Cycle detection (extra space) |

---

## 6. DFS (DEPTH-FIRST SEARCH) PATTERN

### Question Types
- Tree traversal (preorder, inorder, postorder)
- Graph path finding and connected components
- Backtracking problems (permutations, combinations, subsets)
- Topological sorting
- All paths and deep exploration problems

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **Recursive DFS** | O(v+e) | O(h) * | `def dfs(node): if not node: return; dfs(node.left); dfs(node.right)` | Trees, graphs with recursion |
| **Iterative DFS (Stack)** | O(v+e) | O(h) | `stack = [root]; while stack: node = stack.pop(); for child in node.children: stack.append(child)` | Trees, graphs iteratively |
| **DFS + Backtracking** | O(n!) | O(n) | `def backtrack(path): if done: result.append(path); return; for choice in choices: path.append(choice); backtrack(path); path.pop()` | Permutations, combinations, subsets |
| **BFS (Alternative)** | O(v+e) | O(w) ** | `from collections import deque; queue = deque([root]); ...` | Shortest path, level order |

*h = height of tree
**w = max width of tree

---

## 7. BFS (BREADTH-FIRST SEARCH) PATTERN

### Question Types
- Shortest path in unweighted graphs
- Level-order tree traversal
- Connected components finding
- Multi-source shortest path
- Bipartite checking
- Word ladder problems

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **BFS (Standard Queue)** | O(v+e) | O(v) | `from collections import deque; q = deque([start]); while q: node = q.popleft(); ...` | Shortest path unweighted |
| **BFS (Bidirectional)** | O(b^(d/2)) | O(b^(d/2)) | `q1, q2 = deque([start]), deque([end]); while q1 or q2: ...` | Meet in middle, large graphs |
| **DFS Alternative** | O(v+e) | O(h) | `def dfs(node): ...; dfs(node.neighbors)` | Same result, different space |
| **Dijkstra's (Weighted)** | O((v+e)log v) | O(v) | `import heapq; heap = [(0, start)]; ...` | Weighted shortest path |

---

## 8. DYNAMIC PROGRAMMING PATTERN

### Question Types
- Fibonacci and sequence problems
- Knapsack problems (0/1, unbounded)
- Longest subsequence/substring (LIS, LCS, LPS)
- Coin change and partition problems
- Grid path problems (unique paths, minimum cost)
- Matrix chain multiplication

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **DP Bottom-Up (Tabulation)** | O(n*m) | O(n*m) | `dp = [[0]*m for _ in range(n)]; dp[0][0] = 1; for i in range(n): for j in range(m): dp[i][j] = ...` | Most DP problems |
| **DP Top-Down (Memoization)** | O(n*m) | O(n*m) | `memo = {}; def dp(i): if i in memo: return memo[i]; memo[i] = ...; return memo[i]` | Recursive thinking |
| **DP Space-Optimized** | O(n*m) | O(m) or O(n) | `prev = [0]*m; for i in range(n): curr = [0]*m; for j in range(m): curr[j] = ...` | Large input, memory constraint |
| **Greedy (Wrong)** | Variable | Variable | `sort by some criteria; greedy choice` | Only when provably optimal |
| **Brute Force Recursion** | O(2^n) | O(n) | `def fib(n): if n <= 1: return n; return fib(n-1) + fib(n-2)` | Verification, small n only |

---

## 9. GREEDY ALGORITHM PATTERN

### Question Types
- Activity selection and interval problems
- Huffman coding and encoding problems
- Minimum spanning tree variations
- Gas station and jump game problems
- Meeting room scheduling
- Assign cookies/distribution problems

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **Greedy (Sorting)** | O(n log n) | O(1) or O(n) | `events.sort(); for event in events: if can_take(event): take(event)` | Interval, activity selection |
| **Greedy (Max Heap)** | O(n log n) | O(n) | `heap = [-x for x in nums]; heapq.heapify(heap); while heap: ...` | K largest, meetings |
| **Greedy (Min Heap)** | O(n log n) | O(n) | `heap = nums.copy(); heapq.heapify(heap); while heap: ...` | K smallest, Huffman |
| **Greedy (Single Pass)** | O(n) | O(1) | `max_reach = 0; for i in range(n): if i > max_reach: return False; max_reach = max(max_reach, i+nums[i])` | Jump game, gas station |
| **Brute Force (Backtracking)** | O(n!) | O(n) | All combinations check | Verification only |

---

## 10. BACKTRACKING PATTERN

### Question Types
- Permutations and combinations
- Subset generation (power set)
- Combination sum problems
- N-Queens and Sudoku solver
- Word search and palindrome partitioning
- Letter combinations and digit sequences

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **Backtracking (DFS)** | O(n!) or O(2^n) | O(n) | `def backtrack(path, remaining): if condition: result.append(path); return; for choice in remaining: path.append(choice); backtrack(path, remaining-choice); path.pop()` | Permutations, combinations |
| **Backtracking + Pruning** | O(2^n) | O(n) | `if pruning_condition: return; ...` | Optimize backtracking |
| **Backtracking (Iterative)** | O(n!) | O(n) | `stack = [(path, remaining)]; while stack: ...` | Avoid recursion depth |
| **Brute Force Iteration** | O(n!) | O(n) | `itertools.permutations(nums)` | Simple generation |

---

## 11. GRAPH PATTERN (GENERAL)

### Question Types
- Connected components and islands
- Graph coloring and bipartite checking
- Topological sorting (directed acyclic graphs)
- Shortest path variants
- Union-Find/Disjoint Set Union problems
- Graph construction and adjacency

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **Union-Find (Disjoint Set)** | O(α(n)) ≈ O(1) | O(n) | `parent = list(range(n)); def find(x): if parent[x] != x: parent[x] = find(parent[x]); return parent[x]` | Connected components, cycles |
| **DFS Graph Traversal** | O(v+e) | O(v) | `def dfs(node, graph): visited.add(node); for neighbor in graph[node]: if neighbor not in visited: dfs(neighbor)` | All connectivity, paths |
| **BFS Graph Traversal** | O(v+e) | O(v) | `queue = deque([start]); visited.add(start); while queue: ...` | Shortest paths, levels |
| **Topological Sort (DFS)** | O(v+e) | O(v) | `def dfs(node): visited.add(node); for neighbor in graph[node]: if neighbor not in visited: dfs(neighbor); stack.append(node)` | Dependency ordering |
| **Topological Sort (BFS/Kahn's)** | O(v+e) | O(v) | `in_degree = [0]*v; queue = deque([i for i in range(v) if in_degree[i]==0]); ...` | Alternative topological sort |
| **Brute Force (All Paths)** | O(v * 2^v) | O(v) | Explicit enumeration | Verification only |

---

## 12. INTERVAL PATTERN

### Question Types
- Merge overlapping intervals
- Insert interval into list
- Interval intersection and union
- Meeting rooms and availability
- Calendar scheduling problems

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **Interval Merge (Sort)** | O(n log n) | O(1) or O(n) | `intervals.sort(); result = [intervals[0]]; for start, end in intervals[1:]: if result[-1][1] < start: result.append([start,end]) else: result[-1][1] = max(result[-1][1], end)` | Merge overlapping |
| **Interval Insert (No Sort)** | O(n) | O(n) | Three cases: before, overlap, after | Insert into sorted list |
| **Two Pointers (Intersection)** | O(n + m) | O(1) | `i = j = 0; while i < len(a) and j < len(b): ...` | Intersection of two lists |
| **Brute Force** | O(n²) | O(n) | Compare each pair | Small inputs |

---

## 13. MONOTONIC STACK/QUEUE PATTERN

### Question Types
- Next greater/smaller element
- Largest rectangle in histogram
- Trapping rain water
- Sliding window maximum/minimum
- Daily temperature problems

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **Monotonic Stack (Decreasing)** | O(n) | O(n) | `stack = []; for num in nums: while stack and stack[-1] < num: stack.pop(); stack.append(num)` | Next greater element |
| **Monotonic Stack (Increasing)** | O(n) | O(n) | `stack = []; for num in nums: while stack and stack[-1] > num: stack.pop(); stack.append(num)` | Next smaller element |
| **Monotonic Deque** | O(n) | O(k) | `from collections import deque; dq = deque(); ...` | Sliding window max/min |
| **Brute Force (Nested Loop)** | O(n²) | O(1) | `for i in range(n): for j in range(i+1, n): ...` | Verification |

---

## 14. HEAP/PRIORITY QUEUE PATTERN

### Question Types
- K-th largest/smallest element
- Top K frequent elements
- Merge K sorted lists
- Median of data stream
- Heap sort and reorganization problems

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **Min-Heap (K Largest)** | O(n log k) | O(k) | `import heapq; heap = nums[:k]; heapq.heapify(heap); for num in nums[k:]: if num > heap[0]: heapq.heapreplace(heap, num)` | K largest elements |
| **Max-Heap (K Smallest)** | O(n log k) | O(k) | `heap = [-x for x in nums[:k]]; for num in nums[k:]: if -num < -heap[0]: heapq.heapreplace(heap, -num)` | K smallest elements |
| **Heap Merge (K Lists)** | O(n log k) | O(k) | `heap = [(lst[0], i, 0) for i, lst in enumerate(lists)]; ...` | Merge K sorted lists |
| **Sorting** | O(n log n) | O(n) | `sorted(nums)` | Simple sort |
| **Quickselect** | O(n) avg | O(1) | Partition-based selection | K-th element (faster avg) |

---

## 15. HASH MAP/SET PATTERN

### Question Types
- Duplicate detection and removal
- Anagram and pattern matching
- Valid parentheses and bracket matching
- LRU Cache and frequency tracking
- Character frequency and word problems

### Approaches (Best → Worst Complexity)

| Approach | Time | Space | Python Example | Best For |
|----------|------|-------|-----------------|----------|
| **HashMap/Dict Lookup** | O(n) | O(n) | `count = {}; for x in nums: count[x] = count.get(x, 0) + 1` | Frequency, duplicates |
| **HashSet Lookup** | O(n) | O(n) | `seen = set(); for x in nums: if x in seen: return True; seen.add(x)` | Unique elements, presence |
| **HashMap + Sorting** | O(n log n) | O(n) | `sorted(count.items(), key=lambda x: x[1], reverse=True)` | K most frequent |
| **Counter (Collections)** | O(n) | O(n) | `from collections import Counter; c = Counter(nums); c.most_common(k)` | Frequency queries |
| **Brute Force** | O(n²) | O(1) | Nested loops | Small input only |

---

## COMPLEXITY CHEAT SHEET

### Time Complexities (Best → Worst)

```
O(1)       - Constant: Array index access, hash lookup
O(log n)   - Logarithmic: Binary search, balanced tree operations
O(n)       - Linear: Single loop, linear search, simple traversal
O(n log n) - Linearithmic: Merge sort, heap sort, quicksort (avg)
O(n²)      - Quadratic: Nested loops, bubble sort, insertion sort
O(n³)      - Cubic: Triple nested loops
O(2^n)     - Exponential: Subsets, power set generation
O(n!)      - Factorial: Permutations, all arrangements
```

### Space Complexities

```
O(1)       - Constant: No extra space, in-place operations
O(log n)   - Logarithmic: Recursion depth of balanced tree
O(n)       - Linear: HashMap, HashSet, arrays, lists
O(n²)      - Quadratic: 2D arrays, DP tables
O(2^n)     - Exponential: All subsets, recursive tree
```

---

## QUICK DECISION TREE

### What pattern should I use?

```
Is it about finding in SORTED ARRAY?
  └─ YES → Binary Search (O(log n))
  └─ NO  → Continue

Is it about CONTIGUOUS SUBARRAY/SUBSTRING?
  └─ YES → Sliding Window (O(n))
  └─ NO  → Continue

Is it about PAIRS in SORTED ARRAY?
  └─ YES → Two Pointers (O(n))
  └─ NO  → Continue

Is it about TREE or GRAPH?
  └─ YES → DFS or BFS (O(v+e))
  └─ NO  → Continue

Is it about PERMUTATIONS or COMBINATIONS?
  └─ YES → Backtracking (O(n!) or O(2^n))
  └─ NO  → Continue

Is it about OPTIMAL SUBSTRUCTURE / OVERLAPPING SUBPROBLEMS?
  └─ YES → Dynamic Programming (O(n*m))
  └─ NO  → Continue

Is it about GREEDY CHOICE / LOCAL OPTIMIZATION?
  └─ YES → Greedy Algorithm (O(n log n))
  └─ NO  → Continue

Is it about FINDING K ELEMENTS or TOP K?
  └─ YES → Heap/Priority Queue (O(n log k))
  └─ NO  → Continue

Is it about FREQUENCIES or COUNTING?
  └─ YES → HashMap/HashSet (O(n))
  └─ NO  → Continue

Is it about INTERVALS?
  └─ YES → Interval Merge/Sort (O(n log n))
  └─ NO  → Continue

Is it about NEXT GREATER/SMALLER?
  └─ YES → Monotonic Stack (O(n))
  └─ NO  → Brute Force or revisit problem
```

---

## PYTHON IMPLEMENTATION TEMPLATES

### Template 1: Two Pointers (Sorted Array)
```python
def two_pointers(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []
```

### Template 2: Sliding Window (Dynamic)
```python
def sliding_window(s, k):
    char_count = {}
    left = 0
    result = ""
    
    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        if len(result) < right - left + 1:
            result = s[left:right+1]
    
    return result
```

### Template 3: Binary Search
```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

### Template 4: DFS (Recursive)
```python
def dfs(node, visited, result):
    if node in visited:
        return
    visited.add(node)
    result.append(node.val)
    
    for neighbor in node.neighbors:
        dfs(neighbor, visited, result)
```

### Template 5: BFS (Iterative)
```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = {start}
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        for neighbor in node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result
```

### Template 6: DP (Bottom-Up)
```python
def dp_bottom_up(n):
    dp = [0] * (n + 1)
    dp[0], dp[1] = 0, 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

### Template 7: Backtracking
```python
def backtrack(current_path, remaining_choices, result):
    if not remaining_choices:  # Base case
        result.append(current_path[:])
        return
    
    for i, choice in enumerate(remaining_choices):
        current_path.append(choice)
        new_remaining = remaining_choices[:i] + remaining_choices[i+1:]
        backtrack(current_path, new_remaining, result)
        current_path.pop()
    
    return result
```

### Template 8: Union-Find
```python
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
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

### Template 9: Monotonic Stack
```python
def next_greater(nums):
    stack = []
    result = [-1] * len(nums)
    
    for i in range(len(nums) - 1, -1, -1):
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        if stack:
            result[i] = stack[-1]
        stack.append(nums[i])
    
    return result
```

### Template 10: Heap Operations
```python
import heapq

def k_largest(nums, k):
    # Min-heap of size k to track k largest
    heap = nums[:k]
    heapq.heapify(heap)
    
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    
    return sorted(heap, reverse=True)
```

---

## STUDY TIPS

1. **Recognize Patterns First**: Before jumping to code, identify which pattern the problem belongs to
2. **Verify Complexity**: Always verify your solution's time and space complexity
3. **Edge Cases**: Test with empty input, single element, duplicates, negative numbers
4. **Python Tricks**: 
   - Use `collections.Counter` for frequencies
   - Use `collections.deque` for BFS
   - Use `heapq` for heap operations
   - Use built-in `sorted()` and `.sort()`
5. **Practice**: Start with easy problems of each pattern, then move to medium/hard
6. **Optimize**: After getting a working solution, try to optimize space complexity

---

## PROBLEM DIFFICULTY PROGRESSION

**Easy Foundation** (Get these down first)
- Array basics
- String manipulation  
- Hash tables
- Linked list operations
- Binary search
- Sorting basics

**Medium Intermediate** (Pattern mastery)
- Sliding window
- Two pointers
- DFS/BFS fundamentals
- DP basics
- Graph basics

**Hard Advanced** (Complex combinations)
- Advanced DP
- Complex graph problems
- Backtracking with constraints
- Greedy with proof
- Multiple pattern combinations

---

**Last Updated**: January 2026
**Format**: Markdown Table Reference
**Language**: Python 3
**Use**: Quick lookup and study reference for LeetCode DSA preparation
