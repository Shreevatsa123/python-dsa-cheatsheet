# PYTHON LEETCODE/INTERVIEW GUIDE (Pattern-Focused)

This guide focuses on common *patterns*, *templates*, and *problem-specific structures* used in LeetCode. It assumes familiarity with the core syntax from `README.md`.

## TABLE OF CONTENTS

1.  [Hash Map & Prefix Sum Patterns](#hash-map--prefix-sum-patterns)
    * [Hash Map Tracking](#hash-map-tracking)
    * [Prefix Sum](#prefix-sum)
2.  [Pointer Patterns](#pointer-patterns)
    * [Two Pointers](#two-pointers)
    * [Sliding Window](#sliding-window)
    * [Fast & Slow Pointers (Floyd's)](#fast--slow-pointers-floyds)
3.  [Stack & Queue Patterns](#stack--queue-patterns)
    * [Monotonic Stack](#monotonic-stack)
    * [BFS (Level Order)](#bfs-level-order)
4.  [Recursive & Graph Patterns](#recursive--graph-patterns)
    * [DFS (Depth-First Search)](#dfs-depth-first-search)
    * [Backtracking](#backtracking)
    * [Union Find (Disjoint Set)](#union-find-disjoint-set)
    * [Topological Sort (Kahn's Algo)](#topological-sort-kahns-algo)
    * [Dijkstra (Shortest Path)](#dijkstra-shortest-path)
5.  [Heaps & Intervals](#heaps--intervals)
    * [Top 'K' Elements](#top-k-elements)
    * [Merge Intervals](#merge-intervals)
6.  [Advanced Data Structures](#advanced-data-structures)
    * [Trie (Prefix Tree)](#trie-prefix-tree)
7.  [Dynamic Programming Patterns](#dynamic-programming-patterns)

---

## HASH MAP & PREFIX SUM PATTERNS

### Hash Map Tracking
**Concept:** Use a dictionary to store "seen" elements or indices to achieve O(1) lookups.
**Use when:** "Find a pair that...", "Check if seen before...", "Frequency counting".

```python
# PATTERN: Two Sum
# Find two numbers that add up to target
def twoSum(nums, target):
    seen = {} # val -> index
    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [seen[diff], i]
        seen[num] = i
    return []

# PATTERN: Group Anagrams
# Key Idea: Sorted string or character count tuple as key
from collections import defaultdict
def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        # Key can be sorted string: tuple(sorted(s))
        # Or char count: (0, 1, 0, ..., 2) for 26 chars
        key = tuple(sorted(s)) 
        groups[key].append(s)
    return list(groups.values())

```

### Prefix Sum

**Concept:** Precompute cumulative sums to calculate subarray sums in O(1).
**Use when:** "Subarray sum equals K", "Range Sum queries".

```python
# Basic Prefix Sum Building
nums = [1, 2, 3, 4]
prefix = [0] * (len(nums) + 1)
for i in range(len(nums)):
    prefix[i + 1] = prefix[i] + nums[i]
# prefix is [0, 1, 3, 6, 10]
# Sum of nums[i..j] = prefix[j+1] - prefix[i]

# PATTERN: Subarray Sum Equals K
# Find count of continuous subarrays sum = k
def subarraySum(nums, k):
    count = 0
    curr_sum = 0
    # Map {prefix_sum : frequency}
    prefix_map = {0: 1} # Base case: sum 0 exists once (empty array)
    
    for num in nums:
        curr_sum += num
        # If (curr_sum - k) exists in map, it means the subarray 
        # between that previous point and now adds up to k.
        if (curr_sum - k) in prefix_map:
            count += prefix_map[curr_sum - k]
        
        prefix_map[curr_sum] = prefix_map.get(curr_sum, 0) + 1
    return count

```

---

## POINTER PATTERNS

### Two Pointers

**Use when:** Sorted arrays (finding pairs), reversing, partitioning.

```python
# Standard: Move from ends inward
left, right = 0, len(arr) - 1
while left < right:
    if condition(left, right):
        return True
    left += 1
    right -= 1

# Merge Two Sorted Arrays
p1, p2 = 0, 0
while p1 < len(arr1) and p2 < len(arr2):
    if arr1[p1] < arr2[p2]:
        res.append(arr1[p1])
        p1 += 1
    else:
        res.append(arr2[p2])
        p2 += 1
# Append remaining elements...

```

### Sliding Window

**Use when:** Finding optimal contiguous subarray (longest, shortest, max sum).

```python
# Template: Variable size window
window = {}
left = 0
ans = 0

for right in range(len(arr)):
    # 1. Expand window
    char = arr[right]
    window[char] = window.get(char, 0) + 1
    
    # 2. Shrink window if invalid
    while invalid_condition(window):
        remove_char = arr[left]
        window[remove_char] -= 1
        if window[remove_char] == 0:
            del window[remove_char]
        left += 1
    
    # 3. Update answer (max length)
    ans = max(ans, right - left + 1)

```

### Fast & Slow Pointers (Floyd's)

**Use when:** Cycle detection (Linked Lists, Array jumping), Finding middle.

```python
# Detect Cycle
slow, fast = head, head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        return True # Cycle found

```

---

## STACK & QUEUE PATTERNS

### Monotonic Stack

**Concept:** Keep stack sorted (increasing/decreasing).
**Use when:** "Next Greater Element", "Next Smaller Element", "Daily Temperatures".

```python
# PATTERN: Next Greater Element
# Find the next greater number for every element
def nextGreaterElements(nums):
    stack = [] # Stores indices
    res = [-1] * len(nums)
    
    for i in range(len(nums)):
        # While current num is greater than stack top, 
        # we found the "next greater" for the stack top
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            res[idx] = nums[i]
        stack.append(i)
    return res

```

### BFS (Level Order)

**Use when:** Shortest path in unweighted graph, Level-by-level tree process.

```python
from collections import deque

def bfs(start_node):
    q = deque([start_node])
    visited = {start_node}
    steps = 0
    
    while q:
        size = len(q)
        for _ in range(size):
            node = q.popleft()
            if node == target:
                return steps
            
            for neighbor in node.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        steps += 1
    return -1

```

---

## RECURSIVE & GRAPH PATTERNS

### DFS (Depth-First Search)

**Use when:** Exploring all paths, checking connectivity, Backtracking.

```python
# Recursive Template
def dfs(node, visited):
    if not node or node in visited:
        return
    visited.add(node)
    for neighbor in node.neighbors:
        dfs(neighbor, visited)

# Iterative Template (using Stack)
stack = [start_node]
visited = {start_node}
while stack:
    node = stack.pop()
    for neighbor in node.neighbors:
        if neighbor not in visited:
            visited.add(neighbor)
            stack.append(neighbor)

```

### Union Find (Disjoint Set)

**Concept:** efficiently groups elements and checks connectivity.
**Use when:** "Count connected components", "Cycle detection in undirected graph", "Kruskal's Algorithm".

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
        self.count = n # Number of connected components
    
    def find(self, p):
        if self.parent[p] != p:
            # Path compression: point directly to root
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]
    
    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            # Union by rank: attach smaller tree to larger
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1
            self.count -= 1
            return True
        return False

```

### Topological Sort (Kahn's Algo)

**Use when:** "Course Schedule", "Order of dependencies", "Detect cycle in Directed Graph".

```python
from collections import deque, defaultdict

def topologicalSort(numCourses, prerequisites):
    adj = defaultdict(list)
    in_degree = [0] * numCourses
    
    # 1. Build Graph
    for dest, src in prerequisites:
        adj[src].append(dest)
        in_degree[dest] += 1
        
    # 2. Add 0-in-degree nodes to Queue
    q = deque([i for i in range(numCourses) if in_degree[i] == 0])
    order = []
    
    # 3. Process
    while q:
        node = q.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                q.append(neighbor)
                
    return order if len(order) == numCourses else [] # Empty if cycle

```

### Dijkstra (Shortest Path)

**Use when:** Shortest path in **weighted** graph (non-negative weights).

```python
import heapq

def dijkstra(n, edges, start):
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
        
    # Min Heap: (distance, node)
    min_heap = [(0, start)]
    shortest = {} # node -> dist
    
    while min_heap:
        w1, n1 = heapq.heappop(min_heap)
        
        if n1 in shortest:
            continue
        shortest[n1] = w1
        
        for n2, w2 in adj[n1]:
            if n2 not in shortest:
                heapq.heappush(min_heap, (w1 + w2, n2))
                
    return shortest

```

---

## HEAPS & INTERVALS

### Top 'K' Elements

**Use when:** "Find K largest/smallest elements".

```python
import heapq
# Keep K largest -> Min Heap of size K
def findKthLargest(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0] # The Kth largest is the smallest in the heap

```

### Merge Intervals

**Use when:** "Merge overlapping time slots".

```python
def mergeIntervals(intervals):
    # 1. Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_end = merged[-1][1]
        
        if start <= last_end:
            # Overlap: extend end time
            merged[-1][1] = max(last_end, end)
        else:
            # No overlap: add new interval
            merged.append([start, end])
    return merged

```

---

## ADVANCED DATA STRUCTURES

### Trie (Prefix Tree)

**Use when:** "Autocomplete", "Search words", "Prefix matching".

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

```

---

## DYNAMIC PROGRAMMING PATTERNS

### Top-Down (Memoization)

**Concept:** Recursion + Caching results.

```python
from functools import lru_cache

# Example: Fibonacci / Climbing Stairs
class Solution:
    def climbStairs(self, n: int) -> int:
        @lru_cache(None) # Automatically handles caching
        def dp(i):
            if i > n: return 0
            if i == n: return 1
            return dp(i + 1) + dp(i + 2)
        
        return dp(0)

```

### Bottom-Up (Tabulation)

**Concept:** Iteration + Array.

```python
def climbStairs(n):
    if n <= 2: return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

```

## USEFUL SYSTEM SETTINGS

### Recursion Limit

Python has a default recursion limit (usually 1000). For deep DFS or large trees, you might hit `RecursionError`.

```python
import sys
sys.setrecursionlimit(10**6) # Increase limit

```

### Caching (Memoization)

Avoid writing manual dictionaries for DP. Use `functools.cache` (Python 3.9+) or `lru_cache`.

```python
from functools import cache

@cache
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)

```

