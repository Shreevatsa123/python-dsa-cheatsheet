# PYTHON LEETCODE/INTERVIEW GUIDE (Pattern-Focused)

This guide focuses on common *patterns*, *templates*, and *problem-specific structures* used in LeetCode, assuming you are familiar with the core data structure operations from the main `README.md` cheatsheet.

## TABLE OF CONTENTS

1.  [Core Data Structure Patterns](https://www.google.com/search?q=%23core-data-structure-patterns)
      * [Stack (LIFO)](https://www.google.com/search?q=%23stack-lifo)
      * [Queue (FIFO) & BFS Pattern](https://www.google.com/search?q=%23queue-fifo--bfs-pattern)
2.  [Core Algorithm Patterns](https://www.google.com/search?q=%23core-algorithm-patterns)
      * [Two Pointers](https://www.google.com/search?q=%23two-pointers)
      * [Sliding Window](https://www.google.com/search?q=%23sliding-window)
      * [Manual Binary Search](https://www.google.com/search?q=%23manual-binary-search)
3.  [Recursive Patterns](https://www.google.com/search?q=%23recursive-patterns)
      * [DFS (Recursion)](https://www.google.com/search?q=%23dfs-recursion)
      * [Backtracking](https://www.google.com/search?q=%23backtracking)
4.  [Problem-Specific Structures](https://www.google.com/search?q=%23problem-specific-structures)
      * [Linked List Operations](https://www.google.com/search?q=%23linked-list-operations)
      * [Tree Operations](https://www.google.com/search?q=%23tree-operations)
5.  [Python-Specific Tricks](https://www.google.com/search?q=%23python-specific-tricks)
6.  [Quick Reference: When to Use What](https://www.google.com/search?q=%23quick-reference-when-to-use-what)
7.  [Core Function Reference (from README.md)](https://www.google.com/search?q=%23core-function-reference-from-readmemd)

-----

## CORE DATA STRUCTURE PATTERNS

### Stack (LIFO)

Used for "last-in, first-out" logic, like matching parentheses, path traversal, or monotonicity. Implemented using a plain `list`.

```python
stack = []
stack.append(x)                # Push - O(1)
if stack:                      # Check if not empty
    val = stack.pop()          # Pop - O(1)
    peek = stack[-1]           # Peek - O(1)
```

### Queue (FIFO) & BFS Pattern

Used for "first-in, first-out" logic, essential for **Breadth-First Search (BFS)** and level-order traversal.

**Note:** For the `deque` object itself, see [deque in README.md](https://www.google.com/search?q=%23deque-double-ended-queue).

```python
# BFS / Level-Order Pattern
from collections import deque
q = deque([start_node])
visited = {start_node}

while q:
    # --- For Level-Order Traversal ---
    level_size = len(q)
    current_level = []
    for _ in range(level_size):
    # ---------------------------------
        
        node = q.popleft()     # Dequeue - O(1)
        # current_level.append(node.val) # Add to level
        
        # Process node...

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor) # Enqueue - O(1)
```

-----

## CORE ALGORITHM PATTERNS

### Two Pointers

Used on sorted arrays or strings to find pairs or subarrays.

```python
# Two pointers from ends
left, right = 0, len(arr) - 1

while left < right:
    s = arr[left] + arr[right]
    if s == target:
        # Found
        left += 1
        right -= 1
    elif s < target:
        left += 1
    else:
        right -= 1
```

### Sliding Window

Used on iterables (strings, arrays) to find optimal (min/max) contiguous subarrays.

```python
window = {} # e.g., Counter or set
left = 0
min_len = float('inf')

for right in range(len(arr)):
    # 1. Add right element to window
    window[arr[right]] = window.get(arr[right], 0) + 1
    
    # 2. Check if window is valid
    while condition_is_met:
        # 3. Update result
        min_len = min(min_len, right - left + 1)
        
        # 4. Shrink from left
        window[arr[left]] -= 1
        if window[arr[left]] == 0:
            del window[arr[left]]
        left += 1
```

### Manual Binary Search

Used when `bisect` isn't flexible enough (e.g., finding first/last occurrence, searching on a condition).

**Note:** For the `bisect` module, see [Bisect in README.md](https://www.google.com/search?q=%23bisect-binary-search).

```python
left, right = 0, len(arr) - 1
while left <= right:
    mid = (left + right) // 2
    if arr[mid] == target:
        # Found it, handle logic
        return mid
    elif arr[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
```

-----

## RECURSIVE PATTERNS

### DFS (Recursion)

Used for exploring graphs or trees depth-first.

```python
visited = set()
def dfs(node):
    if not node or node in visited:
        return
    
    visited.add(node)
    
    # Process node (pre-order)
    
    for neighbor in node.neighbors:
        dfs(neighbor)
        
    # Process node (post-order)
```

### Backtracking

Used for finding all possible solutions (e.g., permutations, combinations, subsets, pathfinding).

```python
def backtrack(path, remaining_candidates):
    # 1. Base case: Solution found
    if condition_met:
        result.append(path[:]) # Append a *copy*
        return
    
    # 2. Base case: Invalid path
    if not_valid:
        return
    
    # 3. Iterate through choices
    for i in range(len(remaining_candidates)):
        choice = remaining_candidates[i]
        
        # Add choice to path
        path.append(choice)
        
        # Recurse
        # (Pass modified candidates, e.g., `i + 1` for subsets)
        backtrack(path, new_candidates)
        
        # Backtrack (remove choice)
        path.pop()
```

-----

## PROBLEM-SPECIFIC STRUCTURES

### Linked List Operations

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Reverse linked list (Iterative)
def reverseList(head):
    prev, curr = None, head
    while curr:
        next_node = curr.next # Store next
        curr.next = prev      # Reverse link
        prev = curr           # Move prev
        curr = next_node      # Move curr
    return prev # New head is prev

# Check cycle (Slow/Fast pointers)
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Get kth node from end
def getKthFromEnd(head, k):
    fast = head
    for _ in range(k): # Move fast k steps ahead
        fast = fast.next
    
    slow = head
    while fast: # When fast hits end, slow is at kth from end
        slow = slow.next
        fast = fast.next
    
    return slow
```

### Tree Operations

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Inorder traversal (Left -> Root -> Right)
def inorder(root):
    result = []
    if not root:
        return result
    result.extend(inorder(root.left))
    result.append(root.val)
    result.extend(inorder(root.right))
    return result

# Preorder traversal (Root -> Left -> Right)
def preorder(root):
    result = []
    if not root:
        return result
    result.append(root.val)
    result.extend(preorder(root.left))
    result.extend(preorder(root.right))
    return result

# Postorder traversal (Left -> Right -> Root)
def postorder(root):
    result = []
    if not root:
        return result
    result.extend(postorder(root.left))
    result.extend(postorder(root.right))
    result.append(root.val)
    return result
```

-----

## PYTHON-SPECIFIC TRICKS

```python
# Swap without temp variable
a, b = b, a

# Multiple assignment
x, y, z = 1, 2, 3

# Unpack with * (e.g., get head and tail)
first, *middle, last = [1, 2, 3, 4, 5]
# first=1, middle=[2,3,4], last=5
```

-----

## QUICK REFERENCE: WHEN TO USE WHAT

| Use Case | Data Structure | Why |
|:---|:---|:---|
| Fast lookup (is `x` in set?) | Set / Dict | O(1) average |
| Store ordered items | List | O(1) access by index |
| Frequency count | `Counter` | O(n) build, O(1) access |
| Grouping items | `defaultdict(list)` | Clean, no key checks |
| Priority queue (min/max) | `heapq` | O(log n) push/pop |
| BFS (graph/tree levels) | `deque` | O(1) push/pop both ends |
| Stack (DFS, recursion) | `list` (`append`/`pop`) | O(1) push/pop |
| Remove duplicates | `set(arr)` | O(n) |
| Binary search on sorted data | `bisect` module | O(log n) |
| Immutable key for dict | Tuple | Lists/sets can't be keys |

-----

## CORE FUNCTION REFERENCE (FROM README.MD)

The following topics are covered in detail in the main `README.md` cheatsheet. Use these links for a refresher on core function syntax.

  * [**Sorting** (sorted, .sort, key=)](https://www.google.com/search?q=%23sorting)
  * [**String Operations** (.split, .join, .strip, etc.)](https://www.google.com/search?q=%23strings)
  * [**List Methods** (.append, .pop, .remove, etc.)](https://www.google.com/search?q=%23lists)
  * [**Set Operations** (add, remove, &, |)](https://www.google.com/search?q=%23sets)
  * [**Dictionary Operations** (.get, .keys, .items, etc.)](https://www.google.com/search?q=%23dictionaries)
  * [**defaultdict & Counter**](https://www.google.com/search?q=%23collections-module)
  * [**Heap (Priority Queue)** (heappush, heappop, heapify)](https://www.google.com/search?q=%23heapq-min-heap)
  * [**Bisect Module** (bisect\_left, bisect\_right)](https://www.google.com/search?q=%23bisect-binary-search)
  * [**List Comprehension**](https://www.google.com/search?q=%23list-comprehension)
  * [**Lambda, map, filter**](https://www.google.com/search?q=%23lambda-map-filter-reduce)
  * [**Enumerate**](https://www.google.com/search?q=%23enumerate)
  * [**Zip**](https://www.google.com/search?q=%23zip)
  * [**Any/All**](https://www.google.com/search?q=%23anyall)
  * [**Min/Max/Sum**](https://www.google.com/search?q=%23summinmax)
  * [**Math Functions** (ceil, floor, gcd, etc.)](https://www.google.com/search?q=%23math-module)
  * [**Time Complexity Table**](https://www.google.com/search?q=%23time-complexity-reference)
