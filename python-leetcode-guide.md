# PYTHON DSA CHEAT SHEET - MOST USED LEETCODE/INTERVIEW FUNCTIONS

## MUST-KNOW: TOP 20 MOST USED FUNCTIONS

### 1. Sorting & Ordering
```python
sorted(arr)                    # Return sorted list - O(n log n)
arr.sort()                     # Sort in-place
sorted(arr, reverse=True)      # Descending
sorted(arr, key=lambda x: x[1])  # Sort by specific key
sorted(arr, key=abs)           # Sort by absolute value

# Common: Sort tuples by 2nd element
points.sort(key=lambda p: p[1])
```

### 2. String Operations (VERY COMMON)
```python
s.split()                      # Split by whitespace
s.split(',')                   # Split by delimiter
'-'.join(list)                 # Join list to string
s.strip()                      # Remove leading/trailing whitespace
s.replace('old', 'new')        # Replace substring
s.upper(), s.lower()
s.isdigit(), s.isalpha()
s.count(char)                  # Count occurrences
```

### 3. List Methods (CRITICAL)
```python
arr.append(x)                  # O(1)
arr.extend([x, y])             # O(k) - add multiple
arr.pop()                      # O(1) - remove last
arr.pop(i)                     # O(n) - remove at index
arr.remove(x)                  # O(n) - remove first occurrence
arr.reverse()                  # O(n) - reverse in-place
arr.insert(i, x)               # O(n) - insert at index
arr.index(x)                   # O(n) - find index
arr.count(x)                   # O(n) - count occurrences
arr.sort()                     # O(n log n)
```

### 4. Set Operations (VERY FAST)
```python
s = set(arr)                   # Remove duplicates - O(n)
x in s                         # Check membership - O(1) ⭐
s.add(x), s.remove(x)
s1 & s2                        # Intersection - O(min(len(s1), len(s2)))
s1 | s2                        # Union
s1 - s2                        # Difference
list(set(arr))                 # Unique elements (loses order)
```

### 5. Dictionary Operations (SUPER COMMON)
```python
d = {k: v}
d.get(k)                       # Get with None default - O(1)
d.get(k, default)              # Get with custom default
d[k]                           # Direct access (KeyError if not exist)
k in d                         # Check key existence - O(1) ⭐
d.keys(), d.values(), d.items()
d.pop(k)                       # Remove and return
d.setdefault(k, v)             # Get or set default
d.update({'a': 1})

# VERY COMMON: Count frequencies
from collections import Counter
freq = Counter(arr)            # Dict with counts
freq[x]                        # Get count of x
freq.most_common(k)            # K most common elements
```

### 6. Two Pointers/Sliding Window (ESSENTIAL)
```python
# Two pointers
left, right = 0, len(arr) - 1

# While loop for two pointer
while left < right:
    left += 1
    right -= 1

# Sliding window
window = {}
left = 0
for right in range(len(arr)):
    # Add right
    window[arr[right]] = window.get(arr[right], 0) + 1
    # Shrink from left
    while condition:
        left += 1
```

### 7. Stack Operations
```python
stack = []
stack.append(x)                # Push - O(1)
stack.pop()                    # Pop - O(1)
stack[-1]                      # Peek - O(1)
if stack: ...                  # Check empty
```

### 8. Queue Operations (Use deque!)
```python
from collections import deque
q = deque()
q.append(x)                    # Enqueue right - O(1)
q.popleft()                    # Dequeue left - O(1)
q[0]                           # Peek front
len(q) == 0                    # Check empty

# BFS pattern
from collections import deque
q = deque([start])
visited = {start}
while q:
    node = q.popleft()
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            q.append(neighbor)
```

### 9. Heap (Priority Queue) - IMPORTANT
```python
import heapq
heap = []
heapq.heappush(heap, x)        # Push - O(log n)
heapq.heappop(heap)            # Pop min - O(log n)
heap[0]                        # Peek min - O(1)
heapq.heapify(arr)             # Build heap - O(n)

# Max heap (negate values)
heapq.heappush(heap, -x)
val = -heapq.heappop(heap)

# K largest/smallest
heapq.nlargest(k, arr)         # K largest
heapq.nsmallest(k, arr)        # K smallest
```

### 10. Binary Search - COMMON IN SORTED PROBLEMS
```python
import bisect
arr = [1, 3, 4, 4, 6, 8]
bisect.bisect_left(arr, 4)     # 2 (leftmost)
bisect.bisect_right(arr, 4)    # 4 (rightmost)
bisect.bisect(arr, 4)          # Same as right
bisect.insort(arr, 5)          # Insert in sorted order

# Manual binary search (often needed)
left, right = 0, len(arr) - 1
while left <= right:
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
```

### 11. Hash Map for Counting/Grouping (CRITICAL)
```python
# Count frequencies
count = {}
for x in arr:
    count[x] = count.get(x, 0) + 1

# Or use Counter (cleaner)
from collections import Counter
count = Counter(arr)

# Group by value
from collections import defaultdict
groups = defaultdict(list)
for key, val in items:
    groups[val].append(key)
```

### 12. List Comprehension (QUICK & PYTHONIC)
```python
[x*2 for x in arr]             # Map
[x for x in arr if x > 0]      # Filter
[[0]*m for _ in range(n)]      # 2D array
[sorted(group) for group in groups]  # Transform each
```

### 13. Lambda with map/filter
```python
list(map(lambda x: x**2, [1,2,3]))  # [1, 4, 9]
list(filter(lambda x: x > 0, [-1,0,1]))  # [1]

# Sorting with lambda
sorted(people, key=lambda x: (x[0], -x[1]))  # Multi-key sort
```

### 14. String to List and Vice Versa
```python
s = "hello"
list(s)                        # ['h', 'e', 'l', 'l', 'o']
''.join(['h','i'])             # 'hi'

# String reversal
s[::-1]                        # Reverse
s[::2]                         # Every 2nd char
s[:n]                          # First n chars
s[-n:]                         # Last n chars
```

### 15. Enumerate (VERY USEFUL)
```python
for i, val in enumerate(arr):
    print(i, val)

for i, val in enumerate(arr, start=1):
    print(i, val)               # 1-indexed
```

### 16. Zip (Combine Lists)
```python
a = [1, 2, 3]
b = ['a', 'b', 'c']
for x, y in zip(a, b):
    print(x, y)                # (1, 'a'), (2, 'b'), (3, 'c')

list(zip(a, b))                # [(1, 'a'), (2, 'b'), (3, 'c')]
dict(zip(keys, values))        # Create dict
```

### 17. Any/All
```python
any([False, True, False])      # True - if ANY is True
all([True, True, False])       # False - if ALL are True
any(x > 5 for x in arr)        # Any element > 5?
all(x > 0 for x in arr)        # All positive?
```

### 18. Min/Max/Sum
```python
min(arr), max(arr)
min([1, 2, 3])                 # 1
max(arr, key=lambda x: x[1])   # Max by key
sum(arr)                       # Total
sum(arr, start=10)             # Sum with initial value
```

### 19. Math Functions (OFTEN NEEDED)
```python
import math
math.ceil(3.2)                 # 4 (round up)
math.floor(3.7)                # 3 (round down)
int(x)                         # Truncate (faster than floor)
x // 2                         # Integer division
x % 2                          # Modulo
abs(x)                         # Absolute value
math.sqrt(x)                   # Square root
math.gcd(a, b)                 # GCD
pow(2, 3)                      # 2^3
2 ** 3                         # Also 2^3
math.log(x), math.log2(x)      # Logarithm
```

### 20. defaultdict & Counter (VERY USEFUL)
```python
from collections import defaultdict, Counter

# defaultdict - no KeyError
d = defaultdict(int)           # Default 0
d = defaultdict(list)          # Default []
d = defaultdict(set)           # Default set()
d['key'] += 1                  # No KeyError!

# Counter - count elements
c = Counter([1,2,2,3,3,3])
c[1]                           # 1
c[3]                           # 3
c.most_common(2)               # [(3, 3), (2, 2)]
c.update([1, 2])               # Add counts
```

---

## COMMON PATTERNS FOR LEETCODE

### Pattern 1: Two Pointer
```python
def twoSum(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return [left, right]
        elif s < target:
            left += 1
        else:
            right -= 1
```

### Pattern 2: Sliding Window
```python
def maxWindow(s, k):
    window = {}
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        window[s[right]] = window.get(s[right], 0) + 1
        
        while right - left + 1 > k:
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

### Pattern 3: BFS (Level Order Traversal)
```python
from collections import deque

def bfs(root):
    if not root:
        return []
    
    q = deque([root])
    result = []
    
    while q:
        level_size = len(q)
        current_level = []
        
        for _ in range(level_size):
            node = q.popleft()
            current_level.append(node.val)
            
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        
        result.append(current_level)
    
    return result
```

### Pattern 4: DFS (Recursion)
```python
def dfs(node, target, visited):
    if not node or node in visited:
        return None
    
    visited.add(node)
    
    if node.val == target:
        return node
    
    for neighbor in node.neighbors:
        result = dfs(neighbor, target, visited)
        if result:
            return result
    
    return None
```

### Pattern 5: Backtracking
```python
def backtrack(candidates, target, path, result, start):
    if sum(path) == target:
        result.append(path[:])
        return
    
    if sum(path) > target:
        return
    
    for i in range(start, len(candidates)):
        path.append(candidates[i])
        backtrack(candidates, target, path, result, i)
        path.pop()
```

### Pattern 6: Hash Map Frequency Count
```python
# Most common in LeetCode
count = {}
for item in arr:
    count[item] = count.get(item, 0) + 1

# Find most frequent
most_freq = max(count, key=count.get)
most_freq_count = count[most_freq]
```

### Pattern 7: Sorting Trick
```python
# Sort by multiple criteria
people.sort(key=lambda x: (x[0], -x[1]))  # By age asc, height desc

# Sort list of tuples
pairs = [(1, 'a'), (2, 'b'), (1, 'c')]
pairs.sort()                   # Sort by first element
pairs.sort(key=lambda x: x[1]) # Sort by second element
```

---

## LINKED LIST OPERATIONS (Common)

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Reverse linked list
def reverseList(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev

# Check cycle (slow/fast pointers)
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
    for _ in range(k):
        fast = fast.next
    
    slow = head
    while fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

---

## TREE OPERATIONS (Common)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Inorder traversal (Left -> Root -> Right)
def inorder(root, result=[]):
    if not root:
        return result
    inorder(root.left, result)
    result.append(root.val)
    inorder(root.right, result)
    return result

# Preorder traversal (Root -> Left -> Right)
def preorder(root, result=[]):
    if not root:
        return result
    result.append(root.val)
    preorder(root.left, result)
    preorder(root.right, result)
    return result

# Level order (BFS)
from collections import deque
def levelOrder(root):
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        level_size = len(q)
        level = []
        for _ in range(level_size):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(level)
    return result
```

---

## TIME COMPLEXITY REFERENCE

| Operation | List | Set | Dict | Deque | Heap |
|-----------|------|-----|------|-------|------|
| Access | O(1) | - | O(1) | O(n) | - |
| Search | O(n) | O(1) | O(1) | O(n) | O(n) |
| Insert | O(n) | O(1) | O(1) | O(1) | O(log n) |
| Delete | O(n) | O(1) | O(1) | O(1) | O(log n) |
| Sort | O(n log n) | - | - | - | - |
| Heappush | - | - | - | - | O(log n) |
| Heappop | - | - | - | - | O(log n) |

---

## SPACE-SAVING TRICKS

```python
# Swap without temp variable
a, b = b, a

# Multiple assignment
x, y, z = [1, 2, 3]

# Unpack with *
first, *middle, last = [1, 2, 3, 4, 5]
# first=1, middle=[2,3,4], last=5

# Dictionary comprehension
{k: v for k, v in items}
{k: v**2 for k, v in d.items()}

# Set comprehension
{x**2 for x in range(5)}
```

---

## QUICK REFERENCE: WHEN TO USE WHAT

| Use Case | Data Structure | Why |
|----------|----------------|-----|
| Fast lookup | Set/Dict | O(1) |
| Maintain order | List | O(n) access but keeps order |
| Frequency count | Counter | Built-in, fastest |
| Priority queue | Heap | O(log n) ops |
| Graph traversal | deque+visited set | O(1) operations |
| String manipulation | strings directly | Python optimized |
| Remove duplicates | set(arr) | O(n) |
| Sort custom | sorted(arr, key=...) | Flexible |
