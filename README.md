# PYTHON DSA CHEAT SHEET

## BASIC OPERATIONS

### Input/Output
```python
x = input()                    # String input
x = int(input())              # Integer input
x, y = map(int, input().split())  # Multiple integers
print(x, y, z)                # Print multiple values
print(x, end=' ')             # Print without newline
```

### Type Conversions
```python
int(x), float(x), str(x), bool(x)
list(x), tuple(x), set(x), dict(x)
ord('A')  # 65 (char to ASCII)
chr(65)   # 'A' (ASCII to char)
```

---

## STRINGS

### Basic Operations
```python
s = "hello"
len(s)                        # 5
s[0], s[-1]                   # 'h', 'o'
s[1:4]                        # 'ell' (slice)
s[::-1]                       # 'olleh' (reverse)
s[::2]                        # 'hlo' (every 2nd char)
```

### String Methods
```python
s.lower(), s.upper()
s.strip()                     # Remove whitespace
s.lstrip(), s.rstrip()
s.split()                     # Split by whitespace
s.split(',')                  # Split by delimiter
'-'.join(['a','b'])           # 'a-b'
s.replace('old', 'new')
s.find('sub')                 # Index or -1
s.count('x')                  # Count occurrences
s.startswith('h'), s.endswith('o')
s.isalpha(), s.isdigit(), s.isalnum()
```

---

## LISTS

### Creation & Access
```python
arr = [1, 2, 3]
arr = [0] * n                 # [0, 0, ..., 0]
arr = [[0] * m for _ in range(n)]  # 2D array
arr[0], arr[-1]               # First, last element
arr[1:3]                      # Slice [index 1 to 2]
```

### List Methods
```python
arr.append(x)                 # Add to end - O(1)
arr.insert(i, x)              # Insert at index - O(n)
arr.extend([4, 5])            # Add multiple - O(k)
arr.pop()                     # Remove last - O(1)
arr.pop(i)                    # Remove at index - O(n)
arr.remove(x)                 # Remove first occurrence - O(n)
arr.clear()                   # Remove all
arr.sort()                    # Sort in-place - O(n log n)
arr.reverse()                 # Reverse in-place
sorted(arr)                   # Return sorted copy
arr.count(x)                  # Count occurrences
arr.index(x)                  # Find index
arr.copy()                    # Shallow copy
```

### List Comprehension
```python
[x*2 for x in arr]
[x for x in arr if x > 0]
[x if x > 0 else 0 for x in arr]
```

---

## TUPLES

### Operations (Immutable)
```python
t = (1, 2, 3)
t = 1, 2, 3                   # Without parentheses
t[0], t[-1]                   # Access
t[1:3]                        # Slice
a, b, c = t                   # Unpack
t.count(x), t.index(x)
```

---

## SETS

### Creation & Operations
```python
s = {1, 2, 3}
s = set([1, 2, 3])
s = set()                     # Empty set (NOT {})
```

### Set Methods
```python
s.add(x)                      # Add element
s.remove(x)                   # Remove (error if not exist)
s.discard(x)                  # Remove (no error)
s.pop()                       # Remove arbitrary element
s.clear()                     # Remove all
x in s                        # Membership - O(1)
```

### Set Operations
```python
a | b, a.union(b)             # Union
a & b, a.intersection(b)      # Intersection
a - b, a.difference(b)        # Difference
a ^ b, a.symmetric_difference(b)  # Symmetric difference
a <= b, a.issubset(b)         # Subset
a >= b, a.issuperset(b)       # Superset
```

---

## DICTIONARIES

### Creation & Access
```python
d = {'a': 1, 'b': 2}
d = dict(a=1, b=2)
d = {}                        # Empty dict
d['a']                        # Access (KeyError if not exist)
d.get('a')                    # Access (None if not exist)
d.get('a', 0)                 # Access with default
```

### Dictionary Methods
```python
d.keys()                      # All keys
d.values()                    # All values
d.items()                     # All (key, value) pairs
d.pop('a')                    # Remove and return value
d.pop('a', 0)                 # With default if not exist
d.popitem()                   # Remove and return (key, value)
d.clear()                     # Remove all
d.update({'c': 3})            # Add/update multiple
d.setdefault('a', 0)          # Get or set default
'a' in d                      # Key membership - O(1)
```

### Dictionary Comprehension
```python
{k: v*2 for k, v in d.items()}
{x: x**2 for x in range(5)}
```

---

## COLLECTIONS MODULE

### Counter
```python
from collections import Counter
c = Counter([1, 2, 2, 3, 3, 3])
c.most_common(2)              # [(3, 3), (2, 2)]
c[1]                          # 1
c.elements()                  # Iterator
c.update([1, 2])              # Add counts
```

### defaultdict
```python
from collections import defaultdict
d = defaultdict(int)          # Default 0
d = defaultdict(list)         # Default []
d = defaultdict(set)          # Default set()
d['a'] += 1                   # No KeyError
```

### deque (Double-ended queue)
```python
from collections import deque
dq = deque([1, 2, 3])
dq.append(4)                  # Add right - O(1)
dq.appendleft(0)              # Add left - O(1)
dq.pop()                      # Remove right - O(1)
dq.popleft()                  # Remove left - O(1)
dq.extend([5, 6])             # Extend right
dq.extendleft([0, -1])        # Extend left
dq.rotate(2)                  # Rotate right
dq.rotate(-2)                 # Rotate left
```

---

## HEAPQ (Min Heap)

```python
import heapq
heap = []
heapq.heappush(heap, x)       # Add element - O(log n)
heapq.heappop(heap)           # Pop min - O(log n)
heap[0]                       # Peek min - O(1)
heapq.heapify(list)           # Convert list to heap - O(n)
heapq.heappushpop(heap, x)    # Push then pop - O(log n)
heapq.heapreplace(heap, x)    # Pop then push - O(log n)
heapq.nlargest(k, list)       # K largest
heapq.nsmallest(k, list)      # K smallest
```

### Max Heap (Negate values)
```python
heap = []
heapq.heappush(heap, -x)      # Negate before push
val = -heapq.heappop(heap)    # Negate after pop
```

---

## BISECT (Binary Search)

```python
import bisect
arr = [1, 3, 4, 4, 6, 8]
bisect.bisect_left(arr, 4)    # 2 (leftmost position)
bisect.bisect_right(arr, 4)   # 4 (rightmost position)
bisect.bisect(arr, 4)         # Same as bisect_right
bisect.insort_left(arr, 5)    # Insert at left position
bisect.insort_right(arr, 5)   # Insert at right position
bisect.insort(arr, 5)         # Same as insort_right
```

---

## MATH MODULE

```python
import math
math.ceil(x)                  # Round up
math.floor(x)                 # Round down
math.sqrt(x)                  # Square root
math.pow(x, y)                # x^y
math.log(x)                   # Natural log
math.log10(x), math.log2(x)   # Base 10, base 2
math.gcd(a, b)                # GCD
math.lcm(a, b)                # LCM (Python 3.9+)
math.factorial(n)             # n!
math.pi, math.e, math.inf     # Constants
abs(x)                        # Absolute value
pow(x, y, m)                  # (x^y) % m
min(a, b, c), max(a, b, c)    # Min/max
```

---

## ITERTOOLS

### Combinatorics
```python
from itertools import permutations, combinations, product
permutations([1,2,3])         # All permutations
permutations([1,2,3], 2)      # Length 2 permutations
combinations([1,2,3], 2)      # Length 2 combinations
combinations_with_replacement([1,2,3], 2)
product([1,2], [3,4])         # Cartesian product
product([1,2], repeat=3)      # [1,2]^3
```

### Other Itertools
```python
from itertools import accumulate, chain, count, cycle
accumulate([1,2,3,4])         # [1, 3, 6, 10] (cumsum)
chain([1,2], [3,4])           # [1, 2, 3, 4]
count(start=0, step=1)        # Infinite counter
cycle([1,2,3])                # Infinite cycle
```

---

## LAMBDA, MAP, FILTER, REDUCE

```python
# Lambda
f = lambda x: x**2
f = lambda x, y: x + y

# Map
list(map(lambda x: x**2, [1,2,3]))  # [1, 4, 9]
list(map(int, ['1','2','3']))       # [1, 2, 3]

# Filter
list(filter(lambda x: x > 0, [-1,0,1,2]))  # [1, 2]

# Reduce
from functools import reduce
reduce(lambda x, y: x+y, [1,2,3,4])  # 10
reduce(lambda x, y: x*y, [1,2,3,4])  # 24
```

---

## COMMON PATTERNS

### Sorting
```python
arr.sort()                    # In-place
sorted(arr)                   # Return new
sorted(arr, reverse=True)     # Descending
sorted(arr, key=lambda x: x[1])  # By 2nd element
sorted(arr, key=lambda x: (x[0], -x[1]))  # Multi-key
```

### Enumerate
```python
for i, val in enumerate(arr):
    print(i, val)
for i, val in enumerate(arr, start=1):
    print(i, val)
```

### Zip
```python
a = [1, 2, 3]
b = ['a', 'b', 'c']
for x, y in zip(a, b):
    print(x, y)
dict(zip(a, b))               # {1: 'a', 2: 'b', 3: 'c'}
```

### Range
```python
range(n)                      # 0 to n-1
range(a, b)                   # a to b-1
range(a, b, step)             # a to b-1 with step
range(n, -1, -1)              # n to 0 (reverse)
```

### Any/All
```python
any([False, True, False])     # True
all([True, True, False])      # False
```

### Sum/Min/Max
```python
sum([1, 2, 3])                # 6
min([1, 2, 3]), max([1, 2, 3])
sum(arr, start_value)         # Sum with initial value
```

---

## STRING/LIST OPERATIONS COMPARISON

### Reverse
```python
s[::-1]                       # String reverse
arr[::-1]                     # List reverse
reversed(arr)                 # Iterator (works for both)
```

### Check Empty
```python
if not s: ...                 # Empty string/list
if len(s) == 0: ...           # Alternative
```

### Copy
```python
s2 = s[:]                     # Shallow copy
arr2 = arr.copy()
arr2 = list(arr)
import copy
deep = copy.deepcopy(arr)     # Deep copy
```

---

## ASCII VALUES

```python
ord('A')  # 65
ord('a')  # 97
ord('0')  # 48
chr(65)   # 'A'
chr(97)   # 'a'
```

---

## BIT OPERATIONS

```python
x & y                         # AND
x | y                         # OR
x ^ y                         # XOR
~x                            # NOT
x << n                        # Left shift
x >> n                        # Right shift
bin(x)                        # Binary representation
x.bit_count()                 # Count 1s (Python 3.10+)
x & (x-1)                     # Remove rightmost 1
x & -x                        # Get rightmost 1
```

---

## INFINITY & NAN

```python
float('inf'), float('-inf')
math.inf, -math.inf
float('nan')
math.isnan(x), math.isinf(x)
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
