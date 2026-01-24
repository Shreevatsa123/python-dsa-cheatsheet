<a name="top"></a>

# PYTHON DSA CHEAT SHEET 

## TABLE OF CONTENTS

1.  [Basic Operations](#basic-operations)
2.  [Strings](#strings)
3.  [Lists](#lists)
4.  [Tuples](#tuples)
5.  [Sets](#sets)
6.  [Dictionaries](#dictionaries)
7.  [Collections Module](#collections-module)
    * [Counter](#counter)
    * [defaultdict](#defaultdict)
    * [deque (Double-ended queue)](#deque-double-ended-queue)
9.  [Queue Module](#queue-module)
10. [Heapq (Min Heap)](#heapq-min-heap)
11. [Bisect (Binary Search)](#bisect-binary-search)
12. [Math Module](#math-module)
13. [Itertools](#itertools)
14. [Lambda, Map, Filter, Reduce](#lambda-map-filter-reduce)
15. [Common Patterns](#common-patterns)
    * [Sorting](#sorting)
    * [Enumerate](#enumerate)
    * [Zip](#zip)
    * [Range](#range)
    * [Any / All](#any--all)
    * [Sum / Min / Max](#sum--min--max)    
16. [String/List Operations Comparison](#stringlist-operations-comparison)
17. [ASCII Values](#ascii-values)
18. [Bit Operations](#bit-operations)
19. [Infinity & NaN](#infinity--nan)
20. [GRAPH BUILDING BOILERPLATE](#graph-building-boilerplate)
21. [Time Complexity Reference](#time-complexity-reference)
22. [The Hidden Imports](#the-hidden-imports)

-----

## BASIC OPERATIONS

### Input/Output

```python
# Read a single line from stdin as a string
x = input()

# Read a single line and cast it to an integer
x = int(input())

# Read a space-separated line, split it, and map all parts to int
# Example Input: "10 20"
x, y = map(int, input().split())  # x is 10, y is 20

# Print multiple variables, separated by a space (default)
print(x, y, z)

# Print, but change the ending character from newline ('\n') to a space
print(x, end=' ')
```

#### More Examples (Input/Output):

```python
# Read a line of space-separated numbers into a list
# Input: "5 10 15 20"
my_list = list(map(int, input().split()))  # my_list is [5, 10, 15, 20]

# Read a line of comma-separated strings into a list
# Input: "apple,banana,cherry"
my_strings = input().split(',')  # my_strings is ['apple', 'banana', 'cherry']

# Print with a custom separator
print('a', 'b', 'c', sep='-')  # Output: "a-b-c"

# f-strings (formatted strings) are the modern way to print variables
name = "Alice"
age = 30
print(f"User {name} is {age} years old.")  # Output: "User Alice is 30 years old."
```

### Type Conversions

```python
# Basic casting
int("123")    # 123
float("12.5") # 12.5
str(123)      # "123"

# Boolean casting (key for if statements)
bool(0)       # False
bool(-1)      # True
bool("")      # False
bool("hi")    # True
bool([])      # False (empty collections are False)
bool([1, 2])  # True (non-empty collections are True)

# Collection casting
list("hello")   # ['h', 'e', 'l', 'l', 'o']
tuple([1, 2])   # (1, 2)
set([1, 2, 2, 3]) # {1, 2, 3} (removes duplicates)
dict([('a', 1), ('b', 2)]) # {'a': 1, 'b': 2}

# Character <-> ASCII
ord('A')  # 65 (char to ASCII integer)
chr(65)   # 'A' (ASCII integer to char)
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## STRINGS

### Basic Operations

```python
s = "hello"

# Get length - O(1)
len(s)                        # 5

# Access by index (0-based) - O(1)
s[0]                          # 'h'
# Access from end (-1 is last) - O(1)
s[-1]                         # 'o'

# Slicing [start:stop(exclusive)] - O(k) where k is slice size
s[1:4]                        # 'ell' (indices 1, 2, 3)

# Slicing [start:stop:step]
# Reverse a string
s[::-1]                       # 'olleh'
# Get every 2nd character
s[::2]                        # 'hlo'
```

#### More Examples (String Slicing & Ops):

```python
s = "abcdefgh"

# Get all *except* the last 2 chars
s[:-2]  # 'abcdef'

# Get all *except* the first 2 chars
s[2:]   # 'cdefgh'

# Get the middle 4 chars
s[2:6]  # 'cdef'

# Concatenation (creates a new string) - O(n + m)
s1 = "hello"
s2 = "world"
s3 = s1 + " " + s2  # "hello world"

# Repetition (creates a new string) - O(n * k)
s4 = s1 * 3  # "hellohellohello"
```

### String Methods

```python
# Case conversion (returns new string)
"Hello".lower()  # "hello"
"Hello".upper()  # "HELLO"

# Remove leading/trailing whitespace
"  Hello ".strip()   # "Hello"
# Remove only leading/trailing
"  Hello ".lstrip()  # "Hello "
"  Hello ".rstrip()  # "  Hello"

# Strip specific characters
"...,,,data,,,".strip(',.')  # "data" (removes any combo from ends)

# Split by whitespace (default)
"Hello World".split()  # ['Hello', 'World']

# Split by a specific delimiter
"user:pass:123".split(':')  # ['user', 'pass', '123']

# Join an iterable of strings with a delimiter
'-'.join(['a', 'b', 'c'])  # 'a-b-c'
#join is not used to joing a string with a character. better to use +. also join is to join a list of characters with a specific delimeter and it also is not inplace = True

# Replace all occurrences (returns new string)
"hello".replace('l', 'X')  # "heXXo"

# Find substring (returns first index or -1 if not found)
"hello".find('ell')  # 1
"hello".find('X')    # -1
# .index() is similar but raises ValueError if not found

# Count occurrences of a substring
"banana".count('a')  # 3

# Check prefix/suffix
"file.txt".startswith('file')  # True
"file.txt".endswith('.csv')    # False

# Type checking
"abc".isalpha()    # True
"123".isdigit()    # True
"abc123".isalnum() # True (alphanumeric)
"  ".isspace()    # True
"Title Case".istitle() # True
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## LISTS

### Creation & Access

```python
# Standard creation
arr = [1, 2, 3]

# * acts as a sequence operation when Python sees a sequence on one side
# and an integer on the other.
#
# Rule: sequence * integer (or integer * sequence)
# What it does: Creates a new, longer sequence by repeating the original one.
#
# A "sequence" in Python is anything that is an ordered collection,
# like a list or a string.

# Create a list of n zeros
n = 5
arr = [0] * n                 # [0, 0, 0, 0, 0]

# Create a 2D array (n x m)
# IMPORTANT: Do NOT use [[0] * m] * n (this creates shallow copies!)
n, m = 3, 4
arr = [[0] * m for _ in range(n)]
# [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

# Access (O(1))
arr[0], arr[-1]               # First, last element

# Slice [start:stop] (O(k))
arr[1:3]                      # [index 1 to 2]
```

### List Methods

```python
arr = [1, 2, 3]

# Add to end - O(1) (amortized)
arr.append(4)                 # [1, 2, 3, 4]

# Insert at index - O(n)
arr.insert(1, 99)             # [1, 99, 2, 3, 4]

# Add all items from another list - O(k)
arr.extend([5, 6])            # [1, 99, 2, 3, 4, 5, 6]

# Remove and return last - O(1)
last_item = arr.pop()         # 6; arr is [1, 99, 2, 3, 4, 5]

# Remove and return at index - O(n)
item = arr.pop(1)             # 99; arr is [1, 2, 3, 4, 5]

# Remove first occurrence of a value - O(n)
arr.remove(3)                 # [1, 2, 4, 5] (raises ValueError if not found)

# Remove all - O(n)
arr.clear()                   # []

# Sort in-place - O(n log n)
arr = [3, 1, 5]
arr.sort()                    # [1, 3, 5]

# Reverse in-place - O(n)
arr.reverse()                 # [5, 3, 1]

# Return a new sorted copy - O(n log n)
new_sorted = sorted([8, 1, 4]) # [1, 4, 8]

# Count occurrences - O(n)
arr = [1, 2, 2, 3]
arr.count(2)                  # 2

# Find index of first occurrence - O(n)
arr.index(2)                  # 1 (raises ValueError if not found)

# Return a shallow copy - O(n)
arr_copy = arr.copy()
# Also common: arr_copy = arr[:]

# 1. Shallow Copy (`.copy()`): Creates a new list, but shares references to any nested objects.
# 2. Deep Copy (`copy.deepcopy()`): Creates a new list AND new copies of all nested objects.
# 3. Shallow: Changing a nested object (e.g., an inner list) will affect the original list.
# 4. Deep: Changing a nested object will *not* affect the original list; it's 100% separate.
# 5. You must `import copy` to use the `deepcopy()` function.

```

### List Comprehension

```python
arr = [1, 2, 3, 4, 5]

# Map: Apply an operation to each item
squares = [x * 2 for x in arr]  # [2, 4, 6, 8, 10]

# Filter: Select only items that meet a condition
evens = [x for x in arr if x % 2 == 0]  # [2, 4]

# Map and Filter
even_squares = [x**2 for x in arr if x % 2 == 0]  # [4, 16]

# Ternary operator
# [value_if_true if condition else value_if_false for x in arr]
processed = [x if x > 2 else 0 for x in arr]  # [0, 0, 3, 4, 5]

# Nested comprehension (e.g., flatten a 2D list)
matrix = [[1, 2], [3, 4]]
flat = [num for row in matrix for num in row]  # [1, 2, 3, 4]
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## TUPLES

### Operations (Immutable)

```python
# Tuples are immutable lists. Use them when data should not change.
# Often used as keys in dictionaries if you need a collection as a key.

# Creation
t = (1, 2, 3)
t = 1, 2, 3                   # Parentheses are optional

# Access (O(1))
t[0], t[-1]                   # 1, 3

# Slice (O(k))
t[1:3]                        # (2, 3)

# Unpacking
a, b, c = t                   # a=1, b=2, c=3

# Methods (O(n))
t.count(1)    # 1
t.index(2)    # 1
```

#### More Examples (Tuple):

```python
# Used as dictionary keys
locations = {
    (40.71, -74.00): "New York",
    (34.05, -118.24): "Los Angeles"
}
print(locations[(40.71, -74.00)]) # "New York"

# Returning multiple values from a function
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)

m, mx, s = get_stats([1, 2, 3]) # m=1, mx=3, s=6
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## SETS

### Creation & Operations

```python
# Sets are unordered collections of unique elements.
# Great for membership testing and removing duplicates.

s = {1, 2, 3}
s = set([1, 2, 2, 3])         # {1, 2, 3}

# IMPORTANT: {} is an empty DICT, not an empty set
s = set()                     # Correct way to make an empty set
```

### Set Methods

```python
s = {1, 2, 3}

# Add element - O(1) average
s.add(4)                      # {1, 2, 3, 4}
s.add(4)                      # {1, 2, 3, 4} (no change)

# Remove element - O(1) average
# Raises KeyError if element doesn't exist
s.remove(3)                   # {1, 2, 4}

# Remove element (no error if not exist) - O(1) average
s.discard(99)                 # {1, 2, 4} (no error)

# Remove and return an arbitrary element - O(1) average
# (Useful for algorithms, but don't rely on *which* element)
elem = s.pop()

# Remove all - O(n)
s.clear()                     # set()

# Membership (THE key feature) - O(1) average
x = 1
if x in s:
    print("Found")
```

### Set Operations

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

# Union (all unique elements from both) - O(len(a) + len(b))
a | b            # {1, 2, 3, 4, 5, 6}
a.union(b)

# Intersection (elements in both) - O(min(len(a), len(b)))
a & b            # {3, 4}
a.intersection(b)

# Difference (elements in a, but not in b) - O(len(a))
a - b            # {1, 2}
a.difference(b)

# Symmetric Difference (elements in one, but not both) - O(len(a) + len(b))
a ^ b            # {1, 2, 5, 6}
a.symmetric_difference(b)

# Subset / Superset - O(len(a))
{1, 2}.issubset(a)      # True
a.issuperset({1, 2})    # True
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## DICTIONARIES

### Creation & Access

```python
# Dictionaries (hash maps) store key-value pairs.
# O(1) average time for lookup, insert, and delete.

# Creation
d = {'a': 1, 'b': 2}
d = dict(a=1, b=2)
d = {}                        # Empty dict

# Access (O(1) avg)
# Raises KeyError if key doesn't exist
val = d['a']                  # 1

# Safe access (O(1) avg)
# Returns None if key doesn't exist
val = d.get('a')              # 1
val = d.get('c')              # None

# Safe access with default (O(1) avg)
val = d.get('c', 0)           # 0 (if 'c' not found, return 0)
```

### Dictionary Methods

```python
d = {'a': 1, 'b': 2}

# Get "views" of keys/values/items.
# These are iterables, not lists.
d.keys()                      # dict_keys(['a', 'b'])
d.values()                    # dict_values([1, 2])
d.items()                     # dict_items([('a', 1), ('b', 2)])

# Loop over items (most common)
for key, value in d.items():
    print(f"{key}: {value}")

# Loop over keys
for key in d:  # Same as `for key in d.keys():`
    print(key)

# Remove and return value by key - O(1) avg
# Raises KeyError if not found
val = d.pop('a')              # 1; d is {'b': 2}

# Safe pop with default - O(1) avg
val = d.pop('c', 0)           # 0; d is unchanged

# Remove and return last (key, value) pair (LIFO) - O(1) avg
item = d.popitem()            # ('b', 2)

# Remove all - O(1)
d.clear()                     # {}

# Add/update multiple items from another dict - O(k)
d.update({'c': 3, 'd': 4})

# Get value or set a default if key not present - O(1) avg
# Useful for initializing
d = {}
d.setdefault('a', []).append(1)  # d is {'a': [1]}
d.setdefault('a', []).append(2)  # d is {'a': [1, 2]}

# Key membership - O(1) avg (THE key feature)
if 'a' in d:
    print("Found 'a'")
```

### Dictionary Comprehension

```python
d = {'a': 1, 'b': 2, 'c': 3}

# Map: Create a new dict with transformed values
{k: v * 2 for k, v in d.items()}  # {'a': 2, 'b': 4, 'c': 6}

# Filter: Create a new dict with subset of items
{k: v for k, v in d.items() if v > 1}  # {'b': 2, 'c': 3}

# Create a dict from a list
nums = [1, 2, 3]
{x: x**2 for x in nums}  # {1: 1, 2: 4, 3: 9}

# Invert a dictionary (swap keys and values)
# (Assumes values are unique)
{v: k for k, v in d.items()}  # {1: 'a', 2: 'b', 3: 'c'}
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## COLLECTIONS MODULE

### Counter

```python
# A dict subclass for counting hashable objects.
from collections import Counter

# Create from list
c = Counter([1, 1, 2, 3, 3, 3, 4])
# c is Counter({3: 3, 1: 2, 2: 1, 4: 1})

# Create from string
c_str = Counter("hello world")
# c_str is Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, ...})

# Get k most common elements
# Returns list of (element, count) tuples
c.most_common(2)              # [(3, 3), (1, 2)]

# Access count (like a dict)
# Returns 0 for missing keys, not KeyError
c[1]                          # 2
c[99]                         # 0

# Get an iterator over elements, repeating
list(c.elements())            # [1, 1, 2, 3, 3, 3, 4]

# Add/subtract counts from another iterable
c.update([1, 2])              # c[1] is now 3, c[2] is now 2
c.subtract([3, 4])            # c[3] is now 2, c[4] is now 0
```

### defaultdict

```python
# A dict subclass that calls a factory function for missing keys.
from collections import defaultdict

# Use `int` factory for a 0 default (e.g., frequency counter)
d = defaultdict(int)
d['a'] += 1                   # No KeyError, d['a'] starts at 0, becomes 1
# d is defaultdict(<class 'int'>, {'a': 1})

# Use `list` factory for a [] default (e.g., grouping items)
d = defaultdict(list)
d['a'].append(1)
d['a'].append(2)
d['b'].append(3)
# d is defaultdict(<class 'list'>, {'a': [1, 2], 'b': [3]})

# Use `set` factory for a set() default
d = defaultdict(set)
d['a'].add(1)
d['a'].add(1)
# d is defaultdict(<class 'set'>, {'a': {1}})
```

### deque (Double-ended queue)

```python
# A list-like container with fast O(1) appends and pops from both ends.
# Use this instead of `list.pop(0)` (which is O(n)).
from collections import deque

# Create
dq = deque([1, 2, 3])

# Add to right - O(1)
dq.append(4)                  # deque([1, 2, 3, 4])

# Add to left - O(1)
dq.appendleft(0)              # deque([0, 1, 2, 3, 4])

# Remove from right - O(1)
dq.pop()                      # 4; dq is deque([0, 1, 2, 3])

# Remove from left - O(1)
dq.popleft()                  # 0; dq is deque([1, 2, 3])

# Add multiple to right - O(k)
dq.extend([4, 5, 6])          # deque([1, 2, 3, 4, 5, 6])

# Add multiple to left - O(k)
# Note: items are added one by one, so order is reversed
dq.extendleft([-1, 0])        # deque([0, -1, 1, 2, 3, 4, 5, 6])

# Rotate right by n steps - O(n)
dq = deque([1, 2, 3, 4])
dq.rotate(1)                  # deque([4, 1, 2, 3])
dq.rotate(2)                  # deque([2, 3, 4, 1])

# Rotate left by n steps - O(n)
dq = deque([1, 2, 3, 4])
dq.rotate(-1)                 # deque([2, 3, 4, 1])
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## Queue Module

```python
from queue import Queue, SimpleQueue, LifoQueue
```

### `Queue` (FIFO)

```python
# BOUNDED: Queue(maxsize=N) holds at most N items.
# put() blocks when full (backpressure).
# UNBOUNDED: Queue(maxsize=0) never blocks, can grow until memory exhausted.
q = Queue(maxsize=3)

q.put(x)                       # may block if full
q.put_nowait(x)                # raises Full if full

x = q.get()                    # may block if empty
x = q.get_nowait()             # raises Empty if empty

q.qsize()                      # approx size
q.empty()                      # approx
q.full()                       # approx (bounded only)

q.task_done()
q.join()
```

### `SimpleQueue` (FIFO, unbounded only)

```python
# Always UNBOUNDED: no maxsize, no full(), no backpressure.
sq = SimpleQueue()

sq.put(x)                      # never blocks
x = sq.get()                   # blocks if empty

sq.qsize()
sq.empty()
# no full(), task_done(), join()
```

### `LifoQueue` (LIFO stack)

```python
# BOUNDED: LifoQueue(maxsize=N) blocks when full.
# UNBOUNDED: LifoQueue(maxsize=0) never blocks.
s = LifoQueue(maxsize=3)

s.put(x)                       # top of stack
x = s.get()                    # last pushed

s.put_nowait(x)                # raises Full if full
x = s.get_nowait()             # raises Empty if empty

s.qsize(); s.empty(); s.full()
s.task_done(); s.join()
```

### Which to use

- `Queue`: FIFO + optional bound + task tracking
- `LifoQueue`: LIFO + optional bound + task tracking
- `SimpleQueue`: FIFO only, unbounded, minimal overhead

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## HEAPQ (Min Heap)

```python
# Implements a min heap. Elements are stored in a regular list.
import heapq

# Start with an empty list
heap = []

# Add element - O(log n)
heapq.heappush(heap, 10)
heapq.heappush(heap, 1)
heapq.heappush(heap, 5)       # heap is now [1, 10, 5] (heap property)

# Pop smallest element - O(log n)
min_val = heapq.heappop(heap) # 1; heap is [5, 10]

# Peek at smallest element - O(1)
smallest = heap[0]            # 5

# Convert an existing list to a heap in-place - O(n)
arr = [6, 2, 8, 1, 4]
heapq.heapify(arr)            # arr is now [1, 2, 8, 6, 4]

# Push then pop (more efficient than push then pop) - O(log n)
# Useful for maintaining a fixed-size heap (e.g., top k)
arr = [1, 2, 8] # heapified
val = heapq.heappushpop(arr, 0) # pushes 0, pops 0. arr is [1, 2, 8]
val = heapq.heappushpop(arr, 5) # pushes 5, pops 1. arr is [2, 5, 8]

# Pop then push - O(log n)
# (heap[0] is popped first, then x is pushed)
val = heapq.heapreplace(arr, 9) # pops 2, pushes 9. arr is [5, 9, 8]

# Get k largest/smallest elements - O(n log k)
arr = [9, 1, 8, 2, 7, 3, 6, 4, 5]
heapq.nlargest(3, arr)        # [9, 8, 7]
heapq.nsmallest(3, arr)       # [1, 2, 3]
```

### Max Heap (Trick)

```python
# Python only has a min heap.
# To simulate a max heap, push and pop the negative of values.
max_heap = []
x = 5

# Negate before push
heapq.heappush(max_heap, -x)
heapq.heappush(max_heap, -10)
heapq.heappush(max_heap, -1)
# max_heap is [-10, -5, -1]

# Negate after pop to get original value
val = -heapq.heappop(max_heap)  # 10
```

-----

## BISECT (Binary Search)

```python
# Provides binary search for *sorted* lists.
import bisect

# bisect_left: finds insertion point to maintain sort,
# preferring the *leftmost* position for duplicates.
# bisect_right: finds insertion point,
# preferring the *rightmost* position.

arr = [1, 3, 4, 4, 6, 8]
val = 4

# Find index for val (leftmost) - O(log n)
bisect.bisect_left(arr, val)    # 2 (points to the first 4)

# Find index for val (rightmost) - O(log n)
bisect.bisect_right(arr, val)   # 4 (points *after* the last 4)
bisect.bisect(arr, val)         # Same as bisect_right

# ---
# `insort` methods insert the item at the correct position.
# These are O(n) because they have to shift elements.
# `bisect_left` + `list.insert` is often slower.

# Insert at left position
bisect.insort_left(arr, 5)    # arr is [1, 3, 4, 4, 5, 6, 8]

# Insert at right position
bisect.insort_right(arr, 4)   # arr is [1, 3, 4, 4, 4, 5, 6, 8]
bisect.insort(arr, 4)         # Same as insort_right
```

#### More Examples (Bisect):

```python
# Check if an item exists in a sorted list (O(log n))
def exists(arr, x):
    i = bisect.bisect_left(arr, x)
    # Check if i is in bounds AND if the item at i is actually x
    return i != len(arr) and arr[i] == x

print(exists(arr, 4)) # True
print(exists(arr, 7)) # False

# Count occurrences of an item in a sorted list (O(log n))
count = bisect.bisect_right(arr, 4) - bisect.bisect_left(arr, 4) # 3
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## MATH MODULE

```python
import math

# Rounding
math.ceil(4.2)                # 5 (round up)
math.floor(4.8)               # 4 (round down)
round(4.6)                    # 5 (standard rounding)

# Roots & Powers
math.sqrt(16)                 # 4.0
math.pow(2, 3)                # 8.0 (2^3)
# Note: `**` operator is usually preferred
2 ** 3                        # 8

# Logs
math.log(math.e)              # 1.0 (natural log, base e)
math.log10(100)               # 2.0 (base 10)
math.log2(8)                  # 3.0 (base 2)

# Divisors
math.gcd(18, 24)              # 6 (Greatest Common Divisor)
# LCM (Python 3.9+)
math.lcm(18, 24)              # 72 (Least Common Multiple)

# Other
math.factorial(5)             # 120 (5 * 4 * 3 * 2 * 1)

# Constants
math.pi                       # 3.14159...
math.e                        # 2.71828...
math.inf                      # Positive infinity
-math.inf                     # Negative infinity

# Built-in math (no import needed)
abs(-5)                       # 5 (Absolute value)
pow(2, 3, 5)                  # (2^3) % 5 = 8 % 5 = 3
min(1, 2, 3), max(1, 2, 3)    # 1, 3
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## ITERTOOLS

### Combinatorics

```python
from itertools import permutations, combinations, product

# All unique orderings of all items
list(permutations([1, 2, 3]))
# [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]

# All unique orderings of length r
list(permutations([1, 2, 3], 2))
# [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# All unique combinations of length r (order doesn't matter)
list(combinations([1, 2, 3], 2))
# [(1, 2), (1, 3), (2, 3)]

# Combinations, but items can be repeated
list(combinations_with_replacement([1, 2, 3], 2))
# [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

# Cartesian product (like nested loops)
list(product([1, 2], ['a', 'b']))
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

list(product([1, 2], repeat=2)) # [1, 2] x [1, 2]
# [(1, 1), (1, 2), (2, 1), (2, 2)]
```

### Other Itertools

```python
from itertools import accumulate, chain, count, cycle

# Running total (cumulative sum)
list(accumulate([1, 2, 3, 4]))         # [1, 3, 6, 10]
# Running product
import operator
list(accumulate([1, 2, 3, 4], operator.mul)) # [1, 2, 6, 24]

# Flatten one level of iterables
list(chain([1, 2], [3, 4], ['a']))     # [1, 2, 3, 4, 'a']

# Infinite counter
# count(start=0, step=1)
# for i in count(10): ... (10, 11, 12, ...)

# Infinite cycle
# cycle([1, 2, 3])
# for x in cycle([1, 2]): ... (1, 2, 1, 2, 1, ...)
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## LAMBDA, MAP, FILTER, REDUCE

```python
# Lambda: small, anonymous one-line functions

# f = lambda [args]: [expression]
f_sq = lambda x: x**2
f_add = lambda x, y: x + y
f_sq(5)  # 25
f_add(3, 4) # 7

# ---
# Map: Apply a function to every item in an iterable
# list(map(function, iterable))

# Using lambda
list(map(lambda x: x**2, [1, 2, 3]))  # [1, 4, 9]
# Using existing function
list(map(str, [1, 2, 3]))             # ['1', '2', '3']

# ---
# Filter: Get items from an iterable where function returns True
# list(filter(function, iterable))

list(filter(lambda x: x > 0, [-1, 0, 1, 2]))  # [1, 2]

# ---
# Reduce: Apply a rolling computation to sequential pairs
from functools import reduce

# reduce(function, iterable, [initializer])
# ( (1+2)+3 )+4
reduce(lambda x, y: x + y, [1, 2, 3, 4])  # 10
# ( (1*2)*3 )*4
reduce(lambda x, y: x * y, [1, 2, 3, 4])  # 24

# With initializer (acts as first value)
# ( (10+1)+2 )+3
reduce(lambda x, y: x + y, [1, 2, 3], 10) # 16
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## COMMON PATTERNS

### Sorting

```python
arr = [3, 1, -5, 2]
arr_pairs = [(1, 5), (3, 2), (2, 8)]

# In-place sort
arr.sort()                    # [-5, 1, 2, 3]

# Return new sorted list
sorted_arr = sorted(arr)

# Descending order
arr.sort(reverse=True)        # [3, 2, 1, -5]
sorted(arr, reverse=True)

# Sort by a custom key (e.g., absolute value)
sorted(arr, key=lambda x: abs(x))  # [1, 2, 3, -5]

# Sort by 2nd element of a tuple (index 1)
sorted(arr_pairs, key=lambda x: x[1])  # [(3, 2), (1, 5), (2, 8)]

# Multi-key sort
# Sort by 1st element (ascending), then 2nd (descending)
arr_complex = [(1, 5), (2, 2), (1, 2)]
sorted(arr_complex, key=lambda x: (x[0], -x[1]))
# [(1, 5), (1, 2), (2, 2)]
```

### Enumerate

```python
# Get (index, value) pairs
arr = ['a', 'b', 'c']
for i, val in enumerate(arr):
    print(i, val)
# 0 a
# 1 b
# 2 c

# Start index at 1
for i, val in enumerate(arr, start=1):
    print(i, val)
# 1 a
# 2 b
# 3 c
```

Why enumerate is better than list[i] - 

1. **Cleaner Syntax:** It unpacks the index and value instantly, eliminating the need for cluttered manual lookups like `list[i]`.
2. **Universal Compatibility:** It works on **all** iterables (including generators, files, and streams), whereas `range(len())` fails on data without a known length.
3. **Performance:** It is optimized at the C-level and avoids the overhead of repeated indexing into the list during every iteration.
4. **Flexibility:** It offers a built-in `start` parameter (e.g., `enumerate(list, start=1)`), removing the need for manual math (`i+1`) inside the loop.
5. **Standard Practice:** It is considered "Pythonic," making your code immediately recognizable and easier to maintain for other developers.

---

**Would you like me to create a quick "Bad vs Good" code snippet to go with these 5 lines?**

### Zip

```python
# Combine multiple iterables element-wise
# Stops at the shortest iterable
a = [1, 2, 3]
b = ['a', 'b', 'c', 'd'] # 'd' will be ignored
for x, y in zip(a, b):
    print(x, y)
# 1 a
# 2 b
# 3 c

# Create a dictionary
dict(zip(a, b))               # {1: 'a', 2: 'b', 3: 'c'}
```

### Range

```python
# 0 to n-1
list(range(5))                      # [0, 1, 2, 3, 4]

# a to b-1
list(range(2, 5))                   # [2, 3, 4]

# a to b-1 with step
list(range(0, 10, 2))               # [0, 2, 4, 6, 8]

# Reverse: n to 0
list(range(5, -1, -1))              # [5, 4, 3, 2, 1, 0]
```

### Any/All

```python
# any: True if at least one element is True
any([False, True, False])     # True
any([False, False, False])    # False

# all: True if all elements are True
all([True, True, True])       # True
all([True, True, False])      # False
```

### Sum/Min/Max

```python
nums = [1, 2, 3]
sum(nums)                # 6
min(nums), max(nums)     # (1, 3)

# Sum with an initial start value
sum(nums, 10)            # 16 (10 + 1 + 2 + 3)

# Find min/max of complex items with a key
arr_pairs = [(1, 5), (3, 2), (2, 8)]
min(arr_pairs, key=lambda x: x[1]) # (3, 2)
max(arr_pairs, key=lambda x: x[1]) # (2, 8)
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## STRING/LIST OPERATIONS COMPARISON

### Reverse

```python
# String (creates new string)
s = "abc"
rev_s = s[::-1]               # "cba"

# List (in-place)
arr = [1, 2, 3]
arr.reverse()                 # arr is [3, 2, 1]

# List (creates new list)
rev_arr = arr[::-1]           # [1, 2, 3] (based on reversed arr)

# Get a reverse *iterator* (memory efficient)
for x in reversed(arr):
    print(x)
```

### Check Empty

```python
# Pythonic way (empty collections/strings are "Falsy")
if not s: ...                 # Empty string
if not arr: ...               # Empty list

# Explicit way
if len(s) == 0: ...
```

### Copy

```python
# String (immutable, so no real "copy" needed)
s = "abc"
s2 = s

# ---
# List (Shallow Copy)
# Copies the list, but not the objects *inside*
arr = [[1], [2]]
arr2 = arr.copy()
# arr2 = arr[:]
# arr2 = list(arr)
arr2[0].append(99)
# arr is now [[1, 99], [2]]
# arr2 is also [[1, 99], [2]] (they share inner lists)

# ---
# List (Deep Copy)
# Copies the list AND all objects inside recursively
import copy
arr = [[1], [2]]
deep = copy.deepcopy(arr)
deep[0].append(99)
# arr is [[1], [2]]
# deep is [[1, 99], [2]] (they are fully separate)
```

-----
<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## ASCII VALUES

```python
# Get integer code point
ord('A')  # 65
ord('a')  # 97
ord('0')  # 48

# Get character from code point
chr(65)   # 'A'
chr(97)   # 'a'

# Common trick: Find 0-based index of a letter
char = 'c'
index = ord(char) - ord('a') # 2

# Common trick: Convert number 0-9 to char
num = 5
char = chr(num + ord('0')) # '5'
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## BIT OPERATIONS

```python
x = 5  # 0101
y = 3  # 0011

x & y                         # AND: 0001 (1)
x | y                         # OR:  0111 (7)
x ^ y                         # XOR: 0110 (6)
~x                            # NOT: ...11111010 (-6 in 2's complement)
x << n                        # Left shift (x * 2^n)
x << 1                        # 1010 (10)
x >> n                        # Right shift (x // 2^n)
x >> 1                        # 0010 (2)

# Get binary string (0b prefix)
bin(x)                        # '0b101'

# Count 1s (Python 3.10+)
x.bit_count()                 # 2

# Check if k-th bit is set (0-indexed from right)
k = 0
if (x >> k) & 1: print("0th bit is set") # True
k = 1
if (x >> k) & 1: print("1st bit is set") # False

# Set k-th bit
x = x | (1 << k) # x | (1 << 1) = 0101 | 0010 = 0111 (7)

# Unset k-th bit
x = x & ~(1 << k) # x & ~(1 << 0) = 0101 & 1110 = 0100 (4)

# Flip k-th bit
x = x ^ (1 << k) # 0101 ^ (1 << 2) = 0101 ^ 0100 = 0001 (1)

# Brian Kernighan's Algorithm: Remove rightmost 1
# Used to count 1s in a loop
x & (x - 1)                   # 5 & 4 = 0101 & 0100 = 0100 (4)

# Get rightmost 1
x & -x                        # 5 & -5 = 0101 & ...1011 = 0001 (1)
```

-----

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## INFINITY & NAN

```python
# Positive/Negative Infinity
pos_inf = float('inf')
neg_inf = float('-inf')
# Also from math module
pos_inf = math.inf
neg_inf = -math.inf

# Useful for comparisons (e.g., finding a minimum)
min_val = math.inf
for x in [1, 5, -2]:
    min_val = min(min_val, x) # min_val ends up as -2

# Not a Number
nan = float('nan')

# Check for them
math.isnan(nan)               # True
math.isinf(pos_inf)           # True
# Note: `nan == nan` is *always* False
```

<p align="lefft">
  <a href="#top">Back to Top</a>
</p>

## GRAPH BUILDING BOILERPLATE

Constructing an adjacency list is the first step in 90% of graph problems.

```python
from collections import defaultdict

edges = [[0, 1], [1, 2], [2, 0]]
n = 3

# 1. Undirected Graph
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# 2. Directed Graph
adj = defaultdict(list)
for src, dest in edges:
    adj[src].append(dest)

# 3. Weighted Graph
# edges = [[0, 1, 5], ...]
adj = defaultdict(list)
for u, v, w in edges:
    adj[u].append((v, w))
    adj[v].append((u, w))


-----

```

## TIME COMPLEXITY REFERENCE

| Operation | List | Set (Avg) | Dict (Avg) | Deque | Heap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Access** `[i]` or `[k]` | O(1) | - | O(1) | O(1) (ends) / O(n) (mid) | O(1) `heap[0]` (Peek Min) |
| **Search** `x in ...` | O(n) | O(1) | O(1) `k in d` | O(n) | O(n) (Heaps aren't for search) |
| **Insert (End)** | O(1) `append` | O(1) `add` | O(1) `d[k]=v` | O(1) `append` | O(log n) `heappush` |
| **Insert (Start/Mid)** | O(n) `insert` | - | - | O(1) `appendleft` | - |
| **Delete (End)** | O(1) `pop` | O(1) `remove` | O(1) `pop(k)` | O(1) `pop` | O(log n) `heappop` (Min) |
| **Delete (Start/Mid)** | O(n) `pop(0)` | O(1) `remove` | O(1) `pop(k)` | O(1) `popleft` | O(log n) `heappop` (Min) |
| **Sort** | O(n log n) | - | - | - | O(n) `heapify` (Build heap) |

*(Avg) = Average Case. Set/Dict have a worst-case of O(n) due to hash collisions, but this is rare in practice.*

## The "Hidden" Imports

You don't need to import common libraries. LeetCode automatically imports these for you:

* `collections` (Counter, deque, defaultdict)
* `heapq` (heappush, heappop, heapify)
* `bisect` (bisect_left, bisect_right)
* `math` (inf, ceil, floor, gcd)
* `functools` (cache, lru_cache, cmp_to_key)
* `itertools` (permutations, combinations, product)
* `random` (randint, choice)
