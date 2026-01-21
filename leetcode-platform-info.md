## 1. The Anatomy of a Solution

When you open a problem, you typically see a boilerplate snippet like this:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # Write your code here

```

Here is the breakdown of what each part actually means.

### 🏛️ The `class Solution:`

* **What it is:** This is a standard Python **Class**.
* **Why it's there:** LeetCode wraps your code in a class to keep it organized and isolated. Think of it as a container.
* **How it works:** You don't usually need to mess with this line. LeetCode's hidden "driver code" (the code that runs the tests) will create an *instance* of this class (an object) to run your solution.

### 🐍 The `self` Keyword

* **What it is:** In Python object-oriented programming, `self` refers to the **current instance of the class**.
* **Why it's there:** Because the function (`twoSum` in the example above) is defined *inside* a class, it is technically a **method**. Python methods automatically pass the object instance as the first argument.
* **Practical Tip:** You mostly ignore it, but if you define *helper functions* inside the class (e.g., a separate DFS function), you must call them using `self.myHelperFunction()`.

---

## 2. The Arrow `->` and Type Hints

This is often the most confusing part. You might see `-> List[int]` or `: str`. These are called **Type Hints** (introduced in Python 3.5).

### The Colon (`name: type`)

Inside the parentheses, you see things like `nums: List[int]`.

* **Translation:** "Expect the variable `nums` to be a List containing Integers."
* **Does it enforce anything?** **No.** Python is dynamically typed. These are just "hints" for you (the human) and the IDE (the code editor) to know what data is coming in. If you pass a string instead, Python won't crash immediately—until your code tries to treat it like a list.

### The Arrow (`-> type`)

At the end of the definition, you see `-> List[int]`.

* **Translation:** "This function is expected to **return** a List of Integers."
* **Usage:** It tells you what your final `return` statement should look like. If it says `-> bool`, you must return `True` or `False`. If it says `-> None`, you don't return anything (usually strictly modifying an array in-place).

---

## 3. The "Weird" Types You Will See

LeetCode imports the `typing` module by default. Here is a cheat sheet for the common weird words:

| Type Hint | What it means | Example Data |
| --- | --- | --- |
| `List[int]` | A standard Python list containing numbers. | `[1, 2, 3]` |
| `List[str]` | A list of strings. | `["apple", "banana"]` |
| `List[List[int]]` | A 2D array (matrix). | `[[1, 0], [0, 1]]` |
| `Optional[TreeNode]` | **Crucial:** It means the value can be a `TreeNode` **OR** `None`. | A tree root or `None` (empty tree). |
| `Dict[str, int]` | A dictionary with string keys and integer values. | `{"age": 25, "score": 100}` |

---

## 4. The Hidden "Custom Classes"

In problems involving Linked Lists or Binary Trees, you will see types like `ListNode` or `TreeNode`. You cannot just use these directly in a local Python script on your computer without defining them first, but LeetCode defines them for you in the background.

Usually, there is a commented-out section above your code that looks like this:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

```

* **What to make of it:** This tells you the **attributes** you can access.
* **Usage:** You can type `root.val`, `root.left`, or `root.right` because this definition exists in the hidden context.

---

## 5. Summary Checklist

When you look at a new solution window, do this quick mental check:

1. **Check the Arrow (`->`):** What *must* I return? (A number? A list? Nothing?)
2. **Check the Inputs (`:`):** What data structures am I receiving?
3. **Ignore the Boilerplate:** You don't need to change `class Solution` or `self`.
4. **Write Logic:** Focus entirely on the code *inside* the function.
