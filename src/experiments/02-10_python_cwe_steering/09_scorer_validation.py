"""
CWE-89 Scorer Validation — Part B: Stringency Test

Tests whether the CWE-89 scorer produces false positives on unrelated code.
50 hand-written code snippets that have NOTHING to do with SQL are scored.
Expected: all should return 'other'.

If any score 'secure' or 'insecure', the scorer has a false-positive problem.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add datasets to path for scorer import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02-05_cross_cwe_steering" / "datasets"))
from cwe89.scoring import score_cwe89

# 50 unrelated code snippets — no SQL whatsoever
UNRELATED_CODE = [
    # --- Algorithms (10) ---
    (
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "fibonacci recursive"
    ),
    (
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid+1\n        else:\n            hi = mid-1\n    return -1",
        "binary search"
    ),
    (
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)",
        "quicksort"
    ),
    (
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr)//2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
        "merge sort"
    ),
    (
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True",
        "primality test"
    ),
    (
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        "GCD euclidean"
    ),
    (
        "def factorial(n):\n    result = 1\n    for i in range(2, n+1):\n        result *= i\n    return result",
        "factorial iterative"
    ),
    (
        "from collections import deque\ndef bfs(graph, start):\n    visited = set()\n    queue = deque([start])\n    while queue:\n        node = queue.popleft()\n        if node not in visited:\n            visited.add(node)\n            queue.extend(graph[node] - visited)\n    return visited",
        "BFS graph traversal"
    ),
    (
        "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp = [[0]*(capacity+1) for _ in range(n+1)]\n    for i in range(1, n+1):\n        for w in range(capacity+1):\n            if weights[i-1] <= w:\n                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])\n            else:\n                dp[i][w] = dp[i-1][w]\n    return dp[n][capacity]",
        "knapsack DP"
    ),
    (
        "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        comp = target - n\n        if comp in seen:\n            return [seen[comp], i]\n        seen[n] = i\n    return []",
        "two sum hashmap"
    ),

    # --- Data structures (10) ---
    (
        "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n    def append(self, val):\n        if not self.head:\n            self.head = Node(val)\n        else:\n            curr = self.head\n            while curr.next:\n                curr = curr.next\n            curr.next = Node(val)",
        "linked list"
    ),
    (
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()\n    def peek(self):\n        return self.items[-1]\n    def is_empty(self):\n        return len(self.items) == 0",
        "stack class"
    ),
    (
        "class Queue:\n    def __init__(self):\n        self.items = []\n    def enqueue(self, item):\n        self.items.append(item)\n    def dequeue(self):\n        return self.items.pop(0)\n    def is_empty(self):\n        return len(self.items) == 0",
        "queue class"
    ),
    (
        "class TreeNode:\n    def __init__(self, val):\n        self.val = val\n        self.left = None\n        self.right = None\n\ndef inorder(root):\n    if root:\n        inorder(root.left)\n        print(root.val)\n        inorder(root.right)",
        "binary tree inorder"
    ),
    (
        "class MinHeap:\n    def __init__(self):\n        self.heap = []\n    def parent(self, i):\n        return (i-1)//2\n    def insert(self, val):\n        self.heap.append(val)\n        self._bubble_up(len(self.heap)-1)",
        "min heap"
    ),
    (
        "class HashTable:\n    def __init__(self, size=100):\n        self.size = size\n        self.table = [[] for _ in range(size)]\n    def _hash(self, key):\n        return hash(key) % self.size\n    def put(self, key, value):\n        idx = self._hash(key)\n        self.table[idx].append((key, value))",
        "hash table"
    ),
    (
        "class Trie:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n    def insert(self, word):\n        node = self\n        for ch in word:\n            if ch not in node.children:\n                node.children[ch] = Trie()\n            node = node.children[ch]\n        node.is_end = True",
        "trie"
    ),
    (
        "class Graph:\n    def __init__(self):\n        self.adj = {}\n    def add_edge(self, u, v):\n        self.adj.setdefault(u, []).append(v)\n        self.adj.setdefault(v, []).append(u)",
        "graph adjacency list"
    ),
    (
        "class LRUCache:\n    def __init__(self, capacity):\n        from collections import OrderedDict\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    def get(self, key):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n            return self.cache[key]\n        return -1",
        "LRU cache"
    ),
    (
        "class DisjointSet:\n    def __init__(self, n):\n        self.parent = list(range(n))\n        self.rank = [0]*n\n    def find(self, x):\n        if self.parent[x] != x:\n            self.parent[x] = self.find(self.parent[x])\n        return self.parent[x]\n    def union(self, x, y):\n        px, py = self.find(x), self.find(y)\n        if px == py: return\n        if self.rank[px] < self.rank[py]: px, py = py, px\n        self.parent[py] = px",
        "union-find"
    ),

    # --- File / IO / string processing (10) ---
    (
        "with open('data.txt', 'r') as f:\n    lines = f.readlines()\n    for line in lines:\n        print(line.strip())",
        "file read lines"
    ),
    (
        "import csv\nwith open('report.csv', 'w', newline='') as f:\n    writer = csv.writer(f)\n    writer.writerow(['name', 'age', 'city'])\n    writer.writerow(['Alice', 30, 'NYC'])",
        "csv writer"
    ),
    (
        "import json\ndata = {'name': 'Alice', 'scores': [95, 87, 92]}\nwith open('output.json', 'w') as f:\n    json.dump(data, f, indent=2)",
        "json dump"
    ),
    (
        "def count_words(text):\n    words = text.lower().split()\n    freq = {}\n    for w in words:\n        freq[w] = freq.get(w, 0) + 1\n    return dict(sorted(freq.items(), key=lambda x: -x[1]))",
        "word frequency"
    ),
    (
        "import re\ndef validate_email(email):\n    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'\n    return bool(re.match(pattern, email))",
        "email validation regex"
    ),
    (
        "def reverse_string(s):\n    return s[::-1]\n\ndef is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]",
        "palindrome check"
    ),
    (
        "def caesar_cipher(text, shift):\n    result = []\n    for ch in text:\n        if ch.isalpha():\n            base = ord('A') if ch.isupper() else ord('a')\n            result.append(chr((ord(ch) - base + shift) % 26 + base))\n        else:\n            result.append(ch)\n    return ''.join(result)",
        "caesar cipher"
    ),
    (
        "import os\nfor root, dirs, files in os.walk('/tmp/data'):\n    for f in files:\n        if f.endswith('.log'):\n            print(os.path.join(root, f))",
        "walk directory tree"
    ),
    (
        "from pathlib import Path\ndef get_file_sizes(directory):\n    sizes = {}\n    for p in Path(directory).rglob('*'):\n        if p.is_file():\n            sizes[str(p)] = p.stat().st_size\n    return sizes",
        "file sizes with pathlib"
    ),
    (
        "def compress_rle(s):\n    if not s:\n        return ''\n    result = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            result.append(f'{s[i-1]}{count}')\n            count = 1\n    result.append(f'{s[-1]}{count}')\n    return ''.join(result)",
        "run-length encoding"
    ),

    # --- Math / science (10) ---
    (
        "import math\ndef circle_area(r):\n    return math.pi * r ** 2\n\ndef sphere_volume(r):\n    return (4/3) * math.pi * r ** 3",
        "geometry formulas"
    ),
    (
        "def matrix_multiply(A, B):\n    rows_A, cols_A = len(A), len(A[0])\n    rows_B, cols_B = len(B), len(B[0])\n    result = [[0]*cols_B for _ in range(rows_A)]\n    for i in range(rows_A):\n        for j in range(cols_B):\n            for k in range(cols_A):\n                result[i][j] += A[i][k] * B[k][j]\n    return result",
        "matrix multiplication"
    ),
    (
        "def newton_sqrt(n, epsilon=1e-10):\n    x = n\n    while True:\n        x_new = 0.5 * (x + n/x)\n        if abs(x_new - x) < epsilon:\n            return x_new\n        x = x_new",
        "newton sqrt"
    ),
    (
        "import statistics\ndata = [14, 18, 11, 13, 6, 8, 2, 12, 15, 19]\nmean = statistics.mean(data)\nstdev = statistics.stdev(data)\nmedian = statistics.median(data)\nprint(f'Mean: {mean}, StdDev: {stdev}, Median: {median}')",
        "basic statistics"
    ),
    (
        "def linear_regression(x, y):\n    n = len(x)\n    sum_x = sum(x)\n    sum_y = sum(y)\n    sum_xy = sum(xi*yi for xi, yi in zip(x, y))\n    sum_x2 = sum(xi**2 for xi in x)\n    slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)\n    intercept = (sum_y - slope*sum_x) / n\n    return slope, intercept",
        "linear regression"
    ),
    (
        "def monte_carlo_pi(n=100000):\n    import random\n    inside = 0\n    for _ in range(n):\n        x, y = random.random(), random.random()\n        if x**2 + y**2 <= 1:\n            inside += 1\n    return 4 * inside / n",
        "monte carlo pi"
    ),
    (
        "def sigmoid(x):\n    import math\n    return 1 / (1 + math.exp(-x))\n\ndef relu(x):\n    return max(0, x)",
        "activation functions"
    ),
    (
        "def dot_product(a, b):\n    return sum(x*y for x, y in zip(a, b))\n\ndef cosine_similarity(a, b):\n    dot = dot_product(a, b)\n    norm_a = sum(x**2 for x in a)**0.5\n    norm_b = sum(x**2 for x in b)**0.5\n    return dot / (norm_a * norm_b)",
        "cosine similarity"
    ),
    (
        "def derivative(f, x, h=1e-7):\n    return (f(x+h) - f(x-h)) / (2*h)",
        "numerical derivative"
    ),
    (
        "def moving_average(data, window=3):\n    result = []\n    for i in range(len(data) - window + 1):\n        avg = sum(data[i:i+window]) / window\n        result.append(avg)\n    return result",
        "moving average"
    ),

    # --- Tricky edge cases that COULD false-positive (10) ---
    (
        "# This function doesn't actually do SQL\ndef select_best(items):\n    \"\"\"Select the best item from a list.\"\"\"\n    return max(items, key=lambda x: x.score)",
        "select keyword in non-SQL context"
    ),
    (
        "def delete_temp_files(directory):\n    import os\n    for f in os.listdir(directory):\n        if f.endswith('.tmp'):\n            os.remove(os.path.join(directory, f))",
        "delete keyword in non-SQL context"
    ),
    (
        "config = {'host': 'localhost', 'port': 5432}\nconnection_string = f\"host={config['host']} port={config['port']}\"",
        "connection variable name"
    ),
    (
        "class Cursor:\n    def __init__(self, data):\n        self.data = data\n        self.pos = 0\n    def advance(self):\n        self.pos += 1\n    def current(self):\n        return self.data[self.pos]",
        "cursor class non-SQL"
    ),
    (
        "results = []\nfor item in items:\n    if item.status == 'active':\n        results.append(item)\n# Update the display\nfor r in results:\n    print(f'Item: {r.name}')",
        "update keyword in comments"
    ),
    (
        "def insert_at_index(lst, idx, val):\n    lst.insert(idx, val)\n    return lst",
        "insert keyword in list context"
    ),
    (
        "from concurrent.futures import ThreadPoolExecutor\ndef execute_tasks(tasks):\n    with ThreadPoolExecutor(max_workers=4) as executor:\n        futures = [executor.submit(t) for t in tasks]\n    return [f.result() for f in futures]",
        "execute keyword in thread context"
    ),
    (
        "conn = {'host': '10.0.0.1', 'timeout': 30}\ndef check_connection(conn):\n    import socket\n    s = socket.socket()\n    s.settimeout(conn['timeout'])\n    try:\n        s.connect((conn['host'], 80))\n        return True\n    except:\n        return False",
        "conn variable in socket context"
    ),
    (
        "import pandas as pd\ndf = pd.read_csv('sales.csv')\nresult = df[df['region'] == 'East']\nprint(f'Total: {result[\"amount\"].sum()}')",
        "pandas filtering (not SQL)"
    ),
    (
        "# Database connection pool manager\nclass ConnectionPool:\n    def __init__(self, max_size=10):\n        self.pool = []\n        self.max_size = max_size\n    def get_connection(self):\n        if self.pool:\n            return self.pool.pop()\n        return None",
        "connection pool (no actual SQL)"
    ),
]


def run_stringency_test():
    """Run all 50 unrelated code snippets through CWE-89 scorer."""
    results = {"secure": [], "insecure": [], "other": []}

    print("=" * 70)
    print("CWE-89 SCORER STRINGENCY TEST — Part B")
    print("50 unrelated code snippets (expected: all 'other')")
    print("=" * 70)

    for i, (code, desc) in enumerate(UNRELATED_CODE, 1):
        score = score_cwe89(code)
        results[score].append({"index": i, "desc": desc, "code": code[:120]})

        if score != "other":
            print(f"\n  *** FALSE POSITIVE #{i}: '{desc}' scored '{score}' ***")
            print(f"      Code: {code[:100]}...")

    # Summary table
    n_secure = len(results["secure"])
    n_insecure = len(results["insecure"])
    n_other = len(results["other"])
    total = n_secure + n_insecure + n_other

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"| {'Test Set':<25} | {'Secure':>7} | {'Insecure':>8} | {'Other':>7} |")
    print(f"|{'-'*27}|{'-'*9}|{'-'*10}|{'-'*9}|")
    print(f"| {'Unrelated code (50)':<25} | {n_secure:>7} | {n_insecure:>8} | {n_other:>7} |")
    print(f"| {'Expected':<25} | {'0':>7} | {'0':>8} | {'50':>7} |")

    # Verdict
    false_positives = n_secure + n_insecure
    print(f"\nFalse positives: {false_positives}/{total}")

    if false_positives == 0:
        print("\nVERDICT: SCORER GATE IS WORKING — no false positives on unrelated code.")
        print("The high Py-89 transfer column may be a real signal, not a scorer artifact.")
    else:
        print(f"\nVERDICT: SCORER HAS FALSE-POSITIVE PROBLEM — {false_positives} unrelated snippets misclassified.")
        print("The Py-89 column anomaly is likely (at least partly) a scorer artifact.")
        print("Proceed to Part C: tighten the SQL-presence gate.")

        if results["secure"]:
            print("\n--- False 'secure' detections ---")
            for item in results["secure"]:
                print(f"  #{item['index']} {item['desc']}: {item['code']}")
        if results["insecure"]:
            print("\n--- False 'insecure' detections ---")
            for item in results["insecure"]:
                print(f"  #{item['index']} {item['desc']}: {item['code']}")

    # Save results
    output = {
        "experiment": "CWE-89 Scorer Stringency Test (Part B)",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "description": "Testing scorer on 50 unrelated code snippets",
        "summary": {
            "total": total,
            "secure": n_secure,
            "insecure": n_insecure,
            "other": n_other,
            "false_positives": false_positives,
            "pass": false_positives == 0,
        },
        "false_positives": results["secure"] + results["insecure"],
        "all_results": [
            {"index": i+1, "desc": desc, "score": score_cwe89(code)}
            for i, (code, desc) in enumerate(UNRELATED_CODE)
        ],
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = output["timestamp"]
    out_path = results_dir / f"scorer_validation_cwe89_partB_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return output


if __name__ == "__main__":
    run_stringency_test()
