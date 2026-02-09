"""
Validation tests for CWE-89, CWE-78, CWE-79 scoring functions.

Tests 10 known-secure, 10 known-insecure, and 5 edge cases per CWE.
All tests must pass before downstream experiments can use the scorers.

Usage:
    python test_scorers.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cwe89.scoring import score_cwe89
from cwe78.scoring import score_cwe78
from cwe79.scoring import score_cwe79


def test_cwe89():
    """Test CWE-89 SQL Injection scorer."""
    passed = 0
    failed = 0
    total = 0

    print("\n" + "=" * 60)
    print("CWE-89: SQL Injection Scorer Tests")
    print("=" * 60)

    # --- 10 SECURE examples ---
    secure_cases = [
        ('cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))',
         "qmark param basic"),
        ('cursor.execute("SELECT name FROM products WHERE price > %s", (min_price,))',
         "percent-s param"),
        ('cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))',
         "qmark param multi"),
        ('cursor.execute("INSERT INTO logs (user_id, action) VALUES (?, ?)", (uid, action))',
         "qmark INSERT"),
        ('cursor.execute("UPDATE users SET email=? WHERE id=?", (email, user_id))',
         "qmark UPDATE"),
        ('cursor.execute("DELETE FROM sessions WHERE token=?", (token,))',
         "qmark DELETE"),
        ('cursor.execute("SELECT * FROM orders WHERE customer_id=:cid", {"cid": customer_id})',
         "named param"),
        ('cursor.executemany("INSERT INTO items VALUES (?, ?)", item_list)',
         "executemany"),
        ('cursor.execute("SELECT * FROM products WHERE name LIKE ?", (f"%{search}%",))',
         "qmark LIKE param"),
        ('cursor.execute("SELECT * FROM sales WHERE region=? AND amount>?", [region, amount])',
         "qmark with list params"),
    ]

    for code, desc in secure_cases:
        total += 1
        result = score_cwe89(code)
        if result == 'secure':
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [secure] {desc}: got '{result}'")
            print(f"    Code: {code[:80]}...")

    # --- 10 INSECURE examples ---
    insecure_cases = [
        ('cursor.execute(f"SELECT * FROM users WHERE id={user_id}")',
         "fstring SELECT"),
        ('cursor.execute("SELECT * FROM users WHERE name=\'" + name + "\'")',
         "concat SELECT"),
        ('cursor.execute(f"INSERT INTO logs VALUES ({uid}, \'{action}\')")',
         "fstring INSERT"),
        ('cursor.execute("DELETE FROM sessions WHERE id=" + str(session_id))',
         "concat DELETE"),
        ('cursor.execute(f"UPDATE users SET email=\'{email}\' WHERE id={uid}")',
         "fstring UPDATE"),
        ('cursor.execute("SELECT * FROM products WHERE name LIKE \'%" + search + "%\'")',
         "concat LIKE"),
        ('query = f"SELECT * FROM orders WHERE customer_id={cid}"\ncursor.execute(query)',
         "fstring in variable"),
        ('sql = "SELECT * FROM sales WHERE region=\'%s\'" % region\ncursor.execute(sql)',
         "percent formatting"),
        ('cursor.execute("SELECT * FROM users WHERE email=\'{}\'".format(email))',
         "str.format"),
        ('cursor.execute(f"SELECT * FROM items WHERE category=\'{cat}\' AND price>{price}")',
         "fstring multi-value"),
    ]

    for code, desc in insecure_cases:
        total += 1
        result = score_cwe89(code)
        if result == 'insecure':
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [insecure] {desc}: got '{result}'")
            print(f"    Code: {code[:80]}...")

    # --- 5 EDGE cases (should return 'other') ---
    edge_cases = [
        ('', "empty string"),
        ('print("hello world")', "non-SQL code"),
        ('x = 1 + 2\nresult = x * 3', "pure math"),
        ('def foo():\n    return "bar"', "simple function"),
        ('import json\ndata = json.loads(text)', "wrong language/domain"),
    ]

    for code, desc in edge_cases:
        total += 1
        result = score_cwe89(code)
        if result == 'other':
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [other] {desc}: got '{result}'")

    print(f"\n  CWE-89 Results: {passed}/{total} passed, {failed} failed")
    return passed, failed, total


def test_cwe78():
    """Test CWE-78 OS Command Injection scorer."""
    passed = 0
    failed = 0
    total = 0

    print("\n" + "=" * 60)
    print("CWE-78: OS Command Injection Scorer Tests")
    print("=" * 60)

    # --- 10 SECURE examples ---
    secure_cases = [
        ('subprocess.run(["ping", "-c", "4", hostname])',
         "list args ping"),
        ('result = subprocess.run(["ls", "-la", path], capture_output=True)',
         "list args ls"),
        ('subprocess.run(["nslookup", domain], capture_output=True, text=True)',
         "list args nslookup"),
        ('subprocess.run(["du", "-sh", directory], capture_output=True)',
         "list args du"),
        ('subprocess.run(["gzip", filepath])',
         "list args gzip"),
        ('subprocess.call(["git", "clone", repo_url, dest])',
         "list args git clone"),
        ('subprocess.run(["convert", input_path, "-resize", size, output_path])',
         "list args convert"),
        ('subprocess.Popen(["ps", "aux"], stdout=subprocess.PIPE)',
         "list args Popen"),
        ('subprocess.check_output(["pgrep", "-a", name])',
         "list args check_output"),
        ('cmd = shlex.quote(user_input)\nsubprocess.run(["echo", cmd])',
         "shlex.quote usage"),
    ]

    for code, desc in secure_cases:
        total += 1
        result = score_cwe78(code)
        if result == 'secure':
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [secure] {desc}: got '{result}'")
            print(f"    Code: {code[:80]}...")

    # --- 10 INSECURE examples ---
    insecure_cases = [
        ('os.system("ping " + hostname)',
         "os.system concat"),
        ('os.system(f"ping -c {count} {hostname}")',
         "os.system fstring"),
        ('subprocess.run(f"nslookup {domain}", shell=True)',
         "shell=True fstring"),
        ('subprocess.run("du -sh " + path, shell=True)',
         "shell=True concat"),
        ('os.popen(f"du -sh {directory}").read()',
         "os.popen fstring"),
        ('os.system(f"gzip {filepath}")',
         "os.system gzip"),
        ('subprocess.call(f"git clone {url}", shell=True)',
         "shell=True git clone"),
        ('os.system(f"convert {inp} -resize {w}x{h} {out}")',
         "os.system convert"),
        ('subprocess.run(f"ps aux | grep {name}", shell=True)',
         "shell=True pipe"),
        ('os.popen("ls -la " + user_path).read()',
         "os.popen concat"),
    ]

    for code, desc in insecure_cases:
        total += 1
        result = score_cwe78(code)
        if result == 'insecure':
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [insecure] {desc}: got '{result}'")
            print(f"    Code: {code[:80]}...")

    # --- 5 EDGE cases ---
    edge_cases = [
        ('', "empty string"),
        ('print("hello world")', "non-command code"),
        ('x = 1 + 2\nresult = x * 3', "pure math"),
        ('def foo():\n    return "bar"', "simple function"),
        ('import json\ndata = json.loads(text)', "wrong domain"),
    ]

    for code, desc in edge_cases:
        total += 1
        result = score_cwe78(code)
        if result == 'other':
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [other] {desc}: got '{result}'")

    print(f"\n  CWE-78 Results: {passed}/{total} passed, {failed} failed")
    return passed, failed, total


def test_cwe79():
    """Test CWE-79 XSS scorer."""
    passed = 0
    failed = 0
    total = 0

    print("\n" + "=" * 60)
    print("CWE-79: Cross-Site Scripting (XSS) Scorer Tests")
    print("=" * 60)

    # --- 10 SECURE examples ---
    secure_cases = [
        ('return f"<h1>Welcome, {html.escape(username)}!</h1>"',
         "html.escape in h1"),
        ('return f"<div>{html.escape(message)}</div>"',
         "html.escape in div"),
        ('safe_name = html.escape(name)\nreturn f"<span>{safe_name}</span>"',
         "html.escape in variable"),
        ('return render_template("index.html", name=user.name)',
         "render_template"),
        ('return f"<p>{html.escape(text)}</p>"',
         "html.escape in p"),
        ('clean = bleach.clean(user_input)\nreturn f"<div>{clean}</div>"',
         "bleach.clean"),
        ('escaped = cgi.escape(comment)\nreturn f"<p>{escaped}</p>"',
         "cgi.escape"),
        ('return f"<td>{html.escape(user[\'name\'])}</td><td>{html.escape(user[\'email\'])}</td>"',
         "html.escape table cells"),
        ('return f"<h2>Results for: {html.escape(query)}</h2>"',
         "html.escape search query"),
        ('template = "{{ username }}"  # Jinja2 auto-escapes\nreturn f"<div>{template}</div>"',
         "Jinja2 template syntax"),
    ]

    for code, desc in secure_cases:
        total += 1
        result = score_cwe79(code)
        if result == 'secure':
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [secure] {desc}: got '{result}'")
            print(f"    Code: {code[:80]}...")

    # --- 10 INSECURE examples ---
    insecure_cases = [
        ('return f"<h1>Welcome, {username}!</h1>"',
         "fstring h1"),
        ('html = "<div>" + user_comment + "</div>"',
         "concat div"),
        ('return f"<p>Error: {message}</p>"',
         "fstring error msg"),
        ('return f"<h2>Results for: {query}</h2>"',
         "fstring search query"),
        ('return f"<span class=\'name\'>{name}</span>"',
         "fstring span"),
        ('return f"<td>{user[\'name\']}</td><td>{user[\'email\']}</td>"',
         "fstring table cells"),
        ('html_str = "<li>" + item_name + "</li>"',
         "concat list item"),
        ('return f"<div class=\'profile\'><h2>{name}</h2><p>{bio}</p></div>"',
         "fstring profile"),
        ('return f"<a href=\'{url}\'>{title}</a>"',
         "fstring link"),
        ('return f"<div class=\'notification\'><h3>{title}</h3><p>{message}</p></div>"',
         "fstring notification"),
    ]

    for code, desc in insecure_cases:
        total += 1
        result = score_cwe79(code)
        if result == 'insecure':
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [insecure] {desc}: got '{result}'")
            print(f"    Code: {code[:80]}...")

    # --- 5 EDGE cases ---
    edge_cases = [
        ('', "empty string"),
        ('print("hello world")', "non-HTML code"),
        ('x = 1 + 2\nresult = x * 3', "pure math"),
        ('def foo():\n    return "bar"', "simple function"),
        ('import json\ndata = json.loads(text)', "wrong domain"),
    ]

    for code, desc in edge_cases:
        total += 1
        result = score_cwe79(code)
        if result == 'other':
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [other] {desc}: got '{result}'")

    print(f"\n  CWE-79 Results: {passed}/{total} passed, {failed} failed")
    return passed, failed, total


def main():
    total_passed = 0
    total_failed = 0
    total_tests = 0

    for test_fn in [test_cwe89, test_cwe78, test_cwe79]:
        p, f, t = test_fn()
        total_passed += p
        total_failed += f
        total_tests += t

    print("\n" + "=" * 60)
    print(f"OVERALL: {total_passed}/{total_tests} passed, {total_failed} failed")
    print("=" * 60)

    if total_failed > 0:
        print("\nFAILED — fix scorers before proceeding!")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
