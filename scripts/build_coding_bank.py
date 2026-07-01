"""Solver-driven authoring tool for the coding-question bank.

Each problem is defined ONCE here: metadata + dual-language starter scaffolds +
a reference `solve` (the oracle) + curated visible inputs + a random `gen` for
hidden fuzz cases. The tool then COMPUTES every expected output by running the
oracle, so test answers are correct by construction — no hand-typed expecteds.

    python -m scripts.build_coding_bank            # dry run: print what it'd add
    python -m scripts.build_coding_bank --write    # merge new problems into the bank

Existing problems already in coding_questions.json are left untouched; a problem
here whose `id` already exists is skipped (so this only ever ADDS).

Only clean stdin/stdout problems live here (array/string/hashmap/stack/math/
two-pointer). Linked-list / tree / design problems are intentionally excluded.

Conventions:
  - booleans print as "true" / "false"
  - integer arrays print space-separated on one line
  - comparison is whitespace-trimmed (matches backend _run_tests)
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable

BANK = os.path.join(os.path.dirname(__file__), "..", "ai_interviewer", "data", "coding_questions.json")
N_HIDDEN = 12  # fuzz cases per problem, on top of the curated visible/edge cases


@dataclass
class Problem:
    id: str
    title: str
    difficulty: str          # easy | medium
    prompt: str              # shown in the editor panel
    spoken_intro: str        # read aloud (shorter, conversational)
    starter: dict            # {python, c++} input-reading scaffold
    visible: list            # [(name, stdin)] cases shown to the candidate
    gen: Callable[[random.Random], str]   # rng -> a random stdin
    solve: Callable[[str], str]           # stdin -> expected stdout (the oracle)
    n_hidden: int = N_HIDDEN


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
          (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
          (5, "V"), (4, "IV"), (1, "I")]
_ROMAN_VAL = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _int_to_roman(n: int) -> str:
    out = []
    for v, sym in _ROMAN:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def _roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s.strip()):
        v = _ROMAN_VAL[ch]
        total += v if v >= prev else -v
        prev = v
    return total


def _ints(line: str) -> list:
    return list(map(int, line.split()))


def _first_line(stdin: str) -> str:
    # the first line, preserving internal/trailing spaces (only the newline drops)
    return stdin.split("\n", 1)[0]


# ---------------------------------------------------------------------------
# solvers (the oracles) and generators
# ---------------------------------------------------------------------------

def _solve_roman(stdin):
    return str(_roman_to_int(_first_line(stdin)))


def _gen_roman(rng):
    return _int_to_roman(rng.randint(1, 3999)) + "\n"


def _solve_last_word(stdin):
    words = _first_line(stdin).split()
    return str(len(words[-1]) if words else 0)


def _gen_last_word(rng):
    words = ["".join(rng.choice("abcde") for _ in range(rng.randint(1, 6)))
             for _ in range(rng.randint(1, 5))]
    return " ".join(words) + " " * rng.randint(0, 3) + "\n"


def _lcp(words):
    if not words:
        return ""
    pre = words[0]
    for w in words[1:]:
        while not w.startswith(pre):
            pre = pre[:-1]
            if not pre:
                return ""
    return pre


def _solve_lcp(stdin):
    return _lcp(_first_line(stdin).split())


def _gen_lcp(rng):
    pre = "".join(rng.choice("abc") for _ in range(rng.randint(0, 3)))
    words = [pre + "".join(rng.choice("abc") for _ in range(rng.randint(0, 3)))
             for _ in range(rng.randint(1, 4))]
    return " ".join(words) + "\n"


def _solve_palindrome(stdin):
    f = [c.lower() for c in _first_line(stdin) if c.isalnum()]
    return "true" if f == f[::-1] else "false"


def _gen_palindrome(rng):
    half = "".join(rng.choice("abAB12 ,.") for _ in range(rng.randint(0, 6)))
    if rng.random() < 0.5:
        # bias toward true palindromes so both branches get covered
        core = [c for c in half if c.isalnum()]
        return half + "".join(reversed(core)) + "\n"
    return half + "".join(rng.choice("abAB12 ,.") for _ in range(rng.randint(0, 6))) + "\n"


def _solve_is_subsequence(stdin):
    lines = stdin.split("\n")
    s, t = lines[0], lines[1] if len(lines) > 1 else ""
    it = iter(t)
    return "true" if all(c in it for c in s) else "false"


def _gen_is_subsequence(rng):
    t = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 10)))
    if rng.random() < 0.5 and t:
        idx = sorted(rng.sample(range(len(t)), rng.randint(0, len(t))))
        s = "".join(t[i] for i in idx)
    else:
        s = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 5)))
    return f"{s}\n{t}\n"


def _solve_majority(stdin):
    return str(Counter(_ints(_first_line(stdin))).most_common(1)[0][0])


def _gen_majority(rng):
    n = rng.randint(1, 9)
    maj = rng.randint(0, 5)
    k = n // 2 + 1
    arr = [maj] * k + [rng.randint(0, 5) for _ in range(n - k)]
    rng.shuffle(arr)
    return " ".join(map(str, arr)) + "\n"


def _solve_jump_game(stdin):
    nums = _ints(_first_line(stdin))
    reach = 0
    for i, n in enumerate(nums):
        if i > reach:
            return "false"
        reach = max(reach, i + n)
    return "true"


def _gen_jump_game(rng):
    return " ".join(str(rng.randint(0, 4)) for _ in range(rng.randint(1, 10))) + "\n"


def _solve_product_except_self(stdin):
    nums = _ints(_first_line(stdin))
    n = len(nums)
    out = [1] * n
    pre = 1
    for i in range(n):
        out[i] = pre
        pre *= nums[i]
    suf = 1
    for i in range(n - 1, -1, -1):
        out[i] *= suf
        suf *= nums[i]
    return " ".join(map(str, out))


def _gen_product_except_self(rng):
    return " ".join(str(rng.randint(-3, 3)) for _ in range(rng.randint(2, 7))) + "\n"


def _solve_stock_ii(stdin):
    prices = _ints(_first_line(stdin))
    return str(sum(max(0, b - a) for a, b in zip(prices, prices[1:])))


def _gen_stock_ii(rng):
    return " ".join(str(rng.randint(1, 12)) for _ in range(rng.randint(1, 10))) + "\n"


def _solve_search_insert(stdin):
    lines = stdin.split("\n")
    nums = _ints(lines[0])
    target = int(lines[1])
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return str(lo)


def _gen_search_insert(rng):
    n = rng.randint(1, 8)
    nums = sorted(rng.sample(range(-10, 20), n))
    target = rng.randint(-12, 22)
    return f"{' '.join(map(str, nums))}\n{target}\n"


def _two_lines(stdin):
    parts = stdin.split("\n")
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _solve_add_binary(stdin):
    a, b = _two_lines(stdin)
    return bin(int(a.strip() or "0", 2) + int(b.strip() or "0", 2))[2:]


def _gen_add_binary(rng):
    def b():
        L = rng.randint(1, 10)
        return rng.choice("01") if L == 1 else "1" + "".join(rng.choice("01") for _ in range(L - 1))
    return f"{b()}\n{b()}\n"


def _solve_plus_one(stdin):
    digits = _ints(_first_line(stdin))
    n = int("".join(map(str, digits))) + 1
    return " ".join(str(n))


def _gen_plus_one(rng):
    L = rng.randint(1, 6)
    d = [rng.randint(0, 9)] if L == 1 else [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(L - 1)]
    return " ".join(map(str, d)) + "\n"


def _solve_anagram(stdin):
    s, t = _two_lines(stdin)
    return "true" if sorted(s) == sorted(t) else "false"


def _gen_anagram(rng):
    s = "".join(rng.choice("abc") for _ in range(rng.randint(1, 6)))
    if rng.random() < 0.5:
        t = list(s)
        rng.shuffle(t)
        t = "".join(t)
    else:
        t = "".join(rng.choice("abc") for _ in range(rng.randint(1, 6)))
    return f"{s}\n{t}\n"


def _solve_ransom(stdin):
    note, mag = _two_lines(stdin)
    cn, cm = Counter(note), Counter(mag)
    return "true" if all(cm[k] >= v for k, v in cn.items()) else "false"


def _gen_ransom(rng):
    note = "".join(rng.choice("abc") for _ in range(rng.randint(1, 5)))
    mag = "".join(rng.choice("abc") for _ in range(rng.randint(1, 8)))
    return f"{note}\n{mag}\n"


def _solve_word_pattern(stdin):
    pattern, words_line = _two_lines(stdin)
    pattern, words = pattern.strip(), words_line.split()
    if len(pattern) != len(words):
        return "false"
    p2w, w2p = {}, {}
    for c, w in zip(pattern, words):
        if p2w.setdefault(c, w) != w or w2p.setdefault(w, c) != c:
            return "false"
    return "true"


def _gen_word_pattern(rng):
    letters = "abc"
    n = rng.randint(1, 4)
    pattern = "".join(rng.choice(letters) for _ in range(n))
    vocab = ["dog", "cat", "fish", "bird"]
    if rng.random() < 0.5:  # consistent mapping (likely true)
        mp = {c: rng.choice(vocab) for c in set(pattern)}
        words = [mp[c] for c in pattern]
    else:
        words = [rng.choice(vocab) for _ in range(n)]
    return f"{pattern}\n{' '.join(words)}\n"


def _solve_summary_ranges(stdin):
    nums = _ints(_first_line(stdin))
    res, i, n = [], 0, len(nums)
    while i < n:
        j = i
        while j + 1 < n and nums[j + 1] == nums[j] + 1:
            j += 1
        res.append(str(nums[i]) if i == j else f"{nums[i]}->{nums[j]}")
        i = j + 1
    return " ".join(res)


def _gen_summary_ranges(rng):
    n = rng.randint(0, 8)
    nums = sorted(rng.sample(range(-5, 15), n))
    return " ".join(map(str, nums)) + "\n"


def _solve_climb(stdin):
    n = int(_first_line(stdin))
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return str(a)


def _gen_climb(rng):
    return f"{rng.randint(1, 30)}\n"


def _solve_single(stdin):
    r = 0
    for x in _ints(_first_line(stdin)):
        r ^= x
    return str(r)


def _gen_single(rng):
    k = rng.randint(0, 4)
    vals = rng.sample(range(0, 30), k + 1)
    arr = [vals[0]] + [v for v in vals[1:] for _ in range(2)]
    rng.shuffle(arr)
    return " ".join(map(str, arr)) + "\n"


def _solve_two_sum_ii(stdin):
    nums_line, target_line = _two_lines(stdin)
    nums, target = _ints(nums_line), int(target_line)
    i, j = 0, len(nums) - 1
    while i < j:
        s = nums[i] + nums[j]
        if s == target:
            return f"{i + 1} {j + 1}"
        i, j = (i + 1, j) if s < target else (i, j - 1)
    return ""


def _gen_two_sum_ii(rng):
    while True:
        n = rng.randint(2, 8)
        nums = sorted(rng.sample(range(-10, 20), n))
        i, j = sorted(rng.sample(range(n), 2))
        target = nums[i] + nums[j]
        # keep only inputs with a unique pair, so the two-pointer answer is canonical
        cnt = sum(1 for a in range(n) for b in range(a + 1, n) if nums[a] + nums[b] == target)
        if cnt == 1:
            return f"{' '.join(map(str, nums))}\n{target}\n"


def _solve_happy(stdin):
    n, seen = int(_first_line(stdin)), set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d) ** 2 for d in str(n))
    return "true" if n == 1 else "false"


def _gen_happy(rng):
    return f"{rng.randint(1, 200)}\n"


# ----- batch 3 solvers/generators (medium) -----
# Every problem here has a single unambiguous correct output (a unique int,
# count, index, or canonical string) so the exact-match grader stays fair.

def _solve_max_subarray(stdin):
    nums = _ints(_first_line(stdin))
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return str(best)


def _gen_max_subarray(rng):
    return " ".join(str(rng.randint(-5, 5)) for _ in range(rng.randint(1, 10))) + "\n"


def _solve_longest_unique(stdin):
    s = _first_line(stdin)
    last, start, best = {}, 0, 0
    for i, c in enumerate(s):
        if c in last and last[c] >= start:
            start = last[c] + 1
        last[c] = i
        best = max(best, i - start + 1)
    return str(best)


def _gen_longest_unique(rng):
    return "".join(rng.choice("abcd") for _ in range(rng.randint(0, 12))) + "\n"


def _solve_max_area(stdin):
    h = _ints(_first_line(stdin))
    i, j, best = 0, len(h) - 1, 0
    while i < j:
        best = max(best, (j - i) * min(h[i], h[j]))
        if h[i] < h[j]:
            i += 1
        else:
            j -= 1
    return str(best)


def _gen_max_area(rng):
    return " ".join(str(rng.randint(0, 9)) for _ in range(rng.randint(2, 10))) + "\n"


def _solve_coin_change(stdin):
    lines = stdin.split("\n")
    coins, amount = _ints(lines[0]), int(lines[1])
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return str(dp[amount] if dp[amount] != INF else -1)


def _gen_coin_change(rng):
    coins = sorted(rng.sample([1, 2, 3, 5, 7, 10], rng.randint(1, 3)))
    return f"{' '.join(map(str, coins))}\n{rng.randint(0, 25)}\n"


def _solve_house_robber(stdin):
    prev = cur = 0
    for x in _ints(_first_line(stdin)):
        prev, cur = cur, max(cur, prev + x)
    return str(cur)


def _gen_house_robber(rng):
    return " ".join(str(rng.randint(0, 10)) for _ in range(rng.randint(1, 10))) + "\n"


def _solve_unique_paths(stdin):
    m, n = _ints(_first_line(stdin))
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return str(dp[-1])


def _gen_unique_paths(rng):
    return f"{rng.randint(1, 8)} {rng.randint(1, 8)}\n"


def _solve_decode_ways(stdin):
    s = _first_line(stdin).strip()
    if not s:
        return "0"
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 0 if s[0] == "0" else 1
    for i in range(2, n + 1):
        if s[i - 1] != "0":
            dp[i] += dp[i - 1]
        if 10 <= int(s[i - 2:i]) <= 26:
            dp[i] += dp[i - 2]
    return str(dp[n])


def _gen_decode_ways(rng):
    return "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 6))) + "\n"


def _solve_max_product(stdin):
    nums = _ints(_first_line(stdin))
    best = hi = lo = nums[0]
    for x in nums[1:]:
        cands = (x, hi * x, lo * x)
        hi, lo = max(cands), min(cands)
        best = max(best, hi)
    return str(best)


def _gen_max_product(rng):
    return " ".join(str(rng.randint(-3, 3)) for _ in range(rng.randint(1, 8))) + "\n"


def _solve_search_rotated(stdin):
    lines = stdin.split("\n")
    nums, target = _ints(lines[0]), int(lines[1])
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return str(mid)
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return "-1"


def _gen_search_rotated(rng):
    n = rng.randint(1, 8)
    base = sorted(rng.sample(range(-10, 20), n))
    k = rng.randint(0, n - 1)
    nums = base[k:] + base[:k]
    target = rng.choice(nums) if rng.random() < 0.6 else rng.randint(-12, 22)
    return f"{' '.join(map(str, nums))}\n{target}\n"


# ---------------------------------------------------------------------------
# batch 1 — 10 clean stdin/stdout problems from the Top Interview 150
# ---------------------------------------------------------------------------

PROBLEMS: list[Problem] = [
    Problem(
        id="roman-to-integer",
        title="Roman to Integer",
        difficulty="easy",
        prompt=("Read a line containing a Roman numeral (I, V, X, L, C, D, M) and print "
                "its integer value.\n\nInput:\n  MCMXCIV\nOutput:\n  1994"),
        spoken_intro=("Here's a parsing one. You'll get a Roman numeral as a string, and you "
                      "print the integer it represents. Run the tests and submit when ready."),
        starter={
            "python": "s = input().strip()\n\n# TODO: print the integer value of the Roman numeral\n",
            "c++": ("#include <iostream>\n#include <string>\nusing namespace std;\n\n"
                    "int main() {\n    string s;\n    getline(cin, s);\n\n"
                    "    // TODO: print the integer value of the Roman numeral s\n    return 0;\n}\n"),
        },
        visible=[("example", "MCMXCIV\n"), ("simple", "III\n"), ("subtractive", "IV\n")],
        gen=_gen_roman, solve=_solve_roman,
    ),
    Problem(
        id="length-of-last-word",
        title="Length of Last Word",
        difficulty="easy",
        prompt=("Read a line of words separated by spaces (there may be trailing spaces). "
                "Print the length of the last word.\n\nInput:\n  Hello World\nOutput:\n  5"),
        spoken_intro=("A quick string one. Given a line of words, print the length of the last "
                      "word. Watch out for trailing spaces. Run the tests and submit when ready."),
        starter={
            "python": "line = input()\n\n# TODO: print the length of the last word\n",
            "c++": ("#include <iostream>\n#include <string>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n\n"
                    "    // TODO: print the length of the last word\n    return 0;\n}\n"),
        },
        visible=[("example", "Hello World\n"), ("trailing spaces", "   fly me   to   the moon  \n"),
                 ("single word", "luffy\n")],
        gen=_gen_last_word, solve=_solve_last_word,
    ),
    Problem(
        id="longest-common-prefix",
        title="Longest Common Prefix",
        difficulty="easy",
        prompt=("Read a line of space-separated lowercase words. Print the longest common "
                "prefix shared by all of them, or an empty line if there is none.\n\n"
                "Input:\n  flower flow flight\nOutput:\n  fl"),
        spoken_intro=("Given a few words, find the longest starting string they all share. "
                      "Print it, or an empty line if there's no common prefix. Submit when ready."),
        starter={
            "python": "words = input().split()\n\n# TODO: print the longest common prefix (empty line if none)\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\n#include <string>\n"
                    "using namespace std;\n\nint main() {\n    string line;\n    getline(cin, line);\n"
                    "    vector<string> words;\n    stringstream ss(line);\n    string w;\n"
                    "    while (ss >> w) words.push_back(w);\n\n"
                    "    // TODO: print the longest common prefix (empty line if none)\n    return 0;\n}\n"),
        },
        visible=[("example", "flower flow flight\n"), ("no prefix", "dog racecar car\n"),
                 ("identical", "abc abc abc\n")],
        gen=_gen_lcp, solve=_solve_lcp,
    ),
    Problem(
        id="valid-palindrome",
        title="Valid Palindrome",
        difficulty="easy",
        prompt=("Read a line. Considering only alphanumeric characters and ignoring case, "
                "print \"true\" if it reads the same forwards and backwards, else \"false\".\n\n"
                "Input:\n  A man, a plan, a canal: Panama\nOutput:\n  true"),
        spoken_intro=("A classic. Given a phrase, ignore punctuation and case, and tell me "
                      "whether it's a palindrome. Print true or false, then submit when ready."),
        starter={
            "python": "s = input()\n\n# TODO: print \"true\" if s is a palindrome (alphanumeric, case-insensitive)\n",
            "c++": ("#include <iostream>\n#include <string>\n#include <cctype>\nusing namespace std;\n\n"
                    "int main() {\n    string s;\n    getline(cin, s);\n\n"
                    "    // TODO: print \"true\" if s is a palindrome (alphanumeric, case-insensitive)\n"
                    "    return 0;\n}\n"),
        },
        visible=[("example", "A man, a plan, a canal: Panama\n"), ("not palindrome", "race a car\n"),
                 ("only punctuation", " .,\n")],
        gen=_gen_palindrome, solve=_solve_palindrome,
    ),
    Problem(
        id="is-subsequence",
        title="Is Subsequence",
        difficulty="easy",
        prompt=("Read two lines: a string s, then a string t. Print \"true\" if s is a "
                "subsequence of t (the characters of s appear in t in order, not necessarily "
                "contiguous), else \"false\".\n\nInput:\n  abc\n  ahbgdc\nOutput:\n  true"),
        spoken_intro=("You'll get two strings, s and t. Tell me whether s is a subsequence of "
                      "t — same characters, in order, gaps allowed. Print true or false."),
        starter={
            "python": "s = input()\nt = input()\n\n# TODO: print \"true\" if s is a subsequence of t\n",
            "c++": ("#include <iostream>\n#include <string>\nusing namespace std;\n\n"
                    "int main() {\n    string s, t;\n    getline(cin, s);\n    getline(cin, t);\n\n"
                    "    // TODO: print \"true\" if s is a subsequence of t\n    return 0;\n}\n"),
        },
        visible=[("example", "abc\nahbgdc\n"), ("not subsequence", "axc\nahbgdc\n"),
                 ("empty s", "\nahbgdc\n")],
        gen=_gen_is_subsequence, solve=_solve_is_subsequence,
    ),
    Problem(
        id="majority-element",
        title="Majority Element",
        difficulty="easy",
        prompt=("Read a line of space-separated integers. One value appears more than half the "
                "time; print that value.\n\nInput:\n  2 2 1 1 1 2 2\nOutput:\n  2"),
        spoken_intro=("Given a list where one value appears more than half the time, find and "
                      "print that value. Run the tests and submit when you're happy."),
        starter={
            "python": "nums = list(map(int, input().split()))\n\n# TODO: print the majority element\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> nums;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) nums.push_back(x);\n\n"
                    "    // TODO: print the majority element\n    return 0;\n}\n"),
        },
        visible=[("example", "3 2 3\n"), ("longer", "2 2 1 1 1 2 2\n"), ("single", "7\n")],
        gen=_gen_majority, solve=_solve_majority,
    ),
    Problem(
        id="jump-game",
        title="Jump Game",
        difficulty="medium",
        prompt=("Read a line of space-separated non-negative integers. Each value is the max "
                "jump length from that position, starting at index 0. Print \"true\" if you can "
                "reach the last index, else \"false\".\n\nInput:\n  2 3 1 1 4\nOutput:\n  true"),
        spoken_intro=("Here's a greedy one. Each number is how far you can jump from that spot, "
                      "starting at the front. Can you reach the end? Print true or false."),
        starter={
            "python": "nums = list(map(int, input().split()))\n\n# TODO: print \"true\" if the last index is reachable\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> nums;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) nums.push_back(x);\n\n"
                    "    // TODO: print \"true\" if the last index is reachable\n    return 0;\n}\n"),
        },
        visible=[("reachable", "2 3 1 1 4\n"), ("stuck at zero", "3 2 1 0 4\n"), ("single", "0\n")],
        gen=_gen_jump_game, solve=_solve_jump_game,
    ),
    Problem(
        id="product-except-self",
        title="Product of Array Except Self",
        difficulty="medium",
        prompt=("Read a line of space-separated integers. Print a line where each position holds "
                "the product of all the other numbers (not the one at that position).\n\n"
                "Input:\n  1 2 3 4\nOutput:\n  24 12 8 6"),
        spoken_intro=("For each position, print the product of every other number in the list. "
                      "Try to do it without division. Run the tests and submit when ready."),
        starter={
            "python": "nums = list(map(int, input().split()))\n\n# TODO: print the product-except-self array, space-separated\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> nums;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) nums.push_back(x);\n\n"
                    "    // TODO: print the product-except-self array, space-separated\n    return 0;\n}\n"),
        },
        visible=[("example", "1 2 3 4\n"), ("with zero", "-1 1 0 -3 3\n"), ("two elements", "2 3\n")],
        gen=_gen_product_except_self, solve=_solve_product_except_self,
    ),
    Problem(
        id="best-time-stock-ii",
        title="Best Time to Buy and Sell Stock II",
        difficulty="medium",
        prompt=("Read a line of space-separated integers (daily stock prices). You may buy and "
                "sell many times (but hold at most one share at a time). Print the maximum total "
                "profit.\n\nInput:\n  7 1 5 3 6 4\nOutput:\n  7"),
        spoken_intro=("Like the earlier stock problem, but now you can trade as many times as "
                      "you like. Find the maximum total profit and print it. Submit when ready."),
        starter={
            "python": "prices = list(map(int, input().split()))\n\n# TODO: print the maximum total profit\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> prices;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) prices.push_back(x);\n\n"
                    "    // TODO: print the maximum total profit\n    return 0;\n}\n"),
        },
        visible=[("example", "7 1 5 3 6 4\n"), ("monotonic up", "1 2 3 4 5\n"), ("no profit", "7 6 4 3 1\n")],
        gen=_gen_stock_ii, solve=_solve_stock_ii,
    ),
    Problem(
        id="search-insert-position",
        title="Search Insert Position",
        difficulty="easy",
        prompt=("Read a line of sorted, distinct space-separated integers, then a line with a "
                "target. Print the index (0-based) of the target, or the index where it would be "
                "inserted to keep the list sorted.\n\nInput:\n  1 3 5 6\n  5\nOutput:\n  2"),
        spoken_intro=("You'll get a sorted list and a target. Print where the target is, or where "
                      "it would go to keep things sorted. Binary search is the intended approach."),
        starter={
            "python": "nums = list(map(int, input().split()))\ntarget = int(input())\n\n# TODO: print the index of target, or its insert position\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> nums;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) nums.push_back(x);\n"
                    "    int target;\n    cin >> target;\n\n"
                    "    // TODO: print the index of target, or its insert position\n    return 0;\n}\n"),
        },
        visible=[("found", "1 3 5 6\n5\n"), ("insert middle", "1 3 5 6\n2\n"), ("insert end", "1 3 5 6\n7\n")],
        gen=_gen_search_insert, solve=_solve_search_insert,
    ),

    # ----- batch 2 -----
    Problem(
        id="add-binary",
        title="Add Binary",
        difficulty="easy",
        prompt=("Read two lines, each a binary string. Print their sum as a binary string.\n\n"
                "Input:\n  11\n  1\nOutput:\n  100"),
        spoken_intro=("You'll get two binary numbers as strings. Add them and print the result, "
                      "also in binary. Run the tests and submit when ready."),
        starter={
            "python": "a = input().strip()\nb = input().strip()\n\n# TODO: print the binary sum of a and b\n",
            "c++": ("#include <iostream>\n#include <string>\nusing namespace std;\n\n"
                    "int main() {\n    string a, b;\n    getline(cin, a);\n    getline(cin, b);\n\n"
                    "    // TODO: print the binary sum of a and b\n    return 0;\n}\n"),
        },
        visible=[("example", "11\n1\n"), ("with carry", "1010\n1011\n"), ("zeros", "0\n0\n")],
        gen=_gen_add_binary, solve=_solve_add_binary,
    ),
    Problem(
        id="plus-one",
        title="Plus One",
        difficulty="easy",
        prompt=("Read a line of space-separated digits representing a non-negative integer (most "
                "significant first, no leading zeros). Add one and print the resulting digits, "
                "space-separated.\n\nInput:\n  1 2 3\nOutput:\n  1 2 4"),
        spoken_intro=("A list of digits represents a number. Add one to it and print the digits "
                      "back out, minding any carry. Run the tests and submit when ready."),
        starter={
            "python": "digits = list(map(int, input().split()))\n\n# TODO: print the digits of (number + 1), space-separated\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> digits;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) digits.push_back(x);\n\n"
                    "    // TODO: print the digits of (number + 1), space-separated\n    return 0;\n}\n"),
        },
        visible=[("example", "1 2 3\n"), ("carry", "9\n"), ("multi carry", "9 9\n")],
        gen=_gen_plus_one, solve=_solve_plus_one,
    ),
    Problem(
        id="valid-anagram",
        title="Valid Anagram",
        difficulty="easy",
        prompt=("Read two lines: strings s and t. Print \"true\" if t is an anagram of s (same "
                "letters with the same counts), else \"false\".\n\nInput:\n  anagram\n  nagaram\n"
                "Output:\n  true"),
        spoken_intro=("Given two strings, tell me whether one is an anagram of the other — same "
                      "letters, same counts. Print true or false, then submit when ready."),
        starter={
            "python": "s = input()\nt = input()\n\n# TODO: print \"true\" if t is an anagram of s\n",
            "c++": ("#include <iostream>\n#include <string>\nusing namespace std;\n\n"
                    "int main() {\n    string s, t;\n    getline(cin, s);\n    getline(cin, t);\n\n"
                    "    // TODO: print \"true\" if t is an anagram of s\n    return 0;\n}\n"),
        },
        visible=[("anagram", "anagram\nnagaram\n"), ("not anagram", "rat\ncar\n"), ("different length", "a\nab\n")],
        gen=_gen_anagram, solve=_solve_anagram,
    ),
    Problem(
        id="ransom-note",
        title="Ransom Note",
        difficulty="easy",
        prompt=("Read two lines: a ransom note, then a magazine string. Print \"true\" if the note "
                "can be built using the magazine's letters (each magazine letter used at most "
                "once), else \"false\".\n\nInput:\n  aa\n  aab\nOutput:\n  true"),
        spoken_intro=("You'll get a note and a magazine. Can you build the note from the "
                      "magazine's letters, using each at most once? Print true or false."),
        starter={
            "python": "note = input()\nmagazine = input()\n\n# TODO: print \"true\" if note can be built from magazine\n",
            "c++": ("#include <iostream>\n#include <string>\nusing namespace std;\n\n"
                    "int main() {\n    string note, magazine;\n    getline(cin, note);\n    getline(cin, magazine);\n\n"
                    "    // TODO: print \"true\" if note can be built from magazine\n    return 0;\n}\n"),
        },
        visible=[("buildable", "aa\naab\n"), ("not enough", "aa\nab\n"), ("missing letter", "a\nb\n")],
        gen=_gen_ransom, solve=_solve_ransom,
    ),
    Problem(
        id="word-pattern",
        title="Word Pattern",
        difficulty="easy",
        prompt=("Read a pattern line (lowercase letters), then a line of space-separated words. "
                "Print \"true\" if the words follow the pattern with a one-to-one mapping between "
                "letters and words, else \"false\".\n\nInput:\n  abba\n  dog cat cat dog\n"
                "Output:\n  true"),
        spoken_intro=("Given a pattern of letters and a list of words, decide whether the words "
                      "follow the pattern as a strict one-to-one mapping. Print true or false."),
        starter={
            "python": "pattern = input().strip()\nwords = input().split()\n\n# TODO: print \"true\" if words follow pattern (bijection)\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\n#include <string>\n"
                    "using namespace std;\n\nint main() {\n    string pattern, line;\n    getline(cin, pattern);\n"
                    "    getline(cin, line);\n    vector<string> words;\n    stringstream ss(line);\n    string w;\n"
                    "    while (ss >> w) words.push_back(w);\n\n"
                    "    // TODO: print \"true\" if words follow pattern (bijection)\n    return 0;\n}\n"),
        },
        visible=[("match", "abba\ndog cat cat dog\n"), ("broken", "abba\ndog cat cat fish\n"),
                 ("not bijection", "aaaa\ndog cat cat dog\n")],
        gen=_gen_word_pattern, solve=_solve_word_pattern,
    ),
    Problem(
        id="summary-ranges",
        title="Summary Ranges",
        difficulty="easy",
        prompt=("Read a line of sorted, distinct space-separated integers (possibly empty). Print "
                "the covering ranges, space-separated: a single number as \"x\", a run as "
                "\"a->b\".\n\nInput:\n  0 1 2 4 5 7\nOutput:\n  0->2 4->5 7"),
        spoken_intro=("Given a sorted list of distinct integers, collapse the consecutive runs "
                      "into ranges and print them. Run the tests and submit when ready."),
        starter={
            "python": "nums = list(map(int, input().split()))\n\n# TODO: print the summary ranges, space-separated\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> nums;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) nums.push_back(x);\n\n"
                    "    // TODO: print the summary ranges, space-separated\n    return 0;\n}\n"),
        },
        visible=[("example", "0 1 2 4 5 7\n"), ("singles and runs", "0 2 3 4 6 8 9\n"), ("empty", "\n")],
        gen=_gen_summary_ranges, solve=_solve_summary_ranges,
    ),
    Problem(
        id="climbing-stairs",
        title="Climbing Stairs",
        difficulty="easy",
        prompt=("Read a single integer n. You climb 1 or 2 steps at a time. Print the number of "
                "distinct ways to reach the top of n steps.\n\nInput:\n  3\nOutput:\n  3"),
        spoken_intro=("A small dynamic-programming one. You climb one or two steps at a time — how "
                      "many distinct ways are there to reach the top of n steps? Print the count."),
        starter={
            "python": "n = int(input())\n\n# TODO: print the number of distinct ways to climb n steps\n",
            "c++": ("#include <iostream>\nusing namespace std;\n\n"
                    "int main() {\n    int n;\n    cin >> n;\n\n"
                    "    // TODO: print the number of distinct ways to climb n steps\n    return 0;\n}\n"),
        },
        visible=[("two steps", "2\n"), ("three steps", "3\n"), ("one step", "1\n")],
        gen=_gen_climb, solve=_solve_climb,
    ),
    Problem(
        id="single-number",
        title="Single Number",
        difficulty="easy",
        prompt=("Read a line of space-separated integers in which every value appears exactly "
                "twice except one, which appears once. Print that single value.\n\n"
                "Input:\n  4 1 2 1 2\nOutput:\n  4"),
        spoken_intro=("Every number here appears twice except one. Find the loner and print it — "
                      "ideally in linear time and constant space. Submit when ready."),
        starter={
            "python": "nums = list(map(int, input().split()))\n\n# TODO: print the value that appears only once\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> nums;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) nums.push_back(x);\n\n"
                    "    // TODO: print the value that appears only once\n    return 0;\n}\n"),
        },
        visible=[("example", "4 1 2 1 2\n"), ("single", "1\n"), ("pair plus one", "2 2 7\n")],
        gen=_gen_single, solve=_solve_single,
    ),
    Problem(
        id="two-sum-ii-sorted",
        title="Two Sum II - Input Array Is Sorted",
        difficulty="medium",
        prompt=("Read a line of sorted space-separated integers, then a line with a target. Exactly "
                "one pair adds up to the target; print their 1-based indices in ascending order, "
                "separated by a space.\n\nInput:\n  2 7 11 15\n  9\nOutput:\n  1 2"),
        spoken_intro=("Like the first two-sum, but the array is sorted and indices are 1-based. "
                      "Find the unique pair that hits the target. The two-pointer trick fits well."),
        starter={
            "python": "nums = list(map(int, input().split()))\ntarget = int(input())\n\n# TODO: print the 1-based indices of the pair summing to target\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> nums;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) nums.push_back(x);\n"
                    "    int target;\n    cin >> target;\n\n"
                    "    // TODO: print the 1-based indices of the pair summing to target\n    return 0;\n}\n"),
        },
        visible=[("example", "2 7 11 15\n9\n"), ("ends", "2 3 4\n6\n"), ("negatives", "-1 0\n-1\n")],
        gen=_gen_two_sum_ii, solve=_solve_two_sum_ii,
    ),
    Problem(
        id="happy-number",
        title="Happy Number",
        difficulty="easy",
        prompt=("Read a single integer n. Repeatedly replace it with the sum of the squares of its "
                "digits. Print \"true\" if this reaches 1, else \"false\" (it loops forever).\n\n"
                "Input:\n  19\nOutput:\n  true"),
        spoken_intro=("Take a number, replace it with the sum of the squares of its digits, and "
                      "repeat. If you reach one, it's happy. Print true or false. Submit when ready."),
        starter={
            "python": "n = int(input())\n\n# TODO: print \"true\" if n is a happy number, else \"false\"\n",
            "c++": ("#include <iostream>\n#include <unordered_set>\nusing namespace std;\n\n"
                    "int main() {\n    int n;\n    cin >> n;\n\n"
                    "    // TODO: print \"true\" if n is a happy number, else \"false\"\n    return 0;\n}\n"),
        },
        visible=[("happy", "19\n"), ("unhappy", "2\n"), ("one", "1\n")],
        gen=_gen_happy, solve=_solve_happy,
    ),

    # ----- batch 3 (medium) -----
    Problem(
        id="maximum-subarray",
        title="Maximum Subarray",
        difficulty="medium",
        prompt=("Read a line of space-separated integers. Print the largest sum obtainable from a "
                "contiguous subarray (at least one element).\n\n"
                "Input:\n  -2 1 -3 4 -1 2 1 -5 4\nOutput:\n  6"),
        spoken_intro=("A classic — Kadane's. Given a list of integers, find the contiguous stretch "
                      "with the largest sum and print that sum. Run the tests and submit when ready."),
        starter={
            "python": "nums = list(map(int, input().split()))\n\n# TODO: print the maximum contiguous subarray sum\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> nums;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) nums.push_back(x);\n\n"
                    "    // TODO: print the maximum contiguous subarray sum\n    return 0;\n}\n"),
        },
        visible=[("example", "-2 1 -3 4 -1 2 1 -5 4\n"), ("all negative", "-3 -1 -2\n"), ("single", "1\n")],
        gen=_gen_max_subarray, solve=_solve_max_subarray,
    ),
    Problem(
        id="longest-substring-no-repeat",
        title="Longest Substring Without Repeating Characters",
        difficulty="medium",
        prompt=("Read a line (lowercase letters, possibly empty). Print the length of the longest "
                "substring that contains no repeated character.\n\n"
                "Input:\n  abcabcbb\nOutput:\n  3"),
        spoken_intro=("A sliding-window one. Given a string, find the longest run with no repeated "
                      "character and print its length. Run the tests and submit when ready."),
        starter={
            "python": "s = input()\n\n# TODO: print the length of the longest substring without repeats\n",
            "c++": ("#include <iostream>\n#include <string>\nusing namespace std;\n\n"
                    "int main() {\n    string s;\n    getline(cin, s);\n\n"
                    "    // TODO: print the length of the longest substring without repeats\n    return 0;\n}\n"),
        },
        visible=[("example", "abcabcbb\n"), ("all same", "bbbbb\n"), ("mixed", "pwwkew\n")],
        gen=_gen_longest_unique, solve=_solve_longest_unique,
    ),
    Problem(
        id="container-most-water",
        title="Container With Most Water",
        difficulty="medium",
        prompt=("Read a line of space-separated non-negative integers (heights of vertical lines). "
                "Choosing two lines, the water they hold is the shorter height times their distance "
                "apart. Print the maximum.\n\nInput:\n  1 8 6 2 5 4 8 3 7\nOutput:\n  49"),
        spoken_intro=("Two-pointer time. Each number is the height of a vertical line; pick two to "
                      "hold the most water — shorter side times the gap. Print the maximum area."),
        starter={
            "python": "height = list(map(int, input().split()))\n\n# TODO: print the maximum water the container can hold\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> height;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) height.push_back(x);\n\n"
                    "    // TODO: print the maximum water the container can hold\n    return 0;\n}\n"),
        },
        visible=[("example", "1 8 6 2 5 4 8 3 7\n"), ("two lines", "1 1\n"), ("tall ends", "4 3 2 1 4\n")],
        gen=_gen_max_area, solve=_solve_max_area,
    ),
    Problem(
        id="coin-change",
        title="Coin Change",
        difficulty="medium",
        prompt=("Read a line of space-separated coin denominations, then a line with a target amount. "
                "Print the fewest coins that sum to the amount, or -1 if it cannot be made.\n\n"
                "Input:\n  1 2 5\n  11\nOutput:\n  3"),
        spoken_intro=("A DP one. Given coin denominations and a target amount, print the fewest coins "
                      "that make the amount, or minus one if it's impossible. Submit when ready."),
        starter={
            "python": "coins = list(map(int, input().split()))\namount = int(input())\n\n# TODO: print the fewest coins to make amount, or -1\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> coins;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) coins.push_back(x);\n"
                    "    int amount;\n    cin >> amount;\n\n"
                    "    // TODO: print the fewest coins to make amount, or -1\n    return 0;\n}\n"),
        },
        visible=[("example", "1 2 5\n11\n"), ("impossible", "2\n3\n"), ("zero amount", "1 2 5\n0\n")],
        gen=_gen_coin_change, solve=_solve_coin_change,
    ),
    Problem(
        id="house-robber",
        title="House Robber",
        difficulty="medium",
        prompt=("Read a line of space-separated non-negative integers (money in each house). You "
                "cannot rob two adjacent houses. Print the maximum you can rob.\n\n"
                "Input:\n  2 7 9 3 1\nOutput:\n  12"),
        spoken_intro=("Another DP. Each number is the cash in a house, but you can't hit two houses "
                      "in a row. Print the maximum you can take. Run the tests and submit when ready."),
        starter={
            "python": "nums = list(map(int, input().split()))\n\n# TODO: print the maximum non-adjacent sum\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> nums;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) nums.push_back(x);\n\n"
                    "    // TODO: print the maximum non-adjacent sum\n    return 0;\n}\n"),
        },
        visible=[("example", "2 7 9 3 1\n"), ("small", "1 2 3 1\n"), ("single", "5\n")],
        gen=_gen_house_robber, solve=_solve_house_robber,
    ),
    Problem(
        id="unique-paths",
        title="Unique Paths",
        difficulty="medium",
        prompt=("Read two integers m and n on one line: the number of rows and columns of a grid. "
                "Starting top-left and moving only right or down, print how many distinct paths "
                "reach the bottom-right cell.\n\nInput:\n  3 7\nOutput:\n  28"),
        spoken_intro=("On an m-by-n grid you move only right or down from the top-left. Print how "
                      "many distinct paths reach the bottom-right corner. Submit when ready."),
        starter={
            "python": "m, n = map(int, input().split())\n\n# TODO: print the number of distinct paths\n",
            "c++": ("#include <iostream>\nusing namespace std;\n\n"
                    "int main() {\n    int m, n;\n    cin >> m >> n;\n\n"
                    "    // TODO: print the number of distinct paths\n    return 0;\n}\n"),
        },
        visible=[("example", "3 7\n"), ("square", "3 3\n"), ("single row", "1 5\n")],
        gen=_gen_unique_paths, solve=_solve_unique_paths,
    ),
    Problem(
        id="decode-ways",
        title="Decode Ways",
        difficulty="medium",
        prompt=("Read a line of digits. Using the mapping A=1, B=2, ..., Z=26, print the number of "
                "ways to decode the string into letters. A leading zero in any group is invalid.\n\n"
                "Input:\n  226\nOutput:\n  3"),
        spoken_intro=("Digits map to letters — one is A, up to twenty-six is Z. Print how many ways "
                      "the digit string can be decoded, watching out for zeros. Submit when ready."),
        starter={
            "python": "s = input().strip()\n\n# TODO: print the number of ways to decode s\n",
            "c++": ("#include <iostream>\n#include <string>\nusing namespace std;\n\n"
                    "int main() {\n    string s;\n    getline(cin, s);\n\n"
                    "    // TODO: print the number of ways to decode s\n    return 0;\n}\n"),
        },
        visible=[("example", "226\n"), ("leading zero", "06\n"), ("simple", "12\n")],
        gen=_gen_decode_ways, solve=_solve_decode_ways,
    ),
    Problem(
        id="maximum-product-subarray",
        title="Maximum Product Subarray",
        difficulty="medium",
        prompt=("Read a line of space-separated integers. Print the largest product obtainable from "
                "a contiguous subarray (at least one element).\n\n"
                "Input:\n  2 3 -2 4\nOutput:\n  6"),
        spoken_intro=("Like maximum subarray, but with products — and negatives can flip the sign. "
                      "Print the largest product of a contiguous stretch. Submit when ready."),
        starter={
            "python": "nums = list(map(int, input().split()))\n\n# TODO: print the maximum contiguous product\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> nums;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) nums.push_back(x);\n\n"
                    "    // TODO: print the maximum contiguous product\n    return 0;\n}\n"),
        },
        visible=[("example", "2 3 -2 4\n"), ("with zero", "-2 0 -1\n"), ("negatives", "-2 3 -4\n")],
        gen=_gen_max_product, solve=_solve_max_product,
    ),
    Problem(
        id="search-rotated-sorted",
        title="Search in Rotated Sorted Array",
        difficulty="medium",
        prompt=("Read a line of distinct space-separated integers that was originally sorted "
                "ascending then rotated, and a line with a target. Print the 0-based index of the "
                "target, or -1 if it is absent.\n\nInput:\n  4 5 6 7 0 1 2\n  0\nOutput:\n  4"),
        spoken_intro=("A sorted array got rotated at some pivot. Find the target in it — ideally in "
                      "log time — and print its index, or minus one if it's not there."),
        starter={
            "python": "nums = list(map(int, input().split()))\ntarget = int(input())\n\n# TODO: print the index of target, or -1\n",
            "c++": ("#include <iostream>\n#include <sstream>\n#include <vector>\nusing namespace std;\n\n"
                    "int main() {\n    string line;\n    getline(cin, line);\n    vector<int> nums;\n"
                    "    stringstream ss(line);\n    int x;\n    while (ss >> x) nums.push_back(x);\n"
                    "    int target;\n    cin >> target;\n\n"
                    "    // TODO: print the index of target, or -1\n    return 0;\n}\n"),
        },
        visible=[("found", "4 5 6 7 0 1 2\n0\n"), ("absent", "4 5 6 7 0 1 2\n3\n"), ("single", "1\n0\n")],
        gen=_gen_search_rotated, solve=_solve_search_rotated,
    ),
]


# ---------------------------------------------------------------------------
# build + merge
# ---------------------------------------------------------------------------

def build_entry(p: Problem, seed: int) -> dict:
    rng = random.Random(seed)
    seen: set = set()

    def make(name: str, stdin: str) -> dict:
        return {"name": name, "stdin": stdin, "expected": p.solve(stdin)}

    tests = []
    for name, stdin in p.visible:
        seen.add(stdin.strip())
        tests.append(make(name, stdin))

    hidden = []
    attempts = 0
    while len(hidden) < p.n_hidden and attempts < p.n_hidden * 20:
        attempts += 1
        stdin = p.gen(rng)
        if stdin.strip() in seen:
            continue
        seen.add(stdin.strip())
        hidden.append(make(f"hidden {len(hidden) + 1}", stdin))

    return {
        "id": p.id,
        "title": p.title,
        "difficulty": p.difficulty,
        "prompt": p.prompt,
        "spoken_intro": p.spoken_intro,
        "starter": p.starter,
        "tests": tests,
        "hidden_tests": hidden,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="merge new problems into the bank")
    ap.add_argument("--seed", type=int, default=1234, help="deterministic fuzz seed")
    args = ap.parse_args()

    with open(BANK, encoding="utf-8") as f:
        bank = json.load(f)
    existing = {q["id"] for q in bank}

    added = 0
    for i, p in enumerate(PROBLEMS):
        if p.id in existing:
            print(f"  skip {p.id:<26} (already in bank)")
            continue
        entry = build_entry(p, args.seed + i)
        bank.append(entry)
        added += 1
        print(f"  add  {p.id:<26} {p.difficulty:<7} "
              f"{len(entry['tests'])} visible + {len(entry['hidden_tests'])} hidden")

    print(f"\n{added} new problem(s); bank would hold {len(bank)} total")
    if args.write and added:
        with open(BANK, "w", encoding="utf-8") as f:
            json.dump(bank, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", os.path.normpath(BANK))
    elif not args.write:
        print("(dry run — pass --write to update the bank)")


if __name__ == "__main__":
    main()
