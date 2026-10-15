"""Builds pilot/data/conventions.jsonl -- the "private conventions" benchmark.

    pilot/.venv/bin/python pilot/benchmarks/build_conventions.py

Each family has ONE hidden project rule that the task text never states. The hidden tests enforce it, so a model that
does not know the rule writes reasonable code that fails; a model that has *learned* it from an earlier failure in the
same family passes. Controls have no hidden rule and check that memory does not hurt ordinary tasks.

Every task is validated by actually running it: the reference solution must pass all tests, and the `naive` solution
(what a competent programmer writes without knowing the rule) must FAIL them. Tasks that do not pass this are rejected,
so the benchmark cannot silently contain tests that cannot tell the two apart.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sandbox import run_python  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "conventions.jsonl"
OUT_HELDOUT = Path(__file__).resolve().parents[1] / "data" / "conventions_heldout.jsonl"

SETUP = '''import copy


def _chk(label, got, exp):
    assert got == exp and type(got) is type(exp), f"{label}: expected {exp!r}, got {got!r}"
'''

# family -> (rule, [ (name, question, reference, naive, [(args, expected), ...]) ])
FAMILIES: dict[str, tuple[str, list]] = {}

FAMILIES["money_paise"] = ("All money amounts are integers in paise; every calculation uses integer arithmetic and truncates (floor).", [
    ("apply_discount", "Write a function `apply_discount(price, percent)` that returns the price after applying a percentage discount.",
     "def apply_discount(price, percent):\n    return price * (100 - percent) // 100",
     "def apply_discount(price, percent):\n    return price * (1 - percent / 100)",
     [("1999, 10", 1799), ("1000, 25", 750), ("999, 50", 499), ("500, 0", 500)]),
    ("add_gst", "Write a function `add_gst(amount, rate_percent)` that returns the amount including GST at the given rate.",
     "def add_gst(amount, rate_percent):\n    return amount + amount * rate_percent // 100",
     "def add_gst(amount, rate_percent):\n    return amount * (1 + rate_percent / 100)",
     [("1999, 18", 2358), ("1000, 5", 1050), ("333, 18", 392), ("0, 18", 0)]),
    ("split_bill", "Write a function `split_bill(total, people)` that splits a bill between people and returns each person's share as a list. The shares must add up to the total.",
     "def split_bill(total, people):\n    base, extra = divmod(total, people)\n    return [base + (1 if i < extra else 0) for i in range(people)]",
     "def split_bill(total, people):\n    return [total / people] * people",
     [("1000, 3", [334, 333, 333]), ("10, 4", [3, 3, 2, 2]), ("900, 3", [300, 300, 300]), ("1, 2", [1, 0])]),
    ("cart_total", "Write a function `cart_total(prices, coupon_percent)` that returns the total of the prices after taking a coupon discount off the whole cart.",
     "def cart_total(prices, coupon_percent):\n    return sum(prices) * (100 - coupon_percent) // 100",
     "def cart_total(prices, coupon_percent):\n    return sum(prices) * (1 - coupon_percent / 100)",
     [("[1999, 500], 10", 2249), ("[100, 100, 100], 50", 150), ("[999], 33", 669), ("[], 20", 0)]),
    ("monthly_installment", "Write a function `monthly_installment(total, months)` that returns how much has to be paid each month to clear a purchase.",
     "def monthly_installment(total, months):\n    return total // months",
     "def monthly_installment(total, months):\n    return total / months",
     [("1000, 3", 333), ("1200, 12", 100), ("1999, 4", 499), ("50, 100", 0)]),
    ("refund_amount", "Write a function `refund_amount(paid, used_days, total_days)` that returns the refund for a subscription that was cancelled part-way through.",
     "def refund_amount(paid, used_days, total_days):\n    return paid * (total_days - used_days) // total_days",
     "def refund_amount(paid, used_days, total_days):\n    return paid * (total_days - used_days) / total_days",
     [("3000, 10, 30", 2000), ("1000, 1, 3", 666), ("999, 0, 30", 999), ("500, 30, 30", 0)]),
    ("tip_amount", "Write a function `tip_amount(bill, tip_percent)` that returns the tip to leave on a bill.",
     "def tip_amount(bill, tip_percent):\n    return bill * tip_percent // 100",
     "def tip_amount(bill, tip_percent):\n    return round(bill * tip_percent / 100)",
     [("1999, 15", 299), ("1000, 10", 100), ("333, 15", 49), ("50, 5", 2)]),
    ("format_price", "Write a function `format_price(amount)` that returns the amount as a display string with the rupee sign.",
     "def format_price(amount):\n    return f'₹{amount // 100}.{amount % 100:02d}'",
     "def format_price(amount):\n    return f'₹{amount:.2f}'",
     [("1999", "₹19.99"), ("5", "₹0.05"), ("100000", "₹1000.00"), ("0", "₹0.00")]),
])

FAMILIES["none_on_invalid"] = ("Functions never raise on bad input: they return None instead.", [
    ("parse_age", "Write a function `parse_age(text)` that turns text into a person's age.",
     "def parse_age(text):\n    try:\n        age = int(str(text).strip())\n    except ValueError:\n        return None\n    return age if 0 <= age <= 150 else None",
     "def parse_age(text):\n    return int(text)",
     [("'25'", 25), ("'abc'", None), ("'-3'", None), ("'200'", None), ("' 42 '", 42)]),
    ("safe_divide", "Write a function `safe_divide(a, b)` that divides a by b.",
     "def safe_divide(a, b):\n    return None if b == 0 else a / b",
     "def safe_divide(a, b):\n    return a / b",
     [("6, 3", 2.0), ("1, 0", None), ("0, 5", 0.0), ("-9, 3", -3.0)]),
    ("get_item", "Write a function `get_item(items, index)` that returns the item at the given position in a list.",
     "def get_item(items, index):\n    return items[index] if 0 <= index < len(items) else None",
     "def get_item(items, index):\n    return items[index]",
     [("['a', 'b'], 1", "b"), ("['a', 'b'], 5", None), ("[], 0", None), ("['x'], 0", "x")]),
    ("first_word", "Write a function `first_word(text)` that returns the first word of a piece of text.",
     "def first_word(text):\n    if not isinstance(text, str) or not text.split():\n        return None\n    return text.split()[0]",
     "def first_word(text):\n    return text.split()[0]",
     [("'hello world'", "hello"), ("'   '", None), ("None", None), ("'  hi there'", "hi")]),
    ("to_float", "Write a function `to_float(value)` that converts a value read from a form field into a number.",
     "def to_float(value):\n    try:\n        return float(value)\n    except (TypeError, ValueError):\n        return None",
     "def to_float(value):\n    return float(value)",
     [("'3.5'", 3.5), ("'x'", None), ("None", None), ("'10'", 10.0)]),
    ("average", "Write a function `average(numbers)` that returns the mean of a list of numbers.",
     "def average(numbers):\n    return sum(numbers) / len(numbers) if numbers else None",
     "def average(numbers):\n    return sum(numbers) / len(numbers)",
     [("[1, 2, 3]", 2.0), ("[]", None), ("[5]", 5.0), ("[1, 2]", 1.5)]),
    ("get_path", "Write a function `get_path(data, keys)` that looks up a value in nested dictionaries by following a list of keys.",
     "def get_path(data, keys):\n    for k in keys:\n        if not isinstance(data, dict) or k not in data:\n            return None\n        data = data[k]\n    return data",
     "def get_path(data, keys):\n    for k in keys:\n        data = data[k]\n    return data",
     [("{'a': {'b': 1}}, ['a', 'b']", 1), ("{'a': {'b': 1}}, ['a', 'c']", None), ("{}, ['x']", None), ("{'a': 5}, ['a', 'b']", None)]),
    ("email_domain", "Write a function `email_domain(email)` that returns the domain part of an email address.",
     "def email_domain(email):\n    if not isinstance(email, str) or email.count('@') != 1:\n        return None\n    return email.split('@')[1] or None",
     "def email_domain(email):\n    return email.split('@')[1]",
     [("'a@b.com'", "b.com"), ("'nope'", None), ("'a@b@c'", None), ("'x@y.org'", "y.org")]),
])

FAMILIES["ids_padded"] = ("IDs are strings of exactly 6 digits, zero-padded (the number 42 is the id '000042'); id-returning functions return such strings.", [
    ("make_id", "Write a function `make_id(n)` that returns the id for record number n.",
     "def make_id(n):\n    return f'{int(n):06d}'",
     "def make_id(n):\n    return str(n)",
     [("42", "000042"), ("1", "000001"), ("123456", "123456"), ("0", "000000")]),
    ("next_id", "Write a function `next_id(current)` that returns the id that comes after the current one.",
     "def next_id(current):\n    return f'{int(current) + 1:06d}'",
     "def next_id(current):\n    return int(current) + 1",
     [("'000099'", "000100"), ("'000001'", "000002"), ("'000000'", "000001"), ("41", "000042")]),
    ("is_valid_id", "Write a function `is_valid_id(value)` that returns True if the value is a valid record id.",
     "def is_valid_id(value):\n    return isinstance(value, str) and len(value) == 6 and value.isdigit()",
     "def is_valid_id(value):\n    return str(value).isdigit()",
     [("'000042'", True), ("'42'", False), ("42", False), ("'12345a'", False), ("'1234567'", False)]),
    ("normalize_ids", "Write a function `normalize_ids(values)` that cleans up a list of ids that came from different sources (some numbers, some text).",
     "def normalize_ids(values):\n    return [f'{int(v):06d}' for v in values]",
     "def normalize_ids(values):\n    return [str(v) for v in values]",
     [("[5, '000007', '12']", ["000005", "000007", "000012"]), ("[]", []), ("[1]", ["000001"]), ("['000100', 99]", ["000100", "000099"])]),
    ("order_label", "Write a function `order_label(order_id)` that returns the label printed on an order, which is ORD- followed by its id.",
     "def order_label(order_id):\n    return f'ORD-{int(order_id):06d}'",
     "def order_label(order_id):\n    return f'ORD-{order_id}'",
     [("42", "ORD-000042"), ("'7'", "ORD-000007"), ("'000123'", "ORD-000123"), ("1000", "ORD-001000")]),
    ("id_range", "Write a function `id_range(start, end)` that returns every id from start to end, both included.",
     "def id_range(start, end):\n    return [f'{i:06d}' for i in range(int(start), int(end) + 1)]",
     "def id_range(start, end):\n    return list(range(start, end + 1))",
     [("3, 5", ["000003", "000004", "000005"]), ("7, 7", ["000007"]), ("9, 8", []), ("'000001', '000002'", ["000001", "000002"])]),
    ("max_id", "Write a function `max_id(ids)` that returns the largest id in a list of ids that came from different sources (some numbers, some text).",
     "def max_id(ids):\n    return f'{max(int(i) for i in ids):06d}'",
     "def max_id(ids):\n    return max(int(i) for i in ids)",
     [("[3, '000010', '7']", "000010"), ("['000001']", "000001"), ("[99, 100]", "000100"), ("[0, 0]", "000000")]),
    ("id_from_label", "Write a function `id_from_label(label)` that extracts the id from an order label such as ORD-000042.",
     "def id_from_label(label):\n    return f'{int(label.split(\"-\")[1]):06d}'",
     "def id_from_label(label):\n    return int(label.split('-')[1])",
     [("'ORD-000042'", "000042"), ("'ORD-7'", "000007"), ("'ORD-123456'", "123456"), ("'ORD-000000'", "000000")]),
])

# no hidden rule: ordinary tasks, to check that learning does not hurt
FAMILIES["control"] = ("", [
    ("reverse_words", "Write a function `reverse_words(text)` that reverses the order of the words in a sentence.",
     "def reverse_words(text):\n    return ' '.join(reversed(text.split()))", None,
     [("'a b c'", "c b a"), ("'one'", "one"), ("''", ""), ("'hello big world'", "world big hello")]),
    ("count_vowels", "Write a function `count_vowels(text)` that counts the vowels in a string.",
     "def count_vowels(text):\n    return sum(c in 'aeiouAEIOU' for c in text)", None,
     [("'hello'", 2), ("''", 0), ("'AEIOU'", 5), ("'xyz'", 0)]),
    ("flatten", "Write a function `flatten(lists)` that turns a list of lists into one list.",
     "def flatten(lists):\n    return [x for sub in lists for x in sub]", None,
     [("[[1], [2, 3]]", [1, 2, 3]), ("[]", []), ("[[], [4]]", [4]), ("[['a'], ['b']]", ["a", "b"])]),
    ("is_palindrome", "Write a function `is_palindrome(text)` that tells whether a word reads the same backwards.",
     "def is_palindrome(text):\n    return text == text[::-1]", None,
     [("'level'", True), ("'abc'", False), ("''", True), ("'aa'", True)]),
])

# ---------------------------------------------------------------------------------------------------------------
# Extensions: the first three families grow to 10 tasks; six more families; controls grow to 10.
# ---------------------------------------------------------------------------------------------------------------
FAMILIES["money_paise"][1].extend([
    ("late_fee", "Write a function `late_fee(due, days_late)` that returns the late fee, which is 2 percent of the amount due for each day it is late.",
     "def late_fee(due, days_late):\n    return due * 2 * days_late // 100",
     "def late_fee(due, days_late):\n    return due * 0.02 * days_late",
     [("1000, 3", 60), ("999, 1", 19), ("500, 0", 0), ("1, 5", 0)]),
    ("commission", "Write a function `commission(sale, rate_percent)` that returns the commission a salesperson earns on a sale.",
     "def commission(sale, rate_percent):\n    return sale * rate_percent // 100",
     "def commission(sale, rate_percent):\n    return round(sale * rate_percent / 100)",
     [("1999, 7", 139), ("1000, 10", 100), ("333, 3", 9), ("50, 50", 25)]),
])
FAMILIES["none_on_invalid"][1].extend([
    ("parse_percent", "Write a function `parse_percent(text)` that reads a percentage such as '45%' or '45' and returns it as a number.",
     "def parse_percent(text):\n    try:\n        n = int(str(text).strip().rstrip('%'))\n    except ValueError:\n        return None\n    return n if 0 <= n <= 100 else None",
     "def parse_percent(text):\n    return int(text.strip().rstrip('%'))",
     [("'45%'", 45), ("'abc'", None), ("'150'", None), ("'7'", 7)]),
    ("last_item", "Write a function `last_item(items)` that returns the last item of a list.",
     "def last_item(items):\n    return items[-1] if items else None",
     "def last_item(items):\n    return items[-1]",
     [("[1, 2, 3]", 3), ("[]", None), ("['a']", "a"), ("[0, 0]", 0)]),
])
FAMILIES["ids_padded"][1].extend([
    ("is_same_id", "Write a function `is_same_id(a, b)` that tells whether two ids, which may come from different sources, refer to the same record.",
     "def is_same_id(a, b):\n    return f'{int(a):06d}' == f'{int(b):06d}'",
     "def is_same_id(a, b):\n    return a == b",
     [("5, '000005'", True), ("'000005', '5'", True), ("'000005', '000006'", False), ("7, 7", True)]),
    ("batch_ids", "Write a function `batch_ids(start, count)` that returns the ids for `count` consecutive records beginning at `start`.",
     "def batch_ids(start, count):\n    return [f'{int(start) + i:06d}' for i in range(count)]",
     "def batch_ids(start, count):\n    return [int(start) + i for i in range(count)]",
     [("3, 2", ["000003", "000004"]), ("999999, 1", ["999999"]), ("0, 0", []), ("'000010', 2", ["000010", "000011"])]),
])

FAMILIES["dates_dmy"] = ("Dates are strings in DD-MM-YYYY format (for example '05-03-2024'); functions that take or return a date use it.", [
    ("format_date", "Write a function `format_date(year, month, day)` that returns the date as text.",
     "def format_date(year, month, day):\n    return f'{day:02d}-{month:02d}-{year}'",
     "def format_date(year, month, day):\n    return f'{year}-{month:02d}-{day:02d}'",
     [("2024, 3, 5", "05-03-2024"), ("1999, 12, 31", "31-12-1999"), ("2000, 1, 1", "01-01-2000")]),
    ("parse_date", "Write a function `parse_date(text)` that splits a date given as text into a (year, month, day) tuple of integers.",
     "def parse_date(text):\n    d, m, y = map(int, text.split('-'))\n    return (y, m, d)",
     "def parse_date(text):\n    y, m, d = map(int, text.split('-'))\n    return (y, m, d)",
     [("'05-03-2024'", (2024, 3, 5)), ("'31-12-1999'", (1999, 12, 31)), ("'01-01-2000'", (2000, 1, 1))]),
    ("next_day", "Write a function `next_day(date_text)` that returns the date of the following day.",
     "def next_day(date_text):\n    from datetime import datetime, timedelta\n    return (datetime.strptime(date_text, '%d-%m-%Y') + timedelta(days=1)).strftime('%d-%m-%Y')",
     "def next_day(date_text):\n    from datetime import datetime, timedelta\n    return (datetime.strptime(date_text, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')",
     [("'28-02-2024'", "29-02-2024"), ("'31-12-1999'", "01-01-2000"), ("'01-03-2023'", "02-03-2023")]),
    ("days_between", "Write a function `days_between(start, end)` that returns the number of days from the start date to the end date.",
     "def days_between(start, end):\n    from datetime import datetime\n    f = '%d-%m-%Y'\n    return (datetime.strptime(end, f) - datetime.strptime(start, f)).days",
     "def days_between(start, end):\n    from datetime import date\n    return (date.fromisoformat(end) - date.fromisoformat(start)).days",
     [("'01-01-2024', '31-01-2024'", 30), ("'31-12-2023', '01-01-2024'", 1), ("'05-03-2024', '05-03-2024'", 0)]),
    ("is_weekend", "Write a function `is_weekend(date_text)` that tells whether a date falls on a Saturday or Sunday.",
     "def is_weekend(date_text):\n    from datetime import datetime\n    return datetime.strptime(date_text, '%d-%m-%Y').weekday() >= 5",
     "def is_weekend(date_text):\n    from datetime import date\n    return date.fromisoformat(date_text).weekday() >= 5",
     [("'06-04-2024'", True), ("'07-04-2024'", True), ("'08-04-2024'", False)]),
    ("year_of", "Write a function `year_of(date_text)` that returns the year of a date.",
     "def year_of(date_text):\n    return int(date_text.split('-')[2])",
     "def year_of(date_text):\n    return int(date_text.split('-')[0])",
     [("'05-03-2024'", 2024), ("'31-12-1999'", 1999), ("'01-01-2000'", 2000)]),
    ("add_days", "Write a function `add_days(date_text, days)` that returns the date a number of days later.",
     "def add_days(date_text, days):\n    from datetime import datetime, timedelta\n    return (datetime.strptime(date_text, '%d-%m-%Y') + timedelta(days=days)).strftime('%d-%m-%Y')",
     "def add_days(date_text, days):\n    from datetime import date, timedelta\n    return (date.fromisoformat(date_text) + timedelta(days=days)).isoformat()",
     [("'28-02-2024', 2", "01-03-2024"), ("'31-12-2023', 1", "01-01-2024"), ("'15-06-2024', 0", "15-06-2024")]),
    ("sort_dates", "Write a function `sort_dates(dates)` that returns a list of dates in chronological order.",
     "def sort_dates(dates):\n    return sorted(dates, key=lambda s: tuple(reversed(s.split('-'))))",
     "def sort_dates(dates):\n    return sorted(dates)",
     [("['01-02-2024', '15-01-2024', '31-12-2023']", ['31-12-2023', '15-01-2024', '01-02-2024']), ("[]", []), ("['05-03-2024']", ['05-03-2024'])]),
    ("latest_date", "Write a function `latest_date(dates)` that returns the most recent date in a list.",
     "def latest_date(dates):\n    return max(dates, key=lambda s: tuple(reversed(s.split('-'))))",
     "def latest_date(dates):\n    return max(dates)",
     [("['01-02-2024', '15-01-2024']", "01-02-2024"), ("['31-12-2023', '01-01-2024']", "01-01-2024"), ("['05-03-2024']", "05-03-2024")]),
    ("age_on", "Write a function `age_on(birth_date, on_date)` that returns a person's age in whole years on a given date.",
     "def age_on(birth_date, on_date):\n    bd, bm, by = map(int, birth_date.split('-'))\n    od, om, oy = map(int, on_date.split('-'))\n    return oy - by - ((om, od) < (bm, bd))",
     "def age_on(birth_date, on_date):\n    from datetime import date\n    b, o = date.fromisoformat(birth_date), date.fromisoformat(on_date)\n    return o.year - b.year - ((o.month, o.day) < (b.month, b.day))",
     [("'15-06-2000', '14-06-2024'", 23), ("'15-06-2000', '15-06-2024'", 24), ("'01-01-2000', '31-12-2023'", 23)]),
])

# no_mutation: each case is (first_argument_literal, remaining_arguments, expected); the tests also check the input is unchanged
FAMILIES["no_mutation"] = ("Functions never modify their arguments: they return a new list or dict and leave the input untouched.", [
    ("sorted_desc", "Write a function `sorted_desc(items)` that returns the items ordered from largest to smallest.",
     "def sorted_desc(items):\n    return sorted(items, reverse=True)",
     "def sorted_desc(items):\n    items.sort(reverse=True)\n    return items",
     [("[3, 1, 2]", "", [3, 2, 1]), ("[1]", "", [1]), ("[]", "", []), ("[2, 2, 1]", "", [2, 2, 1])]),
    ("with_item", "Write a function `with_item(items, x)` that returns the list with x added at the end.",
     "def with_item(items, x):\n    return items + [x]",
     "def with_item(items, x):\n    items.append(x)\n    return items",
     [("[1, 2]", "3", [1, 2, 3]), ("[]", "'a'", ['a']), ("[5]", "5", [5, 5])]),
    ("without_duplicates", "Write a function `without_duplicates(items)` that returns the items with repeats removed, keeping the first occurrence of each.",
     "def without_duplicates(items):\n    return list(dict.fromkeys(items))",
     "def without_duplicates(items):\n    items[:] = list(dict.fromkeys(items))\n    return items",
     [("[1, 2, 1, 3]", "", [1, 2, 3]), ("[]", "", []), ("['a', 'a']", "", ['a'])]),
    ("merged", "Write a function `merged(first, second)` that returns one dictionary holding the entries of both, preferring the second on clashes.",
     "def merged(first, second):\n    return {**first, **second}",
     "def merged(first, second):\n    first.update(second)\n    return first",
     [("{'a': 1}", "{'b': 2}", {'a': 1, 'b': 2}), ("{'a': 1}", "{'a': 5}", {'a': 5}), ("{}", "{}", {})]),
    ("add_to_all", "Write a function `add_to_all(numbers, n)` that returns the numbers each increased by n.",
     "def add_to_all(numbers, n):\n    return [x + n for x in numbers]",
     "def add_to_all(numbers, n):\n    for i in range(len(numbers)):\n        numbers[i] += n\n    return numbers",
     [("[1, 2, 3]", "10", [11, 12, 13]), ("[]", "5", []), ("[0]", "-1", [-1])]),
    ("without_key", "Write a function `without_key(data, key)` that returns the dictionary without the given key.",
     "def without_key(data, key):\n    return {k: v for k, v in data.items() if k != key}",
     "def without_key(data, key):\n    data.pop(key, None)\n    return data",
     [("{'a': 1, 'b': 2}", "'a'", {'b': 2}), ("{'a': 1}", "'z'", {'a': 1}), ("{}", "'a'", {})]),
    ("reversed_list", "Write a function `reversed_list(items)` that returns the items in reverse order.",
     "def reversed_list(items):\n    return items[::-1]",
     "def reversed_list(items):\n    items.reverse()\n    return items",
     [("[1, 2, 3]", "", [3, 2, 1]), ("[]", "", []), ("['a']", "", ['a'])]),
    ("capped", "Write a function `capped(numbers, limit)` that returns the numbers with any value above the limit lowered to the limit.",
     "def capped(numbers, limit):\n    return [min(x, limit) for x in numbers]",
     "def capped(numbers, limit):\n    for i, x in enumerate(numbers):\n        numbers[i] = min(x, limit)\n    return numbers",
     [("[5, 10, 15]", "10", [5, 10, 10]), ("[]", "3", []), ("[1]", "0", [0])]),
    ("rotated", "Write a function `rotated(items, k)` that returns the items moved k places to the left, with the front items wrapping round to the end.",
     "def rotated(items, k):\n    if not items:\n        return []\n    k %= len(items)\n    return items[k:] + items[:k]",
     "def rotated(items, k):\n    for _ in range(k):\n        items.append(items.pop(0))\n    return items",
     [("[1, 2, 3, 4]", "1", [2, 3, 4, 1]), ("[1, 2, 3]", "3", [1, 2, 3]), ("[]", "2", [])]),
    ("uppercased", "Write a function `uppercased(words)` that returns the words in upper case.",
     "def uppercased(words):\n    return [w.upper() for w in words]",
     "def uppercased(words):\n    for i, w in enumerate(words):\n        words[i] = w.upper()\n    return words",
     [("['a', 'b']", "", ['A', 'B']), ("[]", "", []), ("['Hi']", "", ['HI'])]),
])

FAMILIES["username_norm"] = ("Usernames are case-insensitive: they are stored stripped and in lower case, and any function that returns a username returns that normalised form.", [
    ("normalize_username", "Write a function `normalize_username(name)` that cleans up a username typed by a person.",
     "def normalize_username(name):\n    return name.strip().lower()",
     "def normalize_username(name):\n    return name.strip()",
     [("'  Alice '", "alice"), ("'BOB'", "bob"), ("'carol'", "carol")]),
    ("same_user", "Write a function `same_user(a, b)` that tells whether two typed usernames belong to the same person.",
     "def same_user(a, b):\n    return a.strip().lower() == b.strip().lower()",
     "def same_user(a, b):\n    return a == b",
     [("'Alice', 'alice'", True), ("'bob ', ' BOB'", True), ("'a', 'b'", False)]),
    ("dedupe_users", "Write a function `dedupe_users(names)` that returns the usernames with repeats removed, in the order first seen.",
     "def dedupe_users(names):\n    return list(dict.fromkeys(n.strip().lower() for n in names))",
     "def dedupe_users(names):\n    return list(dict.fromkeys(names))",
     [("['Al', 'al', 'Bo']", ['al', 'bo']), ("[]", []), ("['X', ' x ']", ['x'])]),
    ("is_taken", "Write a function `is_taken(existing, name)` that tells whether a username is already in use.",
     "def is_taken(existing, name):\n    return name.strip().lower() in {e.strip().lower() for e in existing}",
     "def is_taken(existing, name):\n    return name in existing",
     [("['alice', 'bob'], 'Alice'", True), ("['alice'], 'carol'", False), ("['Alice'], ' alice '", True)]),
    ("user_handle", "Write a function `user_handle(name)` that returns the @handle shown on a person's profile.",
     "def user_handle(name):\n    return '@' + name.strip().lower()",
     "def user_handle(name):\n    return '@' + name.strip()",
     [("' Alice'", "@alice"), ("'BOB'", "@bob"), ("'carol'", "@carol")]),
    ("sort_users", "Write a function `sort_users(names)` that returns the usernames in alphabetical order.",
     "def sort_users(names):\n    return sorted(n.strip().lower() for n in names)",
     "def sort_users(names):\n    return sorted(names)",
     [("['bob', 'Alice', 'carol']", ['alice', 'bob', 'carol']), ("[]", []), ("['B', 'a']", ['a', 'b'])]),
    ("count_unique_users", "Write a function `count_unique_users(names)` that counts how many different people appear in a list of usernames.",
     "def count_unique_users(names):\n    return len({n.strip().lower() for n in names})",
     "def count_unique_users(names):\n    return len(set(names))",
     [("['A', 'a', 'b']", 2), ("[]", 0), ("['x ', ' X']", 1)]),
    ("user_counts", "Write a function `user_counts(names)` that returns how many times each username appears.",
     "def user_counts(names):\n    out = {}\n    for n in names:\n        k = n.strip().lower()\n        out[k] = out.get(k, 0) + 1\n    return out",
     "def user_counts(names):\n    out = {}\n    for n in names:\n        out[n] = out.get(n, 0) + 1\n    return out",
     [("['A', 'a', 'b']", {'a': 2, 'b': 1}), ("[]", {}), ("['x ', ' X']", {'x': 2})]),
    ("find_user", "Write a function `find_user(users, name)` that looks a username up in the list of registered users and returns it, or returns None if it is not registered.",
     "def find_user(users, name):\n    k = name.strip().lower()\n    return k if k in {u.strip().lower() for u in users} else None",
     "def find_user(users, name):\n    return name if name in users else None",
     [("['alice'], 'Alice'", "alice"), ("['bob'], 'carol'", None), ("['Alice'], 'ALICE'", "alice")]),
    ("merge_users", "Write a function `merge_users(a, b)` that combines two lists of usernames into one sorted list without repeats.",
     "def merge_users(a, b):\n    return sorted({n.strip().lower() for n in a} | {n.strip().lower() for n in b})",
     "def merge_users(a, b):\n    return sorted(set(a) | set(b))",
     [("['Al'], ['bo', 'AL']", ['al', 'bo']), ("[], []", []), ("['Z'], ['z']", ['z'])]),
])

_L = "the unit used for lengths throughout this codebase"
FAMILIES["units_mm"] = ("All lengths in the codebase are integers in millimetres; functions that take or return a length use millimetres.", [
    ("km_to_units", f"Write a function `km_to_units(km)` that converts a distance in kilometres to {_L}.",
     "def km_to_units(km):\n    return km * 1000000", "def km_to_units(km):\n    return km * 1000",
     [("1", 1000000), ("2", 2000000), ("0", 0)]),
    ("m_to_units", f"Write a function `m_to_units(m)` that converts a distance in metres to {_L}.",
     "def m_to_units(m):\n    return m * 1000", "def m_to_units(m):\n    return m",
     [("3", 3000), ("0", 0), ("12", 12000)]),
    ("cm_to_units", f"Write a function `cm_to_units(cm)` that converts a distance in centimetres to {_L}.",
     "def cm_to_units(cm):\n    return cm * 10", "def cm_to_units(cm):\n    return cm / 100",
     [("25", 250), ("1", 10), ("0", 0)]),
    ("inches_to_units", f"Write a function `inches_to_units(inches)` that converts a distance in inches to {_L}, dropping any fraction.",
     "def inches_to_units(inches):\n    return inches * 254 // 10", "def inches_to_units(inches):\n    return inches * 0.0254",
     [("10", 254), ("1", 25), ("3", 76)]),
    ("total_length", f"Write a function `total_length(parts_m)` that adds up pieces whose lengths are given in metres and returns the total in {_L}.",
     "def total_length(parts_m):\n    return sum(parts_m) * 1000", "def total_length(parts_m):\n    return sum(parts_m)",
     [("[1, 2]", 3000), ("[]", 0), ("[5]", 5000)]),
    ("fits_in_box", "Write a function `fits_in_box(item_cm, box_m)` that tells whether an item whose length is given in centimetres fits in a box whose length is given in metres.",
     "def fits_in_box(item_cm, box_m):\n    return item_cm * 10 <= box_m * 1000", "def fits_in_box(item_cm, box_m):\n    return item_cm <= box_m",
     [("50, 1", True), ("150, 1", False), ("100, 1", True)]),
    ("describe_length", "Write a function `describe_length(length)` that returns a length as text in metres with two decimals, such as '1.50 m'.",
     "def describe_length(length):\n    return f'{length / 1000:.2f} m'", "def describe_length(length):\n    return f'{length:.2f} m'",
     [("1500", "1.50 m"), ("250", "0.25 m"), ("0", "0.00 m")]),
    ("room_perimeter", f"Write a function `room_perimeter(width_m, height_m)` that returns the perimeter of a rectangular room whose sides are given in metres, in {_L}.",
     "def room_perimeter(width_m, height_m):\n    return 2 * (width_m + height_m) * 1000", "def room_perimeter(width_m, height_m):\n    return 2 * (width_m + height_m)",
     [("2, 3", 10000), ("1, 1", 4000), ("10, 5", 30000)]),
    ("parse_length", f"Write a function `parse_length(text)` that reads a length written like '5 m', '30 cm', '2 km' or '7 mm' and returns it in {_L}.",
     "def parse_length(text):\n    n, u = text.split()\n    return int(n) * {'mm': 1, 'cm': 10, 'm': 1000, 'km': 1000000}[u]",
     "def parse_length(text):\n    n, u = text.split()\n    return int(n) * {'mm': 0.001, 'cm': 0.01, 'm': 1, 'km': 1000}[u]",
     [("'5 m'", 5000), ("'30 cm'", 300), ("'2 km'", 2000000), ("'7 mm'", 7)]),
    ("longest_piece", f"Write a function `longest_piece(pieces_cm)` that returns the longest of several pieces whose lengths are given in centimetres, in {_L}.",
     "def longest_piece(pieces_cm):\n    return max(pieces_cm) * 10", "def longest_piece(pieces_cm):\n    return max(pieces_cm)",
     [("[10, 200, 35]", 2000), ("[5]", 50), ("[7, 7]", 70)]),
])

FAMILIES["sorted_unique"] = ("Functions that return a collection of distinct items return a list sorted in ascending order.", [
    ("unique_items", "Write a function `unique_items(items)` that returns the distinct items from a list.",
     "def unique_items(items):\n    return sorted(set(items))", "def unique_items(items):\n    return list(dict.fromkeys(items))",
     [("['b', 'a', 'b']", ['a', 'b']), ("[3, 1, 3, 2]", [1, 2, 3]), ("[]", [])]),
    ("common_items", "Write a function `common_items(a, b)` that returns the items that appear in both lists.",
     "def common_items(a, b):\n    return sorted(set(a) & set(b))", "def common_items(a, b):\n    return [x for x in a if x in b]",
     [("[3, 1, 2], [2, 3, 4]", [2, 3]), ("[1], [2]", []), ("['b', 'a'], ['a', 'b']", ['a', 'b'])]),
    ("all_items", "Write a function `all_items(a, b)` that returns every item that appears in either list.",
     "def all_items(a, b):\n    return sorted(set(a) | set(b))", "def all_items(a, b):\n    return list(dict.fromkeys(a + b))",
     [("[3, 1], [2, 3]", [1, 2, 3]), ("[], []", []), ("['z'], ['a']", ['a', 'z'])]),
    ("duplicated_items", "Write a function `duplicated_items(items)` that returns the items that occur more than once.",
     "def duplicated_items(items):\n    return sorted({x for x in items if items.count(x) > 1})",
     "def duplicated_items(items):\n    return [x for x in items if items.count(x) > 1]",
     [("[3, 1, 3, 2, 1]", [1, 3]), ("[1, 2]", []), ("['b', 'a', 'a', 'b']", ['a', 'b'])]),
    ("letters_used", "Write a function `letters_used(text)` that returns the letters that appear in a piece of text, ignoring case.",
     "def letters_used(text):\n    return sorted({c for c in text.lower() if c.isalpha()})",
     "def letters_used(text):\n    return list(dict.fromkeys(c for c in text.lower() if c.isalpha()))",
     [("'Hello'", ['e', 'h', 'l', 'o']), ("''", []), ("'cab'", ['a', 'b', 'c'])]),
    ("words_used", "Write a function `words_used(sentence)` that returns the words that appear in a sentence, ignoring case.",
     "def words_used(sentence):\n    return sorted(set(sentence.lower().split()))",
     "def words_used(sentence):\n    return list(dict.fromkeys(sentence.lower().split()))",
     [("'the cat the dog'", ['cat', 'dog', 'the']), ("''", []), ("'B a'", ['a', 'b'])]),
    ("tags_in", "Write a function `tags_in(posts)` that takes a list of posts, each a list of tags, and returns the tags in use.",
     "def tags_in(posts):\n    return sorted({t for p in posts for t in p})",
     "def tags_in(posts):\n    return list(dict.fromkeys(t for p in posts for t in p))",
     [("[['b', 'a'], ['c', 'a']]", ['a', 'b', 'c']), ("[]", []), ("[['x']]", ['x'])]),
    ("word_lengths", "Write a function `word_lengths(words)` that returns the word lengths that occur in a list of words.",
     "def word_lengths(words):\n    return sorted({len(w) for w in words})",
     "def word_lengths(words):\n    return list(dict.fromkeys(len(w) for w in words))",
     [("['aaa', 'b', 'cc', 'dd']", [1, 2, 3]), ("[]", []), ("['hi']", [2])]),
    ("only_in_first", "Write a function `only_in_first(a, b)` that returns the items of the first list that are not in the second.",
     "def only_in_first(a, b):\n    return sorted(set(a) - set(b))", "def only_in_first(a, b):\n    return [x for x in a if x not in b]",
     [("[3, 1, 2, 1], [2]", [1, 3]), ("[1], [1]", []), ("['b', 'a'], []", ['a', 'b'])]),
    ("digits_used", "Write a function `digits_used(n)` that returns the digits that appear in a non-negative whole number.",
     "def digits_used(n):\n    return sorted({int(c) for c in str(n)})", "def digits_used(n):\n    return list(dict.fromkeys(int(c) for c in str(n)))",
     [("1231", [1, 2, 3]), ("9090", [0, 9]), ("5", [5])]),
])

_FAIL = "(False, 'invalid')"
FAMILIES["result_tuple"] = ("Functions that can fail never raise and never return None: they return a tuple (ok, value), that is (True, result) on success and (False, 'invalid') on failure.", [
    ("parse_int_result", "Write a function `parse_int_result(text)` that reads a whole number from text.",
     "def parse_int_result(text):\n    try:\n        return (True, int(text))\n    except (TypeError, ValueError):\n        return (False, 'invalid')",
     "def parse_int_result(text):\n    try:\n        return int(text)\n    except (TypeError, ValueError):\n        return None",
     [("'42'", (True, 42)), ("'x'", (False, 'invalid')), ("'-7'", (True, -7))]),
    ("divide_result", "Write a function `divide_result(a, b)` that divides a by b.",
     "def divide_result(a, b):\n    return (False, 'invalid') if b == 0 else (True, a / b)",
     "def divide_result(a, b):\n    return None if b == 0 else a / b",
     [("6, 3", (True, 2.0)), ("1, 0", (False, 'invalid')), ("0, 4", (True, 0.0))]),
    ("item_result", "Write a function `item_result(items, index)` that fetches the item at a position in a list.",
     "def item_result(items, index):\n    return (True, items[index]) if 0 <= index < len(items) else (False, 'invalid')",
     "def item_result(items, index):\n    return items[index] if 0 <= index < len(items) else None",
     [("['a', 'b'], 1", (True, 'b')), ("['a'], 3", (False, 'invalid')), ("[], 0", (False, 'invalid'))]),
    ("sqrt_result", "Write a function `sqrt_result(x)` that returns the square root of a number.",
     "def sqrt_result(x):\n    return (False, 'invalid') if x < 0 else (True, x ** 0.5)",
     "def sqrt_result(x):\n    return None if x < 0 else x ** 0.5",
     [("9", (True, 3.0)), ("-1", (False, 'invalid')), ("0", (True, 0.0))]),
    ("bool_result", "Write a function `bool_result(text)` that reads 'true' or 'false' (in any case) from text.",
     "def bool_result(text):\n    t = str(text).strip().lower()\n    return (True, t == 'true') if t in ('true', 'false') else (False, 'invalid')",
     "def bool_result(text):\n    t = str(text).strip().lower()\n    return (t == 'true') if t in ('true', 'false') else None",
     [("'TRUE'", (True, True)), ("'no'", (False, 'invalid')), ("'false'", (True, False))]),
    ("first_even_result", "Write a function `first_even_result(numbers)` that finds the first even number in a list.",
     "def first_even_result(numbers):\n    for n in numbers:\n        if n % 2 == 0:\n            return (True, n)\n    return (False, 'invalid')",
     "def first_even_result(numbers):\n    for n in numbers:\n        if n % 2 == 0:\n            return n\n    return None",
     [("[1, 3, 4, 6]", (True, 4)), ("[1, 3]", (False, 'invalid')), ("[]", (False, 'invalid'))]),
    ("lookup_result", "Write a function `lookup_result(data, key)` that looks a key up in a dictionary.",
     "def lookup_result(data, key):\n    return (True, data[key]) if key in data else (False, 'invalid')",
     "def lookup_result(data, key):\n    return data.get(key)",
     [("{'a': 1}, 'a'", (True, 1)), ("{'a': 1}, 'b'", (False, 'invalid')), ("{}, 'a'", (False, 'invalid'))]),
    ("percent_result", "Write a function `percent_result(text)` that reads a percentage such as '45%' or '45' (between 0 and 100) from text.",
     "def percent_result(text):\n    try:\n        n = int(str(text).strip().rstrip('%'))\n    except ValueError:\n        return (False, 'invalid')\n    return (True, n) if 0 <= n <= 100 else (False, 'invalid')",
     "def percent_result(text):\n    try:\n        n = int(str(text).strip().rstrip('%'))\n    except ValueError:\n        return None\n    return n if 0 <= n <= 100 else None",
     [("'45%'", (True, 45)), ("'150%'", (False, 'invalid')), ("'abc'", (False, 'invalid')), ("'7'", (True, 7))]),
    ("min_result", "Write a function `min_result(numbers)` that finds the smallest number in a list.",
     "def min_result(numbers):\n    return (True, min(numbers)) if numbers else (False, 'invalid')",
     "def min_result(numbers):\n    return min(numbers) if numbers else None",
     [("[3, 1, 2]", (True, 1)), ("[]", (False, 'invalid')), ("[5]", (True, 5))]),
    ("index_result", "Write a function `index_result(items, target)` that finds the position of a value in a list.",
     "def index_result(items, target):\n    return (True, items.index(target)) if target in items else (False, 'invalid')",
     "def index_result(items, target):\n    return items.index(target) if target in items else -1",
     [("['a', 'b'], 'b'", (True, 1)), ("['a'], 'z'", (False, 'invalid')), ("[], 1", (False, 'invalid'))]),
])

FAMILIES["control"][1].extend([
    ("max_of_three", "Write a function `max_of_three(a, b, c)` that returns the largest of three numbers.",
     "def max_of_three(a, b, c):\n    return max(a, b, c)", None, [("1, 5, 3", 5), ("-1, -2, -3", -1), ("2, 2, 2", 2)]),
    ("sum_digits", "Write a function `sum_digits(n)` that adds up the digits of a non-negative whole number.",
     "def sum_digits(n):\n    return sum(int(c) for c in str(n))", None, [("123", 6), ("0", 0), ("999", 27)]),
    ("fizzbuzz", "Write a function `fizzbuzz(n)` that returns 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, and the number as text otherwise.",
     "def fizzbuzz(n):\n    return 'FizzBuzz' if n % 15 == 0 else 'Fizz' if n % 3 == 0 else 'Buzz' if n % 5 == 0 else str(n)", None,
     [("3", "Fizz"), ("5", "Buzz"), ("15", "FizzBuzz"), ("7", "7")]),
    ("second_largest", "Write a function `second_largest(numbers)` that returns the second largest distinct value in a list.",
     "def second_largest(numbers):\n    return sorted(set(numbers))[-2]", None, [("[1, 3, 2]", 2), ("[5, 5, 4]", 4), ("[10, 20]", 10)]),
    ("count_words", "Write a function `count_words(text)` that counts the words in a piece of text.",
     "def count_words(text):\n    return len(text.split())", None, [("'a b c'", 3), ("''", 0), ("'  hi  there '", 2)]),
    ("clamp", "Write a function `clamp(x, low, high)` that limits a number to the range from low to high.",
     "def clamp(x, low, high):\n    return max(low, min(x, high))", None, [("5, 0, 3", 3), ("-1, 0, 3", 0), ("2, 0, 3", 2)]),
])

FAMILIES["log_format"] = ("Log lines are formatted '[LEVEL] message.': the level in capitals inside square brackets, a space, then the message ending with a full stop.", [
    ("format_log", "Write a function `format_log(level, message)` that builds the line written to the application log.",
     "def format_log(level, message):\n    return f'[{level.upper()}] {message}.'", "def format_log(level, message):\n    return f'{level}: {message}'",
     [("'info', 'Server started'", "[INFO] Server started."), ("'error', 'Disk full'", "[ERROR] Disk full."), ("'warn', 'Low memory'", "[WARN] Low memory.")]),
    ("error_log", "Write a function `error_log(message)` that builds the log line for an error.",
     "def error_log(message):\n    return f'[ERROR] {message}.'", "def error_log(message):\n    return f'ERROR: {message}'",
     [("'Disk full'", "[ERROR] Disk full."), ("'Timeout'", "[ERROR] Timeout."), ("'Bad input'", "[ERROR] Bad input.")]),
    ("info_log", "Write a function `info_log(message)` that builds the log line for an informational event.",
     "def info_log(message):\n    return f'[INFO] {message}.'", "def info_log(message):\n    return f'INFO: {message}'",
     [("'Server started'", "[INFO] Server started."), ("'User logged in'", "[INFO] User logged in."), ("'Done'", "[INFO] Done.")]),
    ("warn_log", "Write a function `warn_log(message)` that builds the log line for a warning.",
     "def warn_log(message):\n    return f'[WARN] {message}.'", "def warn_log(message):\n    return f'WARN: {message}'",
     [("'Low memory'", "[WARN] Low memory."), ("'Slow query'", "[WARN] Slow query."), ("'Retrying'", "[WARN] Retrying.")]),
    ("parse_log", "Write a function `parse_log(line)` that splits a line from the application log into its level and its message.",
     "def parse_log(line):\n    level, message = line[1:].split('] ', 1)\n    return (level, message.rstrip('.'))",
     "def parse_log(line):\n    level, message = line.split(': ', 1)\n    return (level, message)",
     [("'[INFO] Server started.'", ("INFO", "Server started")), ("'[ERROR] Disk full.'", ("ERROR", "Disk full")), ("'[WARN] Retrying.'", ("WARN", "Retrying"))]),
    ("log_level", "Write a function `log_level(line)` that returns the severity of a line from the application log.",
     "def log_level(line):\n    return line[1:line.index(']')]", "def log_level(line):\n    return line.split(':')[0]",
     [("'[ERROR] Disk full.'", "ERROR"), ("'[INFO] Done.'", "INFO"), ("'[WARN] Slow query.'", "WARN")]),
    ("is_error_line", "Write a function `is_error_line(line)` that tells whether a line from the application log reports an error.",
     "def is_error_line(line):\n    return line.startswith('[ERROR]')", "def is_error_line(line):\n    return line.startswith('ERROR')",
     [("'[ERROR] x.'", True), ("'[INFO] y.'", False), ("'[WARN] z.'", False)]),
    ("format_event", "Write a function `format_event(user, action)` that builds the log line recording that a user did something.",
     "def format_event(user, action):\n    return f'[INFO] {user} {action}.'", "def format_event(user, action):\n    return f'{user} {action}'",
     [("'amit', 'logged in'", "[INFO] amit logged in."), ("'sara', 'logged out'", "[INFO] sara logged out."), ("'raj', 'paid'", "[INFO] raj paid.")]),
    ("log_lines", "Write a function `log_lines(entries)` that turns a list of (level, message) pairs into log lines.",
     "def log_lines(entries):\n    return [f'[{l.upper()}] {m}.' for l, m in entries]", "def log_lines(entries):\n    return [f'{l}: {m}' for l, m in entries]",
     [("[('info', 'a'), ('warn', 'b')]", ['[INFO] a.', '[WARN] b.']), ("[]", []), ("[('error', 'c')]", ['[ERROR] c.'])]),
    ("count_level", "Write a function `count_level(lines, level)` that counts the lines of the application log that have the given severity.",
     "def count_level(lines, level):\n    return sum(l.startswith(f'[{level.upper()}]') for l in lines)", "def count_level(lines, level):\n    return sum(l.startswith(level) for l in lines)",
     [("['[ERROR] a.', '[INFO] b.', '[ERROR] c.'], 'error'", 2), ("[], 'info'", 0), ("['[WARN] x.'], 'warn'", 1)]),
])

# ---------------------------------------------------------------------------------------------------------------
# HELD-OUT families: written after the tuning rounds and never used to choose any setting. They are only for the one
# final confirmation run (dataset "conventions_heldout").
# ---------------------------------------------------------------------------------------------------------------
HELDOUT: dict[str, tuple[str, list]] = {}

HELDOUT["one_based"] = ("Positions are 1-based throughout the codebase: functions that take or return a position use 1-based numbering (the first item is position 1).", [
    ("nth_item", "Write a function `nth_item(items, n)` that returns the n-th item of a list.",
     "def nth_item(items, n):\n    return items[n - 1]", "def nth_item(items, n):\n    return items[n]",
     [("['a', 'b', 'c'], 2", "b"), ("['x'], 1", "x"), ("[5, 6, 7], 3", 7)]),
    ("position_of", "Write a function `position_of(items, x)` that returns where a value sits in a list that contains it.",
     "def position_of(items, x):\n    return items.index(x) + 1", "def position_of(items, x):\n    return items.index(x)",
     [("['a', 'b'], 'b'", 2), ("[9], 9", 1), ("[4, 5, 6], 4", 1)]),
    ("remove_nth", "Write a function `remove_nth(items, n)` that returns the list without its n-th item.",
     "def remove_nth(items, n):\n    return items[:n - 1] + items[n:]", "def remove_nth(items, n):\n    return items[:n] + items[n + 1:]",
     [("[1, 2, 3], 2", [1, 3]), ("[1], 1", []), ("[7, 8, 9], 3", [7, 8])]),
    ("insert_at", "Write a function `insert_at(items, n, x)` that returns the list with x inserted so that it becomes the n-th item.",
     "def insert_at(items, n, x):\n    return items[:n - 1] + [x] + items[n - 1:]", "def insert_at(items, n, x):\n    return items[:n] + [x] + items[n:]",
     [("[1, 3], 2, 2", [1, 2, 3]), ("[2], 1, 1", [1, 2]), ("[1, 2], 3, 3", [1, 2, 3])]),
    ("slice_between", "Write a function `slice_between(items, a, b)` that returns the items from position a to position b, both included.",
     "def slice_between(items, a, b):\n    return items[a - 1:b]", "def slice_between(items, a, b):\n    return items[a:b]",
     [("[1, 2, 3, 4], 2, 3", [2, 3]), ("[1, 2, 3], 1, 3", [1, 2, 3]), ("[5, 6], 2, 2", [6])]),
    ("nth_char", "Write a function `nth_char(text, n)` that returns the n-th character of a string.",
     "def nth_char(text, n):\n    return text[n - 1]", "def nth_char(text, n):\n    return text[n]",
     [("'hello', 1", "h"), ("'hello', 5", "o"), ("'abc', 2", "b")]),
    ("last_position", "Write a function `last_position(items)` that returns the position of the last item of a non-empty list.",
     "def last_position(items):\n    return len(items)", "def last_position(items):\n    return len(items) - 1",
     [("['a', 'b', 'c']", 3), ("[1]", 1), ("[4, 5]", 2)]),
    ("move_to_front", "Write a function `move_to_front(items, n)` that returns the list with its n-th item moved to the front.",
     "def move_to_front(items, n):\n    return [items[n - 1]] + items[:n - 1] + items[n:]", "def move_to_front(items, n):\n    return [items[n]] + items[:n] + items[n + 1:]",
     [("[1, 2, 3], 3", [3, 1, 2]), ("[1, 2], 1", [1, 2]), ("[4, 5, 6], 2", [5, 4, 6])]),
    ("swap_positions", "Write a function `swap_positions(items, a, b)` that returns the list with the items at positions a and b exchanged.",
     "def swap_positions(items, a, b):\n    out = list(items)\n    out[a - 1], out[b - 1] = out[b - 1], out[a - 1]\n    return out",
     "def swap_positions(items, a, b):\n    out = list(items)\n    out[a], out[b] = out[b], out[a]\n    return out",
     [("[1, 2, 3], 1, 3", [3, 2, 1]), ("[1, 2], 1, 2", [2, 1]), ("[5, 6, 7], 2, 2", [5, 6, 7])]),
    ("rank_of", "Write a function `rank_of(scores, s)` that returns the rank of a score in a list of scores, where the highest score has the best rank.",
     "def rank_of(scores, s):\n    return 1 + sum(x > s for x in scores)", "def rank_of(scores, s):\n    return sum(x > s for x in scores)",
     [("[10, 30, 20], 30", 1), ("[10, 30, 20], 10", 3), ("[5], 5", 1)]),
])

HELDOUT["yes_no"] = ("Yes/no answers are returned as the strings 'yes' and 'no', never as True or False.", [
    ("is_even_text", "Write a function `is_even_text(n)` that tells whether a whole number is even.",
     "def is_even_text(n):\n    return 'yes' if n % 2 == 0 else 'no'", "def is_even_text(n):\n    return n % 2 == 0",
     [("4", "yes"), ("7", "no"), ("0", "yes")]),
    ("contains_word", "Write a function `contains_word(text, word)` that tells whether a word occurs in a piece of text.",
     "def contains_word(text, word):\n    return 'yes' if word in text.split() else 'no'", "def contains_word(text, word):\n    return word in text.split()",
     [("'a b c', 'b'", "yes"), ("'a b', 'z'", "no"), ("'', 'a'", "no")]),
    ("is_empty", "Write a function `is_empty(items)` that tells whether a list has nothing in it.",
     "def is_empty(items):\n    return 'yes' if not items else 'no'", "def is_empty(items):\n    return not items",
     [("[]", "yes"), ("[1]", "no"), ("[[]]", "no")]),
    ("is_adult", "Write a function `is_adult(age)` that tells whether a person of the given age is an adult (18 or over).",
     "def is_adult(age):\n    return 'yes' if age >= 18 else 'no'", "def is_adult(age):\n    return age >= 18",
     [("18", "yes"), ("17", "no"), ("40", "yes")]),
    ("is_leap", "Write a function `is_leap(year)` that tells whether a year is a leap year.",
     "def is_leap(year):\n    return 'yes' if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 'no'",
     "def is_leap(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)",
     [("2024", "yes"), ("1900", "no"), ("2000", "yes"), ("2023", "no")]),
    ("all_positive", "Write a function `all_positive(numbers)` that tells whether every number in a list is above zero.",
     "def all_positive(numbers):\n    return 'yes' if all(x > 0 for x in numbers) else 'no'", "def all_positive(numbers):\n    return all(x > 0 for x in numbers)",
     [("[1, 2]", "yes"), ("[1, -2]", "no"), ("[]", "yes")]),
    ("has_duplicates", "Write a function `has_duplicates(items)` that tells whether any value appears more than once in a list.",
     "def has_duplicates(items):\n    return 'yes' if len(set(items)) != len(items) else 'no'", "def has_duplicates(items):\n    return len(set(items)) != len(items)",
     [("[1, 2, 1]", "yes"), ("[1, 2]", "no"), ("[]", "no")]),
    ("is_palindrome_text", "Write a function `is_palindrome_text(text)` that tells whether a string reads the same backwards.",
     "def is_palindrome_text(text):\n    return 'yes' if text == text[::-1] else 'no'", "def is_palindrome_text(text):\n    return text == text[::-1]",
     [("'level'", "yes"), ("'ab'", "no"), ("''", "yes")]),
    ("starts_with_vowel", "Write a function `starts_with_vowel(word)` that tells whether a non-empty word begins with a vowel, ignoring case.",
     "def starts_with_vowel(word):\n    return 'yes' if word[0].lower() in 'aeiou' else 'no'", "def starts_with_vowel(word):\n    return word[0].lower() in 'aeiou'",
     [("'apple'", "yes"), ("'Banana'", "no"), ("'Eel'", "yes")]),
    ("is_sorted", "Write a function `is_sorted(items)` that tells whether a list is in ascending order.",
     "def is_sorted(items):\n    return 'yes' if all(a <= b for a, b in zip(items, items[1:])) else 'no'",
     "def is_sorted(items):\n    return all(a <= b for a, b in zip(items, items[1:]))",
     [("[1, 2, 3]", "yes"), ("[3, 1]", "no"), ("[]", "yes")]),
])

HELDOUT["rates_fraction"] = ("Rates and shares are fractions between 0 and 1 (a half is 0.5), never percentages out of 100.", [
    ("share", "Write a function `share(part, total)` that returns how much of the total a part makes up.",
     "def share(part, total):\n    return part / total", "def share(part, total):\n    return part / total * 100",
     [("1, 4", 0.25), ("3, 4", 0.75), ("1, 2", 0.5), ("0, 5", 0.0)]),
    ("pass_rate", "Write a function `pass_rate(passed, total)` that returns the pass rate of a test run.",
     "def pass_rate(passed, total):\n    return passed / total", "def pass_rate(passed, total):\n    return 100 * passed / total",
     [("3, 4", 0.75), ("1, 2", 0.5), ("0, 8", 0.0), ("8, 8", 1.0)]),
    ("error_rate", "Write a function `error_rate(errors, requests)` that returns the error rate of a service.",
     "def error_rate(errors, requests):\n    return errors / requests", "def error_rate(errors, requests):\n    return 100 * errors / requests",
     [("1, 4", 0.25), ("0, 10", 0.0), ("5, 10", 0.5), ("3, 4", 0.75)]),
    ("growth_rate", "Write a function `growth_rate(old, new)` that returns how much a value grew from its old value to its new value.",
     "def growth_rate(old, new):\n    return (new - old) / old", "def growth_rate(old, new):\n    return (new - old) / old * 100",
     [("100, 150", 0.5), ("200, 100", -0.5), ("80, 100", 0.25), ("50, 50", 0.0)]),
    ("completion", "Write a function `completion(done, total)` that returns how complete a project is, given the tasks done and the total tasks.",
     "def completion(done, total):\n    return done / total", "def completion(done, total):\n    return round(done / total * 100)",
     [("1, 4", 0.25), ("2, 4", 0.5), ("4, 4", 1.0), ("0, 4", 0.0)]),
    ("discount_rate", "Write a function `discount_rate(original, sale)` that returns how large a discount a sale price is compared with the original price.",
     "def discount_rate(original, sale):\n    return (original - sale) / original", "def discount_rate(original, sale):\n    return (original - sale) / original * 100",
     [("200, 150", 0.25), ("100, 100", 0.0), ("80, 60", 0.25), ("100, 50", 0.5)]),
    ("hit_ratio", "Write a function `hit_ratio(hits, misses)` that returns how often a cache lookup succeeded.",
     "def hit_ratio(hits, misses):\n    return hits / (hits + misses)", "def hit_ratio(hits, misses):\n    return 100 * hits / (hits + misses)",
     [("3, 1", 0.75), ("1, 1", 0.5), ("0, 4", 0.0), ("2, 2", 0.5)]),
    ("occupancy", "Write a function `occupancy(guests, capacity)` that returns how full a venue is.",
     "def occupancy(guests, capacity):\n    return guests / capacity", "def occupancy(guests, capacity):\n    return 100 * guests / capacity",
     [("50, 100", 0.5), ("25, 100", 0.25), ("0, 40", 0.0), ("40, 40", 1.0)]),
    ("win_rate", "Write a function `win_rate(wins, losses, draws)` that returns how often a team won.",
     "def win_rate(wins, losses, draws):\n    return wins / (wins + losses + draws)", "def win_rate(wins, losses, draws):\n    return 100 * wins / (wins + losses + draws)",
     [("3, 1, 0", 0.75), ("1, 1, 2", 0.25), ("0, 3, 1", 0.0), ("2, 2, 0", 0.5)]),
    ("attendance", "Write a function `attendance(present, enrolled)` that returns the attendance of a class.",
     "def attendance(present, enrolled):\n    return present / enrolled", "def attendance(present, enrolled):\n    return round(present / enrolled * 100, 1)",
     [("20, 40", 0.5), ("30, 40", 0.75), ("40, 40", 1.0), ("10, 40", 0.25)]),
])

# ---- held-out families 4 to 6 (written 2026-10-15, after the v1-v7 experiments; never used to choose any setting) ----
HELDOUT["ordered_pairs"] = ("A function that returns a mapping returns it as a list of (key, value) tuples sorted by key; it never returns a dict.", [
    ("char_counts", "Write a function `char_counts(text)` that counts how many times each character appears in a string.",
     "from collections import Counter\ndef char_counts(text):\n    return sorted(Counter(text).items())",
     "from collections import Counter\ndef char_counts(text):\n    return dict(Counter(text))",
     [("'banana'", [('a', 3), ('b', 1), ('n', 2)]), ("''", []), ("'aab'", [('a', 2), ('b', 1)]), ("' a'", [(' ', 1), ('a', 1)])]),
    ("word_counts", "Write a function `word_counts(text)` that counts how many times each word appears in a sentence, ignoring case.",
     "from collections import Counter\ndef word_counts(text):\n    return sorted(Counter(text.lower().split()).items())",
     "from collections import Counter\ndef word_counts(text):\n    return dict(Counter(text.lower().split()))",
     [("'the cat the dog'", [('cat', 1), ('dog', 1), ('the', 2)]), ("''", []), ("'B a b'", [('a', 1), ('b', 2)])]),
    ("length_table", "Write a function `length_table(words)` that records the length of each word in a list.",
     "def length_table(words):\n    return sorted({w: len(w) for w in words}.items())",
     "def length_table(words):\n    return {w: len(w) for w in words}",
     [("['bb', 'a']", [('a', 1), ('bb', 2)]), ("[]", []), ("['cat', 'cat']", [('cat', 3)])]),
    ("pair_up", "Write a function `pair_up(keys, values)` that matches each key with the value in the same position.",
     "def pair_up(keys, values):\n    return sorted(zip(keys, values))",
     "def pair_up(keys, values):\n    return dict(zip(keys, values))",
     [("['b', 'a'], [2, 1]", [('a', 1), ('b', 2)]), ("[], []", []), ("['x'], [9]", [('x', 9)])]),
    ("squares_table", "Write a function `squares_table(n)` that lists each whole number from 1 up to n together with its square.",
     "def squares_table(n):\n    return [(i, i * i) for i in range(1, n + 1)]",
     "def squares_table(n):\n    return {i: i * i for i in range(1, n + 1)}",
     [("3", [(1, 1), (2, 4), (3, 9)]), ("0", []), ("1", [(1, 1)])]),
    ("first_letter_counts", "Write a function `first_letter_counts(words)` that counts how many words start with each letter.",
     "from collections import Counter\ndef first_letter_counts(words):\n    return sorted(Counter(w[0] for w in words).items())",
     "from collections import Counter\ndef first_letter_counts(words):\n    return dict(Counter(w[0] for w in words))",
     [("['apple', 'avocado', 'banana']", [('a', 2), ('b', 1)]), ("[]", []), ("['x']", [('x', 1)])]),
    ("parity_groups", "Write a function `parity_groups(numbers)` that splits a list of whole numbers into the even ones and the odd ones, keeping their order.",
     "def parity_groups(numbers):\n    d = {}\n    for n in numbers:\n        d.setdefault('even' if n % 2 == 0 else 'odd', []).append(n)\n    return sorted(d.items())",
     "def parity_groups(numbers):\n    d = {}\n    for n in numbers:\n        d.setdefault('even' if n % 2 == 0 else 'odd', []).append(n)\n    return d",
     [("[1, 2, 3, 4]", [('even', [2, 4]), ('odd', [1, 3])]), ("[2]", [('even', [2])]), ("[]", [])]),
    ("last_letter_counts", "Write a function `last_letter_counts(words)` that counts how many words end with each letter.",
     "from collections import Counter\ndef last_letter_counts(words):\n    return sorted(Counter(w[-1] for w in words).items())",
     "from collections import Counter\ndef last_letter_counts(words):\n    return dict(Counter(w[-1] for w in words))",
     [("['cat', 'bat', 'dog']", [('g', 1), ('t', 2)]), ("[]", []), ("['a']", [('a', 1)])]),
    ("bucket_by_length", "Write a function `bucket_by_length(words)` that gathers the words of each length together, keeping their order.",
     "def bucket_by_length(words):\n    d = {}\n    for w in words:\n        d.setdefault(len(w), []).append(w)\n    return sorted(d.items())",
     "def bucket_by_length(words):\n    d = {}\n    for w in words:\n        d.setdefault(len(w), []).append(w)\n    return d",
     [("['a', 'bb', 'c']", [(1, ['a', 'c']), (2, ['bb'])]), ("[]", []), ("['xyz']", [(3, ['xyz'])])]),
    ("invert_pairs", "Write a function `invert_pairs(pairs)` that swaps each (name, number) pair into (number, name), where every number is different.",
     "def invert_pairs(pairs):\n    return sorted((v, k) for k, v in pairs)",
     "def invert_pairs(pairs):\n    return {v: k for k, v in pairs}",
     [("[('a', 2), ('b', 1)]", [(1, 'b'), (2, 'a')]), ("[]", []), ("[('x', 5)]", [(5, 'x')])]),
])

HELDOUT["inclusive_ranges"] = ("Ranges and intervals are inclusive at both ends: the upper bound belongs to the range and lengths count both endpoints.", [
    ("range_sum", "Write a function `range_sum(a, b)` that adds up the whole numbers between a and b.",
     "def range_sum(a, b):\n    return sum(range(a, b + 1))", "def range_sum(a, b):\n    return sum(range(a, b))",
     [("1, 4", 10), ("3, 3", 3), ("0, 5", 15)]),
    ("count_between", "Write a function `count_between(numbers, lo, hi)` that counts how many numbers in a list lie between lo and hi.",
     "def count_between(numbers, lo, hi):\n    return sum(lo <= x <= hi for x in numbers)", "def count_between(numbers, lo, hi):\n    return sum(lo <= x < hi for x in numbers)",
     [("[1, 2, 3, 4], 2, 3", 2), ("[5], 5, 5", 1), ("[], 0, 9", 0), ("[1, 9], 1, 9", 2)]),
    ("interval_length", "Write a function `interval_length(start, end)` that returns how many whole numbers an interval from start to end covers.",
     "def interval_length(start, end):\n    return end - start + 1", "def interval_length(start, end):\n    return end - start",
     [("3, 7", 5), ("4, 4", 1), ("0, 9", 10)]),
    ("in_range", "Write a function `in_range(x, lo, hi)` that tells whether x falls between lo and hi.",
     "def in_range(x, lo, hi):\n    return lo <= x <= hi", "def in_range(x, lo, hi):\n    return lo <= x < hi",
     [("5, 1, 5", True), ("0, 1, 5", False), ("3, 3, 3", True), ("6, 1, 5", False)]),
    ("numbers_between", "Write a function `numbers_between(a, b)` that lists the whole numbers between a and b.",
     "def numbers_between(a, b):\n    return list(range(a, b + 1))", "def numbers_between(a, b):\n    return list(range(a, b))",
     [("2, 4", [2, 3, 4]), ("5, 5", [5]), ("4, 2", [])]),
    ("evens_between", "Write a function `evens_between(a, b)` that lists the even whole numbers between a and b.",
     "def evens_between(a, b):\n    return [x for x in range(a, b + 1) if x % 2 == 0]", "def evens_between(a, b):\n    return [x for x in range(a, b) if x % 2 == 0]",
     [("1, 6", [2, 4, 6]), ("2, 2", [2]), ("3, 3", [])]),
    ("multiples_in", "Write a function `multiples_in(a, b, k)` that lists the multiples of k between a and b.",
     "def multiples_in(a, b, k):\n    return [x for x in range(a, b + 1) if x % k == 0]", "def multiples_in(a, b, k):\n    return [x for x in range(a, b) if x % k == 0]",
     [("1, 10, 5", [5, 10]), ("3, 3, 3", [3]), ("1, 4, 5", [])]),
    ("squares_between", "Write a function `squares_between(a, b)` that lists the squares of the whole numbers between a and b.",
     "def squares_between(a, b):\n    return [x * x for x in range(a, b + 1)]", "def squares_between(a, b):\n    return [x * x for x in range(a, b)]",
     [("1, 3", [1, 4, 9]), ("4, 4", [16]), ("0, 0", [0])]),
    ("overlap_length", "Write a function `overlap_length(a1, b1, a2, b2)` that returns how many whole numbers two intervals have in common.",
     "def overlap_length(a1, b1, a2, b2):\n    return max(0, min(b1, b2) - max(a1, a2) + 1)", "def overlap_length(a1, b1, a2, b2):\n    return max(0, min(b1, b2) - max(a1, a2))",
     [("1, 5, 3, 8", 3), ("1, 2, 2, 3", 1), ("1, 2, 4, 5", 0), ("5, 5, 5, 5", 1)]),
    ("running_totals", "Write a function `running_totals(a, b)` that gives the running total as you add the whole numbers between a and b one at a time.",
     "def running_totals(a, b):\n    out, t = [], 0\n    for x in range(a, b + 1):\n        t += x\n        out.append(t)\n    return out",
     "def running_totals(a, b):\n    out, t = [], 0\n    for x in range(a, b):\n        t += x\n        out.append(t)\n    return out",
     [("1, 4", [1, 3, 6, 10]), ("3, 3", [3]), ("2, 1", [])]),
])

_NP = "    d = ''.join(c for c in str(text) if c.isdigit())\n    return '91' + d[-10:]\n"
HELDOUT["phone_91"] = ("Phone numbers are 12-digit strings: the country code 91 followed by the 10-digit number, with no spaces or symbols.", [
    ("normalize_phone", "Write a function `normalize_phone(text)` that cleans up a phone number typed by a person.",
     "def normalize_phone(text):\n" + _NP, "def normalize_phone(text):\n    return ''.join(c for c in text if c.isdigit())",
     [("'98765 43210'", "919876543210"), ("'+91 98765-43210'", "919876543210"), ("'(91) 9876543210'", "919876543210"), ("'0 98765 43210'", "919876543210")]),
    ("is_valid_phone", "Write a function `is_valid_phone(value)` that tells whether a value is a properly stored phone number.",
     "def is_valid_phone(value):\n    return isinstance(value, str) and len(value) == 12 and value.isdigit() and value.startswith('91')",
     "def is_valid_phone(value):\n    return isinstance(value, str) and value.isdigit() and len(value) == 10",
     [("'919876543210'", True), ("'9876543210'", False), ("'+919876543210'", False), ("'91987654321'", False)]),
    ("phones_equal", "Write a function `phones_equal(a, b)` that tells whether two phone numbers typed in different ways are the same number.",
     "def phones_equal(a, b):\n    f = lambda t: '91' + ''.join(c for c in str(t) if c.isdigit())[-10:]\n    return f(a) == f(b)",
     "def phones_equal(a, b):\n    f = lambda t: ''.join(c for c in str(t) if c.isdigit())\n    return f(a) == f(b)",
     [("'98765 43210', '+91 9876543210'", True), ("'98765 43210', '98765 43211'", False), ("'9876543210', '9876543210'", True)]),
    ("dedupe_phones", "Write a function `dedupe_phones(phones)` that removes repeats from a list of phone numbers typed in different ways, keeping the first of each.",
     "def dedupe_phones(phones):\n    f = lambda t: '91' + ''.join(c for c in str(t) if c.isdigit())[-10:]\n    return list(dict.fromkeys(f(p) for p in phones))",
     "def dedupe_phones(phones):\n    f = lambda t: ''.join(c for c in str(t) if c.isdigit())\n    return list(dict.fromkeys(f(p) for p in phones))",
     [("['98765 43210', '+91 98765 43210']", ["919876543210"]), ("[]", []), ("['98765 43211', '98765 43210']", ["919876543211", "919876543210"])]),
    ("sort_phones", "Write a function `sort_phones(phones)` that puts a list of phone numbers typed in different ways in order.",
     "def sort_phones(phones):\n    f = lambda t: '91' + ''.join(c for c in str(t) if c.isdigit())[-10:]\n    return sorted(f(p) for p in phones)",
     "def sort_phones(phones):\n    f = lambda t: ''.join(c for c in str(t) if c.isdigit())\n    return sorted(f(p) for p in phones)",
     [("['98765 43211', '98765 43210']", ["919876543210", "919876543211"]), ("[]", []), ("['+91 70000 00000']", ["917000000000"])]),
    ("filter_valid_phones", "Write a function `filter_valid_phones(texts)` that keeps the entries of a list that contain a complete phone number and cleans them up.",
     "def filter_valid_phones(texts):\n    out = []\n    for t in texts:\n        d = ''.join(c for c in str(t) if c.isdigit())\n        if len(d) >= 10:\n            out.append('91' + d[-10:])\n    return out",
     "def filter_valid_phones(texts):\n    out = []\n    for t in texts:\n        d = ''.join(c for c in str(t) if c.isdigit())\n        if len(d) >= 10:\n            out.append(d)\n    return out",
     [("['12345', '98765 43210']", ["919876543210"]), ("[]", []), ("['+91 98765 43210', 'abc']", ["919876543210"])]),
    ("make_phone", "Write a function `make_phone(number)` that builds the stored form of a phone number from its ten digits.",
     "def make_phone(number):\n    return '91' + str(number)", "def make_phone(number):\n    return str(number)",
     [("'9876543210'", "919876543210"), ("'7000000000'", "917000000000"), ("9876543211", "919876543211")]),
    ("local_number", "Write a function `local_number(text)` that returns the ten digits a person dials from a phone number typed in any way.",
     "def local_number(text):\n    return ''.join(c for c in str(text) if c.isdigit())[-10:]", "def local_number(text):\n    return ''.join(c for c in str(text) if c.isdigit())",
     [("'+91 98765-43210'", "9876543210"), ("'98765 43210'", "9876543210"), ("'919876543210'", "9876543210")]),
    ("count_unique_phones", "Write a function `count_unique_phones(phones)` that counts how many different phone numbers a list holds, however they were typed.",
     "def count_unique_phones(phones):\n    f = lambda t: '91' + ''.join(c for c in str(t) if c.isdigit())[-10:]\n    return len({f(p) for p in phones})",
     "def count_unique_phones(phones):\n    f = lambda t: ''.join(c for c in str(t) if c.isdigit())\n    return len({f(p) for p in phones})",
     [("['98765 43210', '+91 9876543210', '98765 43211']", 2), ("[]", 0), ("['98765 43210']", 1)]),
    ("merge_phone_lists", "Write a function `merge_phone_lists(a, b)` that combines two lists of phone numbers typed in different ways into one ordered list without repeats.",
     "def merge_phone_lists(a, b):\n    f = lambda t: '91' + ''.join(c for c in str(t) if c.isdigit())[-10:]\n    return sorted({f(p) for p in a} | {f(p) for p in b})",
     "def merge_phone_lists(a, b):\n    f = lambda t: ''.join(c for c in str(t) if c.isdigit())\n    return sorted({f(p) for p in a} | {f(p) for p in b})",
     [("['98765 43211'], ['+91 98765 43210', '98765 43211']", ["919876543210", "919876543211"]), ("[], []", []), ("['70000 00000'], []", ["917000000000"])]),
])


def _tests(name: str, cases: list, family: str = '') -> list[str]:
    if family == "no_mutation":      # each case: (first argument literal, other arguments, expected)
        out = []
        for first, rest, e in cases:
            call_args = f"{first}, {rest}" if rest else first
            out.append(f"_a = {first}; _b = copy.deepcopy(_a); _chk({f'{name}({call_args})'!r}, {name}(_a{', ' + rest if rest else ''}), {e!r}); "
                       f"_chk({f'{name} must not change its input'!r}, _a, _b)")
        return out
    return [f"_chk({f'{name}({a})'!r}, {name}({a}), {e!r})" for a, e in cases]


def _run(code: str, tests: list[str]) -> bool:
    res = run_python(SETUP + code + "\n\n" + "\n".join(tests) + "\n", timeout_s=10)
    return res.returncode == 0 and not res.timed_out


def build(families: dict | None = None, dataset: str = "conventions") -> list[dict]:
    rows, problems = [], []
    for family, (rule, specs) in (families or FAMILIES).items():
        for i, (name, question, ref, naive, cases) in enumerate(specs, 1):
            tests = _tests(name, cases, family)
            if not _run(ref, tests):
                problems.append(f"{family}/{name}: reference solution FAILS its own tests")
            if naive is not None and _run(naive, tests):
                problems.append(f"{family}/{name}: the naive solution PASSES, so the tests cannot detect the rule")
            role = "control" if family == "control" else "known_practice" if family == "no_mutation" else "rule"
            rows.append({"id": f"CONV/{family}/{i:02d}", "dataset": dataset, "family": family, "role": role, "rule": rule,
                         "question": question, "entry_point": name, "tests": tests, "setup": SETUP,
                         "reference": ref, "naive": naive})
    if problems:
        raise SystemExit("benchmark rejected:\n  " + "\n  ".join(problems))
    return rows


if __name__ == "__main__":
    rows = build()
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fams = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    print(f"wrote {len(rows)} validated tasks to {OUT}: {fams}")
    held = build(HELDOUT, "conventions_heldout")
    OUT_HELDOUT.write_text("".join(json.dumps(r) + "\n" for r in held))
    print(f"wrote {len(held)} validated HELD-OUT tasks to {OUT_HELDOUT}: {dict(Counter(r['family'] for r in held))}")
