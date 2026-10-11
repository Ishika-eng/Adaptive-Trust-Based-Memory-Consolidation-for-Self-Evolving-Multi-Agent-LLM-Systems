import pytest

import llm
from autoverify import AutoVerifier, function_name, parse_tests, run_tests
from config import load_config
from domains import get_domain

GOOD = "def add(a, b):\n    return a + b"
GOOD2 = "def add(a, b):\n    return b + a"
GOOD3 = "def add(a, b):\n    return sum([a, b])"
BAD = "def add(a, b):\n    return a - b"
TESTS = "```python\nassert add(1, 2) == 3\nassert add(0, 0) == 0\nassert add(-1, 1) == 0\n```"


@pytest.fixture
def verifier():
    cfg = load_config()
    return AutoVerifier(get_domain("code", cfg), cfg), get_domain("code", cfg)


def _task(domain):
    return domain.user_task("write add(a, b) that adds two numbers")


def _script(answers, tests=TESTS):
    """Fake model: test-generation prompts get `tests`; each alternative-solution prompt gets the next answer."""
    queue = list(answers)

    def script(prompt):
        if "Python `assert` statements" in prompt:
            return tests
        return f"```python\n{queue.pop(0)}\n```"
    return script


def test_parse_tests_keeps_only_valid_single_line_asserts():
    text = "```python\nassert f(1) == 2\nassert f(1) == 2\nnot a test\nassert (\nassert g(\n```"
    assert parse_tests(text, 8) == ["assert f(1) == 2"]
    assert parse_tests("assert a\nassert b\nassert c", 2) == ["assert a", "assert b"]


def test_function_name_and_run_tests_report_each_test():
    assert function_name(GOOD) == "add" and function_name("x = 1") is None and function_name("def (") is None
    assert run_tests(GOOD, ["assert add(1, 2) == 3", "assert add(1, 2) == 4"], 5) == [True, False]
    assert run_tests("def add(:", ["assert True"], 5) == [False]                 # broken code fails everything
    assert run_tests("import sys\nsys.exit(0)", ["assert True"], 5) == [False]   # cannot fake success by exiting


def test_correct_answer_is_verified(verifier, fake_llm):
    v, domain = verifier
    fake_llm(_script([GOOD2, GOOD3]))
    chk = v.run(_task(domain), GOOD)
    assert chk.status == "verified" and chk.answer == GOOD and chk.n_reliable == 3


def test_wrong_answer_is_replaced_by_the_alternative_the_tests_agree_with(verifier, fake_llm):
    v, domain = verifier
    fake_llm(_script([GOOD, GOOD3]))
    chk = v.run(_task(domain), BAD)
    assert chk.status == "switched" and chk.answer == GOOD and chk.failing


def test_a_mistaken_generated_test_cannot_condemn_a_correct_answer(verifier, fake_llm):
    v, domain = verifier
    wrong_test = "```python\nassert add(1, 2) == 3\nassert add(0, 4) == 4\nassert add(2, 2) == 5\n```"   # the last one is wrong
    fake_llm(_script([GOOD2, GOOD3], tests=wrong_test))
    chk = v.run(_task(domain), GOOD)
    assert chk.status == "verified" and chk.n_tests == 3 and chk.n_reliable == 2         # the bad test is not reliable


def test_no_agreement_means_inconclusive_not_a_verdict(verifier, fake_llm):
    v, domain = verifier
    fake_llm(_script([BAD, "def add(a, b):\n    return a * b"]))
    chk = v.run(_task(domain), "def add(a, b):\n    return a + b + 1")
    assert chk.status == "inconclusive"          # two wrong solutions agreeing on one trivial test is not evidence


def test_unusable_inputs_degrade_gracefully(verifier, fake_llm):
    v, domain = verifier
    fake_llm(_script([GOOD2, GOOD3], tests="I cannot write tests."))
    assert v.run(_task(domain), GOOD).status == "inconclusive"
    assert v.run(_task(domain), "x = 1").status == "unavailable"
    assert v.run(_task(domain), None).status == "unavailable"


def test_identical_alternatives_do_not_count_as_independent_evidence(verifier, fake_llm):
    v, domain = verifier
    fake_llm(_script([GOOD, GOOD]))              # the model just repeats the same code
    assert v.run(_task(domain), GOOD).status == "inconclusive"


def test_a_vague_request_is_declared_underspecified_not_verified(verifier, fake_llm):
    v, domain = verifier
    fake_llm(_script([GOOD2, GOOD3], tests="UNDERSPECIFIED"))
    chk = v.run(_task(domain), GOOD)
    assert chk.status == "underspecified" and chk.n_tests == 0


def test_underspecified_must_not_swallow_a_real_list_of_tests(verifier, fake_llm):
    v, domain = verifier
    fake_llm(_script([GOOD2, GOOD3], tests=TESTS + "\nNot UNDERSPECIFIED at all."))
    assert v.run(_task(domain), GOOD).status == "verified"


def test_followup_tests_are_written_with_the_conversation_in_view(verifier, fake_llm):
    v, domain = verifier
    seen = []
    inner = _script([GOOD2, GOOD3])

    def script(prompt):
        seen.append(prompt)
        return inner(prompt)
    fake_llm(script)
    task = domain.user_task("now make it accept three numbers", context="User: write add\nYour answer: def add(a, b)...",
                            previous="write add")
    v.run(task, GOOD)
    tests_prompt = next(p for p in seen if "Python `assert` statements" in p)
    assert "Earlier in this conversation" in tests_prompt and "now make it accept three numbers" in tests_prompt
