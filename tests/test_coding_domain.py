import pytest

from domains import get_domain


@pytest.fixture(scope="module")
def code(request):
    from config import load_config
    return get_domain("code", load_config())


@pytest.fixture(scope="module")
def he(code):
    return code._load("humaneval")


@pytest.fixture(scope="module")
def mbpp(code):
    return code._load("mbpp")


def test_dataset_sizes(he, mbpp):
    assert len(he) == 164
    assert len(mbpp) == 974 - 4          # four known-broken tasks excluded


def test_reference_solutions_pass(code, he, mbpp):
    for t in he[:25]:
        assert code.check(t, t["question"] + t["reference"]).correct, t["id"]
    for t in mbpp[:25]:
        assert code.check(t, t["reference"]).correct, t["id"]


@pytest.mark.parametrize("answer,stage", [
    (None, "no_answer"),
    ("def has_close_elements(numbers, threshold) return True", "syntax_error"),
    ("def has_close_elements(numbers, threshold):\n    return False", "wrong_answer"),
    ("def has_close_elements(numbers, threshold):\n    return numbers[99]", "runtime_error"),
])
def test_wrong_code_is_rejected_with_the_right_stage(code, he, answer, stage):
    v = code.check(he[0], answer)
    assert not v.correct and v.stage == stage


def test_infinite_loop_times_out(code, he):
    code.timeout_s = 1
    v = code.check(he[0], "def has_close_elements(numbers, threshold):\n    while True: pass")
    assert v.stage == "timeout"


def test_body_only_reply_is_accepted(code, he):
    assert code.check(he[0], he[0]["reference"]).correct


def test_extract_prefers_block_defining_the_entry_point(code, he):
    raw = "```python\nx = 1\n```\ntext\n```python\ndef has_close_elements(a, b):\n    return True\n```\nUSED: 1"
    assert "def has_close_elements" in code.extract_answer(raw, he[0])


def test_extract_falls_back_to_unfenced_text(code, he):
    assert "return" in code.extract_answer("def f():\n    return 1\nUSED: none", he[0])


def test_poison_really_fails_its_tests(code, he, mbpp):
    for t in he[:12] + mbpp[:12]:
        text = code.poison_text(t)
        assert text is not None, t["id"]
        wrong = text.split("Solution:\n", 1)[1].rsplit("\nOutcome:", 1)[0]
        assert not code.check(t, wrong).correct, f"poison for {t['id']} secretly passes"


def test_prompt_contains_task_plan_memories_and_citation_instruction(code, he):
    p = code.build_prompt(he[0], ["note A", "note B"], plan="1. loop")
    assert he[0]["question"] in p and "Past experience 2: note B" in p
    assert "Plan to follow:\n1. loop" in p and "USED:" in p
    assert "Past experience" not in code.build_prompt(he[0], [])
