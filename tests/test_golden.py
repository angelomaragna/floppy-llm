import pytest

from tests import golden  # noqa: E402


@pytest.mark.parametrize("prompt,fact", golden.prompts_and_facts(), ids=[0, 1, 2])
def test_argmax_streams_match(prompt, fact):
    ref = golden.reference_argmax(prompt)
    got = golden.c_host_argmax(prompt)
    assert len(ref) == len(got), f"stream lengths differ: {len(ref)} vs {len(got)}"
    for i, (r, g) in enumerate(zip(ref, got)):
        if r != g:
            pytest.fail(f"argmax diverges at position {i}: numpy={r}, c={g}")


@pytest.mark.parametrize("prompt,fact", golden.prompts_and_facts(), ids=[0, 1, 2])
def test_completion_recalls_fact(prompt, fact):
    out = golden.c_host_completion(prompt)
    # completion contains the original fact (prefix may be re-emitted)
    assert fact[:60] in out, f"fact not recalled; got: {out!r}"
