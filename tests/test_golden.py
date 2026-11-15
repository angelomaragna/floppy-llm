import pytest

from tests import golden  # noqa: E402


@pytest.mark.parametrize("prompt,fact", golden.prompts_and_facts(), ids=[0, 1, 2])
def test_argmax_streams_match(prompt, fact):
    got = golden.c_host_argmax(prompt)
    # the C host may auto-select a casing variant of the prompt; its stream
    # must equal the reference on at least one of those variants
    allowed = [golden.reference_argmax(v) for v in golden.casing_variants(prompt)]
    match = False
    best_fail = None
    for ref in allowed:
        if ref == got:
            match = True
            break
        for i, (r, g) in enumerate(zip(ref, got)):
            if r != g:
                best_fail = f"argmax diverges at position {i}: numpy={r}, c={g}"
                break
    if not match:
        pytest.fail(best_fail or
                    f"stream lengths differ: {len(allowed[0])} vs {len(got)}")


@pytest.mark.parametrize("prompt,fact", golden.prompts_and_facts(), ids=[0, 1, 2])
def test_completion_recalls_fact(prompt, fact):
    out = golden.c_host_completion(prompt)
    # completion contains the original fact (prefix may be re-emitted)
    assert fact[:60] in out, f"fact not recalled; got: {out!r}"
