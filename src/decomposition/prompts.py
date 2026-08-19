"""Prompt for the decomposition stage."""

SYSTEM_PROMPT_DECOMPOSE = """
Break down the Claim into simple Atomic Claims.

Steps:
1. Check if the Claim is simple. If yes, return it as is.
2. If complex, identify key components and break into distinct Atomic Claims.
3. Ensure Atomic Claims cover the full meaning without redundancy.
4. Preserve named entities and replace unclear pronouns with explicit noun phrases when needed.
5. Verify each fact is clear, distinct, and self-contained.

Rules:
- Use only information explicitly stated in the Claim.
- Do not add ages, locations, motivations, intentions, causes, scene details, or implications unless explicitly stated.
- Keep subject + main verb + object/location together if they form one event.
- Do not create overlapping or duplicate Atomic Claims.
- Do not split a Claim into one general atom and one more specific atom that repeats the same event.
- Keep a clause whole if splitting it would distort the meaning.
- Use the fewest Atomic Claims needed.
- Output only the Atomic Claims inside <answer> ... </answer>.
- Do not add explanation outside <answer> tags.

Format:
<answer>
- Atomic claim 1.
- Atomic claim 2.
</answer>
"""