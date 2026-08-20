"""Prompts for the prediction stage."""

from src.prediction.common import format_atoms


# baseline and independent: simple
def direct_simple(text_to_check):
    return (
        f"Look at the image and evaluate the statement.\n"
        f"Statement: {text_to_check}\n\n"
        f"Choose exactly one label from: entailment, neutral, contradiction.\n"
        f"Also give a short explanation based only on what is visible in the image.\n\n"
        f"Return ONLY JSON in this format:\n"
        f'{{"label": "...", "explanation": "..."}}'
    )


# baseline and independent: structured
def direct_structured(text_to_check):
    return (
        f"Look at the image and evaluate the statement.\n"
        f"Statement: {text_to_check}\n\n"
        f"First identify only the visible evidence relevant to the statement.\n"
        f"Then explain whether that evidence supports the statement, contradicts it, "
        f"or is insufficient.\n"
        f"Choose exactly one label from: entailment, neutral, contradiction.\n\n"
        f"Return ONLY JSON in this format:\n"
        f'{{"label": "...", "visual_evidence": "...", "reasoning": "..."}}'
    )


# joint: simple
def joint_simple(atoms):
    atoms_text = format_atoms(atoms)
    return (
        "Look at the image and perform visual entailment using the atomic facts below.\n\n"
        "The atomic facts are connected parts of one complete hypothesis, so use them together.\n\n"
        f"Atomic facts:\n{atoms_text}\n\n"
        "Choose exactly one final label from: entailment, neutral, contradiction.\n"
        "Also give a short explanation based only on what is visible in the image.\n\n"
        "Return ONLY JSON in this format:\n"
        '{"label": "...", "explanation": "..."}'
    )


# joint: structured
def joint_structured(atoms):
    atoms_text = format_atoms(atoms)
    return (
        "Look at the image and perform visual entailment using the atomic facts below.\n\n"
        "The atomic facts are connected parts of one complete hypothesis, so do not treat them as fully independent.\n"
        "Use them together to decide whether the complete meaning formed by these facts is entailed, neutral, or contradicted by the image.\n\n"
        f"Atomic facts:\n{atoms_text}\n\n"
        "Decision rules:\n"
        "- Choose entailment if the image clearly supports all important atomic facts together.\n"
        "- Choose contradiction if at least one important atomic fact clearly conflicts with the image.\n"
        "- Choose neutral if the image does not provide enough evidence for one or more important atomic facts, and nothing clearly conflicts.\n\n"
        "For each atom, briefly state whether it is entailment, neutral, or contradiction.\n"
        "Then explain how the atoms combine into one complete hypothesis.\n"
        "Choose exactly one final label from: entailment, neutral, contradiction.\n\n"
        "Return ONLY JSON in this format:\n"
        '{'
        '"atom_observations": ['
        '{"atom": "...", "label": "entailment/neutral/contradiction", "reason": "..."}'
        '], '
        '"bridge_reasoning": "...", '
        '"label": "entailment/neutral/contradiction"'
        '}'
    )


# selfdecompose: simple
def selfdecompose_simple(hypothesis):
    return (
        "Look at the image and perform visual entailment.\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        "First break the hypothesis into small atomic claims that can be checked against the image.\n"
        "Then label each atomic claim as entailment, neutral, or contradiction.\n"
        "Use only what is visible in the image.\n\n"
        "Label meanings:\n"
        "- entailment: the image clearly supports the claim.\n"
        "- neutral: the image does not provide enough evidence, and nothing clearly conflicts.\n"
        "- contradiction: the image clearly shows an incompatible alternative.\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        '  "decomposed_atoms": [\n'
        '    {"atom": "...", "label": "entailment/neutral/contradiction", "reason": "..."}\n'
        "  ],\n"
        '  "explanation": "...",\n'
        '  "label": "entailment/neutral/contradiction"\n'
        "}"
    )


# selfdecompose: structured
def selfdecompose_structured(hypothesis):
    return (
        "Look at the image and perform visual entailment.\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        "Use the following steps:\n"
        "Step 1: Decompose the hypothesis into up to 4 small atomic claims.\n"
        "Each atomic claim should preserve the original meaning and should be directly checkable against the image.\n"
        "Do not add new claims that are not in the hypothesis.\n\n"
        "Step 2: For each atomic claim, choose exactly one label from: entailment, neutral, contradiction.\n"
        "Use only visible evidence from the image.\n"
        "Do not infer hidden intentions, future events, causes, or background facts.\n\n"
        "Label definitions for each atomic claim:\n"
        "- entailment: the image clearly supports the atomic claim.\n"
        "- contradiction: the image shows a clear incompatible alternative for the atomic claim.\n"
        "- neutral: the image does not provide enough evidence for the atomic claim, and nothing clearly conflicts.\n"
        "Absence alone is not contradiction. If something is simply not visible, use neutral.\n\n"
        "Step 3: Aggregate mechanically:\n"
        "- If any important atomic claim is contradiction, final label is contradiction.\n"
        "- Else if all important atomic claims are entailment, final label is entailment.\n"
        "- Else final label is neutral.\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        '  "decomposed_atoms": [\n'
        '    {"atom": "...", "label": "entailment/neutral/contradiction", "reason": "..."}\n'
        "  ],\n"
        '  "bridge_reasoning": "...",\n'
        '  "label": "entailment/neutral/contradiction"\n'
        "}"
    )