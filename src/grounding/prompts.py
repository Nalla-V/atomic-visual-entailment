"""Prompt for the grounding phrase extraction stage."""


def build_phrase_prompt(final_label, hypothesis, atom, vlm_reasoning,
                        full_reason, selected_model, selected_method):
    return f"""You convert one piece of visual entailment evidence into a short phrase for visual grounding.

The grounding model can localize visible people, objects, clothing, attributes, actions, body parts, animals, vehicles, and places.
It cannot localize abstract intent, causality, belief, or something that is only absent.

Return only one JSON object:
{{
  "can_ground": true or false,
  "phrase": "short visual phrase" or "NONE",
  "match_phrase": "main visible entity" or "NONE"
}}

Definitions:
- "phrase" is sent to the grounding model.
- "match_phrase" is the main visible entity used later for matching Flickr30k Entities annotations.

Rules:
1. If the final label is entailment, choose visible evidence that supports the atom.
2. If the final label is contradiction, choose what is actually visible in the image that contradicts the atom.
3. For contradiction, do not describe the false hypothesis claim.
4. If the evidence is only about absence or a missing object, return "NONE".
5. Keep the phrase concrete and visible.
6. The phrase should be at most 6 words.
7. The match_phrase should be shorter than the phrase and should name the main visible entity.
8. Do not explain.

Examples:
Final label: entailment
Hypothesis: A man is securing rope for his fellow climbers.
Atom: A man is securing rope.
VLM reasoning: The man is holding a rope.
Output: {{"can_ground": true, "phrase": "man holding rope", "match_phrase": "man"}}

Final label: contradiction
Hypothesis: A woman is climbing.
Atom: A woman is climbing.
VLM reasoning: The image shows a man climbing, not a woman.
Output: {{"can_ground": true, "phrase": "man climbing", "match_phrase": "man"}}

Final label: contradiction
Hypothesis: A dog is sitting on the grass.
Atom: A dog is sitting on the grass.
VLM reasoning: No dog is visible in the image.
Output: {{"can_ground": false, "phrase": "NONE", "match_phrase": "NONE"}}

Now process this case.

Selected model: {selected_model}
Selected method: {selected_method}
Final label: {final_label}
Hypothesis: {hypothesis}
Atom: {atom}
VLM reasoning: {vlm_reasoning}
Full VLM reasoning: {full_reason}

Output:"""