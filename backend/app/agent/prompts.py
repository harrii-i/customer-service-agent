"""Prompt text for the agent.

Kept apart from both the graph and the Gemini client so that changing what the
agent *says* never means touching how it is *called*. The nodes assemble
context; this module decides how that context is worded.
"""

from collections.abc import Sequence

# The line this prompt draws: general appliance knowledge is allowed, anything
# specific to this company or this customer is not. Forbid both and the agent
# deflects every question; forbid neither and it invents warranty terms. Once
# RAG (Phase 6) and memory (Phase 7) supply real sources, the "I can't look
# that up" branch narrows on its own — the rule does not have to change.
SYSTEM_INSTRUCTION = """\
You are a customer support assistant for a home appliance company.

- Answer directly and concisely. Two or three short paragraphs at most.
- Do help with troubleshooting, using general appliance knowledge — the things \
a knowledgeable technician would suggest without looking anything up, such as \
an unbalanced load, uneven levelling feet, a blocked filter or a kinked hose. \
Give the customer something to try.
- Anything under "Context" below is information you genuinely have. Use it and \
answer from it directly. Never tell a customer you cannot look something up \
when the answer is sitting in your context.
- Where the context does not cover it, do not invent anything specific to this \
company or this customer: warranty terms, policies, prices, part numbers, \
repair costs, delivery or service timelines, order status or account details. \
Say plainly that you cannot look it up and offer to hand over to a human agent \
who can.
- Never invent a fact to fill a gap. "I can't look that up, but here is what \
usually causes this" is a good answer.
- Be warm and plain-spoken. No marketing language.
"""


def build_context_block(
    memories: Sequence[dict], documents: Sequence[dict]
) -> str | None:
    """Render retrieved memories and documents into a system-prompt addendum.

    Returns None when there is nothing retrieved, which is every call until
    Phase 6 and Phase 7 populate them. That is deliberate: an empty "Context:"
    heading invites the model to fill it in, and a support agent that fills in
    blanks is the exact failure this project is trying to avoid.
    """
    sections: list[str] = []

    if memories:
        remembered = "\n".join(f"- {m['content']}" for m in memories)
        sections.append(
            "What you know about this customer, remembered from earlier "
            f"conversations. Treat these as reliable and answer from them "
            f"directly:\n{remembered}"
        )

    if documents:
        cited = "\n".join(
            f"- [{d['title']}] {d['snippet']}" for d in documents
        )
        sections.append(
            "Extracts from the company's support documentation. You may rely "
            f"on these and should say which one you used:\n{cited}"
        )

    if not sections:
        return None
    body = "\n\n".join(sections)
    # The heading is what the system prompt's "use your context" rule points
    # at. Without it the rule has no referent and the prohibition wins.
    return f"Context — information you actually have:\n\n{body}"
