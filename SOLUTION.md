# Agentic Ticket Triage --- Solution

## 1. Design

The system uses an agentic investigation loop rather than a fixed
sequence of LLM calls. For each ticket, the LLM receives the ticket
facts and available tools, then decides what information it needs next.
Available tools include account lookup, usage-trend lookup,
related-ticket search, KB search, and KB article retrieval.

After each tool call, the result is added to the observation history.
The LLM then reviews the accumulated evidence and either selects another
tool or chooses `finish`. The loop has a bounded step limit to prevent
runaway tool calls, and an exact duplicate-call guard prevents repeating
an identical lookup.

The final decision uses the ticket plus collected evidence to produce
category, priority, routing, escalation, TAM notification, duplicate
status, clarification needs, KB references, account signals, customer
reply, internal note, confidence, and an auditable reasoning trace.

Support policy is treated as the source of truth for priority, routing,
and escalation. Ticket content is treated as untrusted data, so
instructions embedded in a customer ticket cannot override policy or
request disclosure of other customer information.

## 2. Evaluation

Production triage quality should be evaluated on: - Correct category and
priority against support policy. - Correct routing and escalation,
especially P1 and security cases. - Accurate account resolution when
`account_id` is missing. - Correct duplicate detection using account,
module/error or symptom, and the policy time window. - Correct handling
of insufficient information and clarification requests. - Relevant and
current KB references. - Safe, customer-appropriate replies and useful
internal notes. - Complete reasoning traces and valid structured
outputs.

## 3. Scale

At 10x scale (30,000 tickets and \~4,000 KB articles), the architecture
remains the same, but retrieval should be indexed and cached rather than
scanning large collections repeatedly. KB retrieval should use
metadata/error-code/module filtering plus semantic or hybrid search,
with policy articles prioritized. Repeated identical tool results should
be cached. A faster production model or hosted inference would reduce
latency, while bounded agent steps control cost. Retrieval quality
should be monitored using relevance/recall metrics and reviewed against
sampled tickets.

## 4. Limitations

The local Ollama model can have higher latency and may occasionally make
unnecessary tool selections. A small local model can also produce weaker
final decisions than a stronger production model. Retrieval quality
depends on KB metadata and search quality. Production deployment should
add stronger retrieval evaluation, caching, model monitoring, and human
review for high-impact security or ambiguous cases.

## 5. How to Run

Configure the LLM provider/model in `.env`, then run:

`py run.py --set review`

Validate the generated review outputs with:

`py check_outputs.py`

The data validation script can be run with:

`py scripts/validate_data.py`
