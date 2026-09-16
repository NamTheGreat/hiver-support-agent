# Failure Analysis: Top 5 Real Production Edge Cases

This report analyzes the top 5 operational failure modes observed during the golden evaluation ($N=180$) of the Hiver Customer Support Agent on historical AppleSupport inquiries. Every case represents a real tweet from the evaluation set.

---

### Failure Mode 1: Sentiment False Positives on Resolvable Technical Inquiries

- **Example**: Tweet ID `2497134` (Golden Set ID `21`)
- **Customer Query**:
  > *"needs to stop making phones with shitty battery life. I'm charging my phone, and the % drops, close to dying. What a waste of money."*
- **Ground Truth**: Intent: `battery_charging` | Escalate: `no` (Standard self-serve battery calibration)
- **Agent Prediction**: Intent: `battery_charging` | Routing: `escalate` (`NEGATIVE_SENTIMENT`)
- **Agent Draft Reply**:
  > *"We'd like to help you resolve this. Could you let us know what device model and iOS version you are currently running?"*
- **Judge Evaluation**: Correctness: 4/5 | Groundedness: 5/5 | Tone: 5/5 | Actionability: 4/5 | Composite: 4.5/5
- **Failure Analysis & Root Cause**:
  The customer was frustrated by an abnormal battery drain, leading VADER compound sentiment to register $-0.71$ (well below our $-0.50$ threshold due to words like *"shitty"*, *"dying"*, *"waste"*). However, the underlying issue was a routine, resolvable technical problem. The router prioritized coarse sentiment over technical resolvability, unnecessarily routing a high-volume query to human tier-2 support.
- **Remediation**: Decouple profanity/frustration from escalation when the intent is allowlisted and confidence is $>0.90$. Escalate on sentiment only when negative emotion is directed specifically at brand conduct or legal threats, rather than general device dissatisfaction.

---

### Failure Mode 2: Contextual Blindness on DM Confirmations

- **Example**: Tweet ID `2057289` (Golden Set ID `163`)
- **Customer Query**:
  > *"DMed you, 😊"*
- **Ground Truth**: Intent: `other` | Escalate: `yes`
- **Agent Prediction**: Intent: `other` | Routing: `escalate` (`LOW_CONFIDENCE`)
- **Agent Draft Reply**:
  > *"We'd like to help you resolve this. Could you let us know what device model and iOS version you are currently running?"*
- **Judge Evaluation**: Correctness: 3/5 | Groundedness: 5/5 | Tone: 5/5 | Actionability: 3/5 | Composite: 4.0/5
- **Failure Analysis & Root Cause**:
  This message represents an out-of-channel synchronization turn: the customer has already initiated a private Direct Message to share private account details. Because our architecture purposefully operates on single-turn public context, the agent treats the message as a technical failure and asks for an iOS version. This creates customer friction.
- **Remediation**: Introduce an explicit meta-intent or regex pattern for DM confirmations (`\b(dmed|sent a dm|check dm)\b`) that replies with an acknowledgment: *"Thanks for following up! We've received your DM and will assist you there shortly."*

---

### Failure Mode 3: Language Filter Evasion on Ultra-Terse Non-English Queries

- **Example**: Tweet ID `1746920` (Golden Set ID `173`)
- **Customer Query**:
  > *"أخيرا. 😍😍"* *(Arabic: "Finally. 😍😍")*
- **Ground Truth**: Intent: `other` | Escalate: `yes`
- **Agent Prediction**: Intent: `other` | Routing: `escalate` (`LOW_CONFIDENCE`)
- **Agent Draft Reply**:
  > *"We'd like to help you resolve this. Could you let us know what device model and iOS version you are currently running?"*
- **Judge Evaluation**: Correctness: 3/5 | Groundedness: 5/5 | Tone: 5/5 | Actionability: 3/5 | Composite: 4.0/5
- **Failure Analysis & Root Cause**:
  Our data preparation pipeline filters non-English threads using `langdetect.detect_langs(..., min_confidence=0.90)`. However, n-gram statistical language detectors degrade significantly on ultra-short strings (1 word + emojis), occasionally failing to detect non-Latin scripts. The agent subsequently replied in English to an Arabic speaker.
- **Remediation**: Supplement statistical language detection with a deterministic Unicode script check (`regex [^\x00-\x7F]`) to immediately reject or route non-Latin alphabets to multilingual queues.

---

### Failure Mode 4: Vague Venting Queries Leading to Speculative Diagnostics

- **Example**: Tweet ID `1676982` (Golden Set ID `165`)
- **Customer Query**:
  > *"Done with this shit"*
- **Ground Truth**: Intent: `other` | Escalate: `yes`
- **Agent Prediction**: Intent: `other` | Routing: `escalate` (`LOW_CONFIDENCE`)
- **Agent Draft Reply**:
  > *"We'd like to help you resolve this. Could you let us know what device model and iOS version you are currently running?"*
- **Judge Evaluation**: Correctness: 3/5 | Groundedness: 5/5 | Tone: 5/5 | Actionability: 3/5 | Composite: 4.0/5
- **Failure Analysis & Root Cause**:
  The message contains pure emotional venting with zero technical context (no device mentioned, no application named). While the agent maintained polite tone and correctly escalated, its reply presumed a technical hardware/OS diagnostic flow. The LLM judge docked actionability because asking for an iOS version is a non-sequitur when the customer hasn't stated whether they are upset about a phone, billing charge, or store experience.
- **Remediation**: For ambiguous queries shorter than 25 characters without technical nouns, use an empathetic open-ended clarification template: *"We're sorry to hear you're having trouble. Could you tell us more about what happened so we can assist?"*

---

### Failure Mode 5: Known Widespread Glitches Lacking Fast-Path Hotfix Grounding

- **Example**: Tweet ID `1688547` (Golden Set ID `179`)
- **Customer Query**:
  > *"why are half my emoji's popping up as \"?\""*
- **Ground Truth**: Intent: `system_glitch_bug` | Escalate: `yes`
- **Agent Prediction**: Intent: `system_glitch_bug` | Routing: `escalate` (`LOW_CONFIDENCE`)
- **Agent Draft Reply**:
  > *"We'd like to help you resolve this. Could you let us know what device model and iOS version you are currently running?"*
- **Judge Evaluation**: Correctness: 4/5 | Groundedness: 5/5 | Tone: 5/5 | Actionability: 4/5 | Composite: 4.5/5
- **Failure Analysis & Root Cause**:
  In November 2017, Apple faced a high-profile iOS 11.1 bug where autocorrect substituted letters and emojis with a box containing a question mark (`[?]`). Apple's official support response at the time was a specific temporary workaround: go to `Settings > General > Keyboard > Text Replacement` and set a rule for "I". Because our FAISS retrieval searches top-3 cosine similarity over general support threads, it failed to prioritize this specific hotfix over generic iOS diagnostic replies.
- **Remediation**: Implement a dynamic "Known Outages & High-Impact Glitches" bulletin board in the retrieval layer. When a recurring query pattern spikes in volume, the retrieval system should prepend the specific pinned workaround before falling back to general similarity search.
