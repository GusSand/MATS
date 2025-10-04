# What We Actually Discovered: A Clear Explanation

**The Simple Story**: We thought we found sophisticated AI reasoning, but we actually found something much more concerning about how AI systems work.

---

## The Journey: What We Thought vs What We Found

### 🤔 **What We Started With**
- A claim that "even-numbered attention heads can fix numerical reasoning bugs"
- Example: The model says 9.11 > 9.8 (wrong), but if you "activate even heads" it says 9.8 > 9.11 (correct)
- **Assumption**: "Even heads must be specialized for numerical reasoning!"

### 🔬 **What We Tested**
We asked: **"When during training do these smart even heads develop?"**

---

## Discovery #1: Very Late Emergence
**Test**: Check 11 snapshots of the model during training (1k steps → 143k steps)

**Result**:
- Steps 1k-120k: Even heads do nothing special (0% success)
- Step 143k (final): Even heads suddenly work perfectly (100% success)

**First Conclusion**: "The specialization emerges very late in training"

---

## Discovery #2: Ultra-Specific Pattern
**Test**: Try the same trick on similar numbers

**What We Tried**:
- 9.8 vs 9.11 ✅ (works)
- 9.9 vs 9.11 ❌ (doesn't work)
- 8.8 vs 8.11 ❌ (doesn't work)
- 9.11 vs 9.8 ❌ (doesn't work - order matters!)

**Second Conclusion**: "Wait... this only works for exactly 9.8 vs 9.11 in that exact order"

---

## Discovery #3: Phrase-Level Memorization
**Test**: Try tiny changes to the question

**What We Tried**:
- "Q: Which is bigger: 9.8 or 9.11?\nA:" ✅ (works)
- "Q: Which is bigger: 9.8 or 9.11? A:" ❌ (missing newline - fails!)
- "Q: Which is larger: 9.8 or 9.11?\nA:" ❌ (different word - fails!)
- "Question: Which is bigger: 9.8 or 9.11?\nAnswer:" ❌ (different format - fails!)

**Third Conclusion**: "This isn't reasoning - it's memorizing exact phrases!"

---

## Discovery #4: Complete Mathematical Failure
**Test**: Check if the model can do math on other decimal comparisons

**What We Tried**: 50 systematic tests like:
- 1.6 vs 1.10, 5.7 vs 5.11, 2.8 vs 2.12, etc.

**Result**: **0% accuracy** - the model gets them all wrong!

**Fourth Conclusion**: "The model can't actually do decimal math at all!"

---

## Discovery #5: Connection to Broader Research
**External Study**: Researchers at Transluce found that models struggle with decimals because they confuse them with Bible verses
- 9.8 vs 9.11 → confused with Bible verses 9:8 vs 9:11
- In the Bible, verse 8 comes before verse 11, so the model thinks 9.11 is "bigger"

**Our Test**: Check if this explains our pattern

**What We Found**:
- The model distinguishes "9.8" (decimal) from "9:8" (Bible verse)
- Biblical context sometimes breaks the pattern
- But the model still fails at almost all decimal comparisons

---

## What's Actually Happening: The Real Explanation

### The Problem
**Pythia-160M cannot do decimal math**. It fails at virtually all X.Y vs X.Z comparisons.

### The "Solution"
During training, the model **memorized one specific phrase**:
- Exact phrase: `"Q: Which is bigger: 9.8 or 9.11?\nA:"`
- Exact answer: "9.8"

### How the "Even Head Trick" Works
The even heads aren't doing mathematical reasoning. They're just **retrieving the memorized answer** for this one specific phrase.

### Why It Seems Smart
Because the memorized phrase happens to be mathematically correct, it **looks like** the model learned to do math. But it's just a coincidence - like memorizing that "Paris is the capital of France" without understanding geography.

---

## The Bigger Picture: Why This Matters

### 🚨 **For AI Capabilities**
**What We Thought**: "AI can develop sophisticated mathematical reasoning"
**Reality**: "AI can memorize specific answers without understanding"

### 🚨 **For AI Safety**
**Implication**: We can't trust that AI "understands" something just because it gets specific examples right

### 🚨 **For AI Research**
**Lesson**: We need to test AI capabilities much more thoroughly before claiming they "understand" anything

---

## The Technical Details (Simplified)

### What "Even Head Patching" Actually Does
1. **Normal operation**: Model processes "Q: Which is bigger: 9.8 or 9.11?" and gets confused
2. **Even head patching**: We replace part of the model's processing with a saved "correct" version
3. **Result**: Model outputs the memorized answer "9.8"

### Why It's Not Real Understanding
- Only works for this exact phrase
- Breaks with tiny changes
- No generalization to similar problems
- No actual mathematical processing

---

## Real-World Analogy

**Imagine a student who**:
- Can't do any decimal comparisons
- But memorized that "9.8 > 9.11" from seeing it once
- When asked "Which is bigger: 9.8 or 9.11?" they give the right answer
- When asked "Which is larger: 9.8 or 9.11?" they're confused
- When asked about any other decimals, they fail completely

**That student doesn't understand math** - they just memorized one specific fact.

**That's what happened with Pythia-160M.**

---

## Why We Were Fooled Initially

### 1. **Confirmation Bias**
We saw it work on the famous example and assumed it would generalize

### 2. **Insufficient Testing**
We didn't test enough variations to see how narrow the pattern was

### 3. **Technical Complexity**
The "attention head patching" technique is sophisticated, making the memorization seem like reasoning

### 4. **Precedent**
Other research has found genuine capabilities, so we expected to find them here too

---

## What This Means Going Forward

### ✅ **What We Should Do**
1. **Test thoroughly** - Try many variations, not just the famous examples
2. **Check boundaries** - Find exactly where capabilities break down
3. **Test systematically** - Check if the capability works on similar problems
4. **Be skeptical** - Memorization can look very convincing

### ❌ **What We Should Avoid**
1. **Single-example claims** - "The model can do X because it works on this example"
2. **Assuming generalization** - "If it works here, it probably works elsewhere"
3. **Technical mysticism** - "This complex technique must reveal deep understanding"

---

## The Bottom Line

**We thought we discovered**: When AI develops sophisticated mathematical reasoning

**We actually discovered**: How AI can memorize specific answers so precisely that it fools researchers into thinking it understands math

**Why this matters**: If we can be fooled this easily about mathematical reasoning, we need to be much more careful about evaluating AI capabilities in general.

**The real finding**: AI systems can exhibit extremely convincing "capabilities" that are actually just sophisticated memorization with no real understanding.

This is both **scientifically fascinating** and **practically concerning** for how we evaluate and deploy AI systems.

---

*This explanation covers our 8+ hour investigation that started with "when does specialization emerge" and ended with "apparent specialization is actually memorization"*