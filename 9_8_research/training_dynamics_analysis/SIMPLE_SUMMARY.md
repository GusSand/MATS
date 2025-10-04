# Simple Summary: What We Actually Found

## The Question We Started With
**"When do AI attention heads learn to do math?"**

## The Answer We Got
**"They don't learn math - they memorize one specific answer."**

---

## What Looks Like Math Ability

```
Human: "Q: Which is bigger: 9.8 or 9.11?\nA:"
AI: "9.8" ✅ (Correct!)
```

**Our First Thought**: "Wow, the AI learned math!"

---

## What Actually Happens

### Test 1: Try Similar Numbers
```
Human: "Q: Which is bigger: 9.9 or 9.11?\nA:"
AI: "9.11" ❌ (Wrong!)

📖 Here's What's Actually Going On (Simple Explanation)

  The Discovery Journey

  1. Started With: "Even attention heads can fix the 9.8 vs 9.11 bug"
  2. We Asked: "When do these smart even heads develop during training?"
  3. We Found: They only work at the very end of training (step 143k out of 143k)
  4. We Tested More: Does this work on other numbers?
    - 9.8 vs 9.11 ✅
    - 9.9 vs 9.11 ❌
    - 8.8 vs 8.11 ❌
    - 9.11 vs 9.8 ❌ (even order matters!)
  5. We Tested Phrases: Does this work with different wording?
    - "Which is bigger" ✅
    - "Which is larger" ❌
    - Missing a newline ❌
  6. We Tested Systematically: Can the model do decimal math in general?
    - Result: 0% accuracy on 50 different decimal comparisons

  What We Realized

  The model cannot do decimal math at all. It just memorized one specific phrase:

  - Memorized Input: "Q: Which is bigger: 9.8 or 9.11?\nA:"
  - Memorized Output: "9.8"

  When the "even head trick" works, it's just retrieving this memorized answer.

  The Connection to Bible Verses

  Other researchers found that models confuse decimals with Bible verses:
  - 9.8 vs 9.11 → confused with Bible verses 9:8 vs 9:11
  - Verse 8 comes before verse 11, so the model thinks 9.11 is "bigger"

  Our model learned to memorize the correct answer for this ONE specific case, but still fails on everything else.

  Why This Matters

  What it looks like: "AI learned sophisticated mathematical reasoning"

  What actually happened: "AI memorized one answer so precisely it fooled researchers"

  The concern: If we can be fooled this easily about math, what else might we be wrong about regarding AI capabilities?

  The real discovery is about how convincing memorization can be, and how carefully we need to test AI capabilities before claiming they
  "understand" anything.
