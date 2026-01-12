# Head Logit Lens Analysis Results

**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Layer**: 10
**Prompt**: `Q: Which is bigger: 9.8 or 9.11?
A:`
**Date**: 2026-01-06T23:58:28.739769

---

## Summary

| Head Type | Numerical Tokens in Top-10 |
|-----------|---------------------------|
| Even heads | 1.06 avg |
| Odd heads | 0.88 avg |

---

## Method

For each attention head:
1. Get head's output at last token position
2. Project through o_proj (that head's slice)
3. Project to vocabulary via unembedding matrix
4. Count numerical tokens (0-9, .) in top-10 promoted tokens

---

## Per-Head Results

| Head | Type | #Numerical | Top Promoted Tokens |
|------|------|------------|---------------------|
| H0 | Even | 0 | '-fontawesome', '짝', 'äh' |
| H1 | Odd | 0 | 'enis', 'icken', 'zens' |
| H2 | Even | 1 | ' Kag', ' Wheeler', 'alendar' |
| H3 | Odd | 0 | 'že', '_inline', 'ussen' |
| H4 | Even | 0 | 'Fmt', 'くれ', ' Earn' |
| H5 | Odd | 1 | 'ewan', 'อง', 'celik' |
| H6 | Even | 1 | 'onu', 'erot', 'xae' |
| H7 | Odd | 3 | 'ulur', '_Master', '报' |
| H8 | Even | 0 | 'obili', 'ombat', 'ähr' |
| H9 | Odd | 0 | 'cház', 'enville', '才' |
| H10 | Even | 0 | 'berger', 'іє', 'anki' |
| H11 | Odd | 2 | 'holm', 'ucene', 'ussia' |
| H12 | Even | 0 | 'velt', 'ifa', 'tual' |
| H13 | Odd | 2 | 'ana', ' Ana', 'ansa' |
| H14 | Even | 1 | 'qua', 'ISHED', 'andise' |
| H15 | Odd | 0 | '//{{', '示', 'china' |
| H16 | Even | 1 | 'baugh', ' Sachs', 'REA' |
| H17 | Odd | 2 | '827', '�', '/inet' |
| H18 | Even | 7 | 'viar', ' benefici', '.xhtml' |
| H19 | Odd | 1 | '.scalablytyped', 'ubat', '敗' |
| H20 | Even | 1 | 'ikan', 'quist', 'ToFit' |
| H21 | Odd | 0 | 'ORA', 'ura', 'ạc' |
| H22 | Even | 0 | 'azole', 'ieri', 'σον' |
| H23 | Odd | 0 | 'icari', ' Asset', 'asje' |
| H24 | Even | 0 | 'esson', 'andest', 'oso' |
| H25 | Odd | 0 | 'ерин', 'cci', '陶' |
| H26 | Even | 4 | 'rical', '138', ' Hope' |
| H27 | Odd | 0 | 'occ', 'oller', 'ตร' |
| H28 | Even | 0 | 'ancock', ' Pom', 'ascade' |
| H29 | Odd | 0 | 'oire', 'ær', '="{!!' |
| H30 | Even | 1 | 'ptal', 'DMI', ' kola' |
| H31 | Odd | 3 | '.apple', 'iken', ' -*-
' |

---

## One-Slide Summary

```
HEAD LOGIT LENS: What tokens do heads promote?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method: Project each head's output to vocabulary space

Numerical tokens (0-9, .) in top-10:
  Even heads: 1.06 avg
  Odd heads:  0.88 avg

Interpretation:
  [Based on results]
```
