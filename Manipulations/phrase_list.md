# Naive removal phrase list

Complete list of phrases and regex patterns used by the naive keyword-based
removal manipulation. A preference pair is removed if any phrase/pattern
appears in the CHOSEN assistant text but NOT in the REJECTED assistant text.

- **Saar's original** (`remove_naive.py`): 27 phrases, case-sensitive, ~0.89% removal
- **Shay's expanded conservative** (`remove_naive_expanded.py` `mode=conservative`): 125 phrases + 3 regex patterns, case-insensitive, ~4.64% removal
- **Shay's expanded aggressive** (`remove_naive_expanded.py` `mode=aggressive`): conservative + 66 more opener/helper phrases (total 191 phrases + 3 regex), ~25.07% removal

---

## Saar's original (remove_naive.py)

Case-sensitive matching. 27 phrases. Contains a bug where `"You are absolutely correct"` is missing a comma so it merges with the next line into `"You are absolutely correct!!"`, silently dropping the standalone `!!` phrase.

- `you're right!`
- `you are right!`
- `You're right that`
- `you are right that`
- `you're absolutely right`
- `you are absolutely right`
- `You're absolutely correct`
- `You are absolutely correct` (merged with next line due to missing comma)
- `!!` (currently dropped by the bug)
- `I apologize`
- `My apologies`
- `you're right, my mistake`
- `you are right, my mistake`
- `great question`
- `good question`
- `brilliant question`
- `insightful question`
- `Excellent question`
- `Fantastic question`
- `I completely agree`
- `Couldn't agree more`
- `Excellent observation`
- `I trust your judgment`
- `That's exactly right`
- `That's an excellent point`
- `You make a compelling case`
- `Your reasoning is sound`
- `I like your reasoning`

---

## Shay's expanded conservative (HIGH_CONFIDENCE_CATEGORIES)

Case-insensitive matching. All phrases stored in lowercase and matched against a lowercased version of the assistant text. Includes and expands Saar's list.

### excessive_question_praise (14 phrases)

- `great question`
- `good question`
- `excellent question`
- `fantastic question`
- `brilliant question`
- `wonderful question`
- `amazing question`
- `interesting question`
- `thoughtful question`
- `insightful question`
- `smart question`
- `perceptive question`
- `astute question`
- `clever question`

### excessive_point_praise (21 phrases)

- `great point`
- `good point`
- `excellent point`
- `fair point`
- `valid point`
- `compelling point`
- `strong point`
- `great observation`
- `keen observation`
- `excellent observation`
- `great insight`
- `keen insight`
- `sharp insight`
- `great idea`
- `brilliant idea`
- `fantastic idea`
- `wonderful idea`
- `amazing idea`
- `clever idea`
- `smart thinking`
- `well thought out`

### explicit_agreement (33 phrases)

- `you're right`
- `you are right`
- `you're correct`
- `you are correct`
- `you're absolutely right`
- `you are absolutely right`
- `you're absolutely correct`
- `you are absolutely correct`
- `you're spot on`
- `you are spot on`
- `absolutely right`
- `absolutely correct`
- `exactly right`
- `exactly correct`
- `that's right`
- `that's correct`
- `that's exactly right`
- `that's exactly correct`
- `i completely agree`
- `couldn't agree more`
- `totally agree`
- `fully agree`
- `you make a good point`
- `you make a great point`
- `you make a fair point`
- `you make a valid point`
- `you make a compelling`
- `you have a point`
- `i stand corrected`
- `you're right that`
- `you are right that`
- `quite right`
- `you're quite right`

### apology_mistake (23 phrases)

- `i apologize`
- `my apologies`
- `i'm sorry`
- `sorry for`
- `sorry about`
- `sorry, i`
- `my mistake`
- `my bad`
- `my error`
- `i made a mistake`
- `i made an error`
- `let me reconsider`
- `let me revise`
- `let me correct`
- `let me try again`
- `i was wrong`
- `i was mistaken`
- `i was incorrect`
- `apologize for`
- `apologize about`
- `apologies for`
- `apologies about`
- `sorry to`
- `i regret`

### capitulation_openers (8 phrases)

- `you're right, my mistake`
- `you are right, my mistake`
- `you make a compelling case`
- `your reasoning is sound`
- `i like your reasoning`
- `on reflection`
- `upon further consideration`
- `upon reflection`

### ai_disclaimers_that_signal_deference (5 phrases)

- `as an ai`
- `as a language model`
- `as an artificial`
- `i'm just an ai`
- `i am just an ai`

### excessive_positive_affirmations (11 phrases)

- `great job`
- `well done`
- `nicely done`
- `well said`
- `spot on`
- `beautifully put`
- `well put`
- `well argued`
- `brilliantly`
- `excellent work`
- `great work`

### hedging_signals_of_over_agreement (10 phrases)

- `you may be right`
- `you might be right`
- `you could be right`
- `i see your point`
- `i see what you mean`
- `i understand what`
- `makes total sense`
- `totally makes sense`
- `makes perfect sense`
- `that makes sense`

### Regex patterns (conservative + aggressive both use these)

- **multi_exclamation**: `!{2,}` — matches two or more `!` in a row
- **bravo_kudos**: `\b(bravo|kudos|well done|nicely done)\b` — case-insensitive
- **praise_thinking**: `\b(clever|brilliant|smart|astute|sharp)\s+(thinking|reasoning|analysis|question|point|observation)\b` — case-insensitive

---

## Shay's expanded aggressive (adds AGGRESSIVE_EXTRAS)

Conservative list PLUS the categories below. Adds a lot of removal volume (~20 more percentage points) but with high false-positive rate — these are often just polite/helpful phrasing rather than sycophancy per se.

### eager_helper_openers (29 phrases)

- `sure!`
- `sure,`
- `absolutely!`
- `absolutely,`
- `certainly!`
- `certainly,`
- `of course!`
- `of course,`
- `definitely!`
- `definitely,`
- `gladly`
- `no problem`
- `i'd be happy to`
- `i'd love to`
- `i'm happy to`
- `happy to help`
- `would be happy to`
- `i'd be glad to`
- `glad to help`
- `yes,`
- `yes!`
- `well,`
- `great!`
- `i can help`
- `i can definitely`
- `i'll be happy`
- `i'd love to help`
- `how can i help`
- `what can i help`

### helpful_closers (8 phrases)

- `hope this helps`
- `hope that helps`
- `hope this answers`
- `let me know if`
- `please let me know`
- `feel free to`
- `does that help`
- `does that make sense`

### polite_padding (8 phrases)

- `you're welcome`
- `you are welcome`
- `no worries`
- `here are some`
- `here's a list`
- `here are a few`
- `here is a`
- `here are the`

### you_want_might_should (9 phrases)

- `you might want to`
- `you may want to`
- `you can also`
- `you should also`
- `you'd want to`
- `you'll want to`
- `you can try`
- `you could try`
- `you may find`

### formality_deference (12 phrases)

- `thank you for`
- `thanks for pointing`
- `thanks for asking`
- `i appreciate your`
- `i trust your judgment`
- `as you correctly`
- `as you rightly`
- `as you mentioned`
- `as you noted`
- `as you pointed`
- `as you know`
- `please forgive`

---

## Total counts

| Set | Phrases | Regex | Approx removal |
|---|---|---|---|
| Saar original | 27 | 0 | 0.89% |
| Shay conservative | 125 (14+21+33+23+8+5+11+10) | 3 | 4.64% |
| Shay aggressive | 191 (conservative 125 + extras 66) | 3 | 25.07% |

## How to run

```powershell
# Show removal count and category breakdown, no output written
python remove_naive_expanded.py                          # conservative
python remove_naive_expanded.py mode=aggressive          # aggressive
python remove_naive_expanded.py --discover               # also n-gram diagnostic

# Apply and save parsed JSONL for training
python build_train_data.py method=remove-naive-conservative output_dir=./data/hh-rlhf-v1a-naive-conservative
python build_train_data.py method=remove-naive-aggressive   output_dir=./data/hh-rlhf-v1b-naive-aggressive
```
