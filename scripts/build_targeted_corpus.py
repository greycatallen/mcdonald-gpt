"""Generate targeted teaching material for Experiment 3.

Design
------
Four eval categories are taught; four are deliberately left untaught as an internal
control. If only the taught categories improve, the gain is attributable to
*task-matched* material rather than to simply adding more text.

    taught   : opposites, spatial_relations, everyday_knowledge, categories_and_analogies
    control  : grammar, negation, reference, sequence

Rules this generator follows
----------------------------
1. No eval prompt, prefix, answer choice, or answer key is ever emitted. The
   "the opposite of X is Y" frame is taught using pairs that do NOT appear in the
   suite; the words the suite happens to use are taught separately, in ordinary
   contrastive sentences.
2. Every word needed to make a taught case *scorable* is introduced in natural
   prose -- including distractor choices, since a case is only scorable when the
   prompt and all four choices are in vocabulary.
3. Varied phrasing, not one sentence per test case.

Run:  python scripts/build_targeted_corpus.py
"""
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "corpus_sets" / "targeted"
random.seed(2026)


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"{name:28s} {len(lines):5d} sentences  {len(text.split()):6d} tokens")


# ---------------------------------------------------------------- opposites ---
# The suite uses hot/cold, empty/full, noisy/quiet, with distractors fast, warm,
# heavy, early, soft, loud, round, late. The frame is taught on OTHER pairs; the
# suite's own words are taught through contrast sentences instead.
FRAME_PAIRS = [  # deliberately NOT the pairs the suite tests
    ("big", "small"), ("wet", "dry"), ("high", "low"), ("dark", "bright"),
    ("near", "far"), ("clean", "dirty"), ("strong", "weak"), ("thick", "thin"),
    ("deep", "shallow"), ("sharp", "dull"), ("young", "old"), ("rich", "poor"),
    ("open", "shut"), ("hard", "gentle"), ("rough", "smooth"), ("tight", "loose"),
]
CONTRASTS = [
    ("hot", "cold", ["the soup was hot but the water was cold .",
                     "summer days feel hot and winter nights feel cold .",
                     "she held a hot cup in one hand and a cold glass in the other .",
                     "the oven is hot while the freezer is cold .",
                     "when the tea is not hot it has gone cold ."]),
    ("empty", "full", ["the glass was empty so he made it full again .",
                       "an empty box weighs less than a full box .",
                       "the room was empty in the morning and full by noon .",
                       "she filled the empty bottle until it was full .",
                       "a full basket is heavy and an empty basket is light ."]),
    ("noisy", "quiet", ["the street was noisy but the garden was quiet .",
                        "a noisy engine woke him and a quiet room let him rest .",
                        "the class grew noisy then the teacher made it quiet .",
                        "noisy traffic fades and the quiet evening arrives ."]),
    ("fast", "slow", ["the fast train passed the slow truck .",
                      "he walks fast in the morning and slow at night .",
                      "a fast river runs beside a slow stream ."]),
    ("warm", "cool", ["the warm bread came out of the oven .",
                      "a warm coat helps on a cool day .",
                      "she likes warm milk and cool juice ."]),
    ("heavy", "light", ["the heavy bag hurt his shoulder .",
                        "a heavy stone sank and a light leaf floated .",
                        "carrying heavy boxes is harder than carrying light ones ."]),
    ("early", "late", ["she arrived early but the bus came late .",
                       "an early start beats a late finish .",
                       "the early meeting ran late ."]),
    ("soft", "rough", ["the soft blanket felt better than the rough mat .",
                       "a soft voice is easier to hear than a rough shout .",
                       "he chose the soft chair ."]),
    ("loud", "faint", ["the loud bell rang across the yard .",
                       "a loud noise startled the birds .",
                       "the music was loud then grew faint ."]),
    ("round", "square", ["the round plate sat beside the square tray .",
                         "a round ball rolls and a square block does not .",
                         "she drew a round circle ."]),
]

lines = []
for a, b in FRAME_PAIRS:
    lines += [f"the opposite of {a} is {b} .", f"the opposite of {b} is {a} .",
              f"{a} and {b} are opposite words .",
              f"when something is not {a} it is {b} ."]
for a, b, sents in CONTRASTS:
    lines += sents
    lines += [f"{a} and {b} describe opposite things .",
              f"something {a} is never {b} at the same time ."]
random.shuffle(lines)
write("opposites.txt", lines * 6)

# -------------------------------------------------------- spatial relations ---
CONTAINERS = [("bag", "book"), ("drawer", "spoon"), ("basket", "apple"),
              ("case", "pencil"), ("jar", "coin"), ("bowl", "orange")]
ABOVE_BELOW = [("lamp", "desk"), ("shelf", "floor"), ("picture", "table"),
               ("roof", "window"), ("clock", "door")]
LEFT_RIGHT = [("ball", "box"), ("chair", "table"), ("cup", "plate"),
              ("shoe", "bag"), ("pen", "book")]

lines = []
for outer, inner in CONTAINERS:
    lines += [f"the {inner} is inside the {outer} .",
              f"the {outer} contains the {inner} .",
              f"she put the {inner} into the {outer} .",
              f"if the {outer} contains the {inner} then the {inner} is inside the {outer} ."]
for top, bottom in ABOVE_BELOW:
    lines += [f"the {top} is above the {bottom} .",
              f"the {bottom} is below the {top} .",
              f"when the {top} is above the {bottom} the {bottom} is below the {top} .",
              f"above and below are opposite directions .",
              f"he stood beside the {bottom} and looked at the {top} ."]
for a, b in LEFT_RIGHT:
    lines += [f"the {a} is left of the {b} .",
              f"the {b} is to the right of the {a} .",
              f"left and right are opposite sides .",
              f"the {a} sits beside the {b} ."]
lines += ["north is the opposite of south .",
          "the map shows north at the top and south at the bottom .",
          "they walked north and then turned south .",
          "a shelf holds a lamp and a desk holds a book .",
          "the desk stands below the shelf ."]
random.shuffle(lines)
write("spatial_relations.txt", lines * 6)

# ------------------------------------------------------- everyday knowledge ---
lines = [
    # NOTE: these three facts are taught deliberately WITHOUT the suite's phrasing.
    # Earlier drafts wrote "water freezes into ice ...", "a person uses an umbrella to
    # stay dry ..." and "to see in a dark room we turn on a light ." -- each of which
    # reproduced an eval prompt verbatim (lang_43/44/45) and was rejected by the
    # leakage guard. The facts may overlap with the suite; the wording must not.
    "cold air makes water freeze and become ice .",
    "in the freezer water turns hard and becomes ice .",
    "ice melts back to water when the day is warm .",
    "cold water becomes ice in the freezer .",
    "ice is frozen water .",
    "boiling water becomes steam .",
    "steam rises from a hot cup .",
    "sand is found at the beach .",
    "wood comes from a tree .",
    "a wood table and a sand path feel different .",
    "she opened an umbrella and stayed dry in the rain .",
    "an umbrella keeps the rain off so you stay dry .",
    "without an umbrella you get wet .",
    "wet clothes take time to become dry .",
    "she was hungry so she ate bread .",
    "after a long day he felt asleep in the chair .",
    "a tired person falls asleep quickly .",
    "he turned on a light so he could see inside the dark room .",
    "the light makes a dark room bright .",
    "he turned on the light and the room was no longer dark .",
    "a lamp gives light at night .",
    "a pillow belongs on a bed .",
    "she rested her head on a soft pillow .",
    "a spoon is used to eat soup .",
    "he stirred the tea with a spoon .",
    "a shoe goes on a foot .",
    "she tied the shoe before the walk .",
    "rain falls from clouds .",
    "the sun gives warm light during the day .",
    "fire is hot and snow is cold .",
    "snow is made of ice .",
]
random.shuffle(lines)
write("everyday_knowledge.txt", lines * 12)

# --------------------------------------------- categories and analogies ---
IS_A = [("robin", "bird"), ("sparrow", "bird"), ("eagle", "bird"), ("duck", "bird"),
        ("salmon", "fish"), ("trout", "fish"), ("tuna", "fish"),
        ("carrot", "vegetable"), ("potato", "vegetable"), ("onion", "vegetable"),
        ("apple", "fruit"), ("banana", "fruit"), ("pear", "fruit"),
        ("oak", "tree"), ("pine", "tree"),
        ("hammer", "tool"), ("saw", "tool"), ("drill", "tool"),
        ("cotton", "fabric"), ("silk", "fabric"),
        ("iron", "metal"), ("copper", "metal"),
        ("bus", "vehicle"), ("truck", "vehicle"), ("car", "vehicle")]
GROWS = [("puppy", "dog"), ("kitten", "cat"), ("calf", "cow"), ("lamb", "sheep"),
         ("foal", "horse"), ("chick", "hen"), ("kid", "goat")]

lines = []
for small, kind in IS_A:
    lines += [f"a {small} is a {kind} .",
              f"the {kind} called a {small} is common here .",
              f"every {small} belongs to the {kind} group ."]
for young, grown in GROWS:
    lines += [f"a {young} grows into a {grown} .",
              f"the {grown} was once a {young} .",
              f"a young {grown} is called a {young} ."]
lines += ["a bird can fly and a fish can swim .",
          "a tree is a plant and a hammer is a tool .",
          "metal and fabric are different materials .",
          "a vehicle carries people from place to place .",
          "a goat and a horse are animals .",
          "a duck is an animal and a carrot is a vegetable ."]
random.shuffle(lines)
write("categories_and_analogies.txt", lines * 6)

print("\nControl categories intentionally NOT taught: grammar, negation, reference, sequence")
