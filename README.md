# SLR(1) Parser — `slr-parser`

Small SLR(1) parser generator example implemented in `slr.py`.

This README explains how to run the parser in a Python virtual environment on macOS (zsh).

## Requirements

- Python 3.8+ (3.10 or later recommended)
- `rich` (used for pretty tables)
- Graphviz (optional — to visualize the exported DOT file)

If you prefer not to install a `requirements.txt`, you can install packages directly:

```
python3 -m pip install --upgrade pip
python3 -m pip install rich
```

To visualize the LR(0) automaton (optional):

```
# macOS (Homebrew)
brew install graphviz
```

## Setup (create and activate virtual environment)

Run the following from the `slr-parser` directory (zsh):

```
# create venv
python3 -m venv .venv

# activate (zsh)
source .venv/bin/activate

# upgrade pip and install dependencies
pip install --upgrade pip
pip install rich
```

If you have a `requirements.txt` in your project root, you can instead run:

```
pip install -r requirements.txt
```

## Running the parser

From the `slr-parser` folder with the virtualenv activated:

```
python3 slr.py
```

What this does:
- Computes FIRST and FOLLOW sets for the example grammar in `slr.py`.
- Builds and prints the SLR(1) `ACTION` and `GOTO` tables (rendered with `rich`).
- Exports the LR(0) automaton as `lr0_automaton.dot` in the same directory.
- Runs a few built-in test inputs and prints whether they are accepted.

## Visualizing the LR(0) automaton

After running `python3 slr.py` you will have `lr0_automaton.dot`. To convert it to PNG:

```
dot -Tpng lr0_automaton.dot -o lr0_automaton.png
open lr0_automaton.png
```

Or create an SVG:

```
dot -Tsvg lr0_automaton.dot -o lr0_automaton.svg
open lr0_automaton.svg
```

## Using the parser on custom input

The script includes a `parse` method and a small test harness. By default the script tests several sample token lists. To test a custom token list (for example `a b b`), modify the `tests` list at the bottom of `slr.py` or open an interactive Python session:

```
python3 -i slr.py
# then in the interactive prompt:
G.parse(['a','b','b'])
```

Notes
- The grammar used in the example is:

```
S -> A A
A -> a A | b
```

- The parser expects tokens as simple strings matching the grammar terminals (e.g. `'a'`, `'b'`).

## Troubleshooting

- If `rich` is not installed you will see an ImportError; install with `pip install rich`.
- If `dot` is not found when converting DOT files, install Graphviz (`brew install graphviz`).

## Files

- `slr.py` — parser implementation and example harness
- `lr0_automaton.dot` — generated after running `python3 slr.py`

---
If you'd like, I can also add a `requirements.txt` to the `slr-parser` folder or a small wrapper script that accepts a token string from the command line.