---
name: diagnosis-notebook
description: Write a runnable MyST Markdown notebook that diagnoses one specific technical problem - a disagreement between two functions, a suspected bug, a surprising numerical result - and shows the evidence, the mechanism, the candidate fixes, and what each fix would change. Use when asked to explain, diagnose, demonstrate or write up a problem in notebook form, or for a notebook named `on_<topic>.md`. Produces a `.md` notebook paired to `.ipynb` through jupytext.
---

# Diagnosis notebook

A diagnosis notebook discusses one problem, or a set of problems.

It keeps a tight focus on the *nature* of the problem, and proceeds first by the results of research on the problem, second by concise but readable mathematical analysis of the problem, and third by demonstration of the problem suitable simulations or code examples.

## 1. Investigate first, write second

Do the whole investigation before you write a line of the notebook. The
notebook reports findings. It is not where you look for them.

Every number in the notebook must come from a cell that computes it. Never
paste a figure measured elsewhere unless the notebook re-derives it. If you
write prose first, you will assert things you have not measured, and the
measurement will contradict you.

Where possible, prefer simple mathematical demonstrations, followed by their
instantiation in code.

Where possible, prove current errors, or potential solutions, and then show that these work in implementations.

## 2. Plan to explain from first principles

Once you have identified a problem, seek to explain the basis of the problem to the reader, as an intelligent outsider to the field.  Do not go into great depth, but identify the key ideas behind the algorithms used, and investigate the tutorial and explanatory resources available.  Link to these resources where useful, and, where possible, base your explanations on the identified tutorial starting points.

Consider Wikipedia a good start for technical discussions, and look for links in Wikipedia pages to other tutorials.  Also use tutorials from standard software tutorial and explanatory sites, such at the Scipy manual and tutorial pages.

## 3. When in doubt, explain with graphics

Tend to produce plots and graphics, in addition to text explanations.  Where possible, produce these with simple code.  Consider linking useful graphical explanations in other tutorials.

## 4. Resolve the output directory

Take the first that applies, and say which you used:

1. A path given in the request.
2. `$DIAGNOSIS_NOTEBOOK_DIR`.
3. The first line of a `.notebook-out` file, searching upward from the working
   directory.
4. A directory named by an "Output" instruction in the nearest `AGENTS.md`.
5. The working directory.

To set a project default, write the directory into `.notebook-out`.

## 5. Ensure a jupytext config

Look for `jupytext.toml`, `jupytext.yaml`, `jupytext.yml`, `.jupytext.toml`, or
a `[tool.jupytext]` table in `pyproject.toml`, in the output directory and its
parents. If one exists, leave it alone.

Only if none exists, write `jupytext.yaml` in the output directory:

```yaml
# https://jupytext.readthedocs.io/en/latest/config.html
# Pair ipynb notebooks to MyST Markdown text notebooks.
formats: ipynb,md:myst
```

## 6. Name, frontmatter and table of contents

Name the file `on_<topic>.md`: lower case, underscores, no date.

The notebook is a MyST Markdown document (https://mystmd.org), so its
frontmatter carries the MyST fields as well as the notebook metadata. Start the
file with:

```
---
title: <Title Case title>
date: <YYYY-MM-DD, the day you write the notebook>
options:
  updated: <YYYY-MM-DD, the day of this edit>
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
```

`date` is the date of writing. Do not change it when you edit the notebook
later. Set `options.updated` to the current date on every edit. MyST has no
standard modification-date field, so `updated` goes under `options`, which
takes arbitrary keys. No template renders it; it is there for the reader of
the source.

The repository `jupytext.toml` sets `formats = "ipynb,md:myst"`, so a per-file
`jupytext:` block is not needed. Jupytext adds one back when it writes the file
from a paired `.ipynb`; leave what it writes alone. `kernelspec` is needed to
execute the notebook.

With `title` in the frontmatter, do not repeat the title as a `#` heading in
the body. Start the body sections at `##`.

Add a new notebook to the top of the `toc` list in `myst.yml`, directly after
`index.md`. The list is in order of creation, newest first. An edit to a
notebook does not move it: its place follows `date`, not `options.updated`.

## 7. Shape

This order earns its keep. Keep the numbered sections; drop any that has
nothing to say.

1. **Title and the question.** Two or three sentences: what disagrees, and what
   the notebook settles. Add any convention the reader needs first, such as
   which coordinate order the document uses.
2. **Setup**, then **helpers**. See section 6.
3. **The problem**, on the smallest example that shows it. One picture.
4. **One section per mechanism.** Explain each algorithm in prose plus Python
   code where concise, pseudocode otherwise. Be careful to explain all
   variables used in code and pseudocode.  Then show its result on that same
   small example.
5. **Side by side** on the small example, with the difference visible.
6. **How often, and how big.** Measure over a corpus, not one case. State the
   corpus. A rate that depends on the corpus must say so.
7. **Properties that ought to hold.** Symmetry, invariance, agreement with an
   independent reference. This is where a defect becomes undeniable, because
   you are testing a promise rather than comparing two opinions.
8. **Comparators.** Other libraries, or a construction you can prove correct
   (padding once, an exact integer method, an analytic answer). Comparators
   turn "these differ" into "this one is wrong".
9. **Ways forward.** Implement each candidate fix as a runnable cell. Show
   before and after.
10. **What each fix costs.** Measured: how many results change, how much
    slower, what breaks.
11. **What not to do.** The tempting wrong conclusion, and why it fails.
12. **Summary table**, then the limits: corpus size, versions, what was not
    tested.

## 8. Cells

- **One idea per cell.** Split the setup: plotting imports; the subject under
  test; comparators; palette and `rcParams`. A one-line comment heads a group
  when its purpose is not obvious, as in `# Comparators.`
- **Plain imports.** No `try`/`except ImportError` guards and no optional-
  dependency branches. State the dependency and let it fail loudly.
- **Real headings.** Use `###` for a subsection, never a bold run-in such as
  `**Translation invariance.**`. Separate adjacent markdown cells with `+++`.
- Put helpers in their own section with a one-line docstring each. Everything
  after should be built from them.
- Prose between cells says what the reader is about to see and what it means.
  A cell with no lead-in is a cell the reader skips. Write it in the plain
  style of section 16.
- You can be relatively liberal using comments in notebook code.  Shorter is
  generally better.  If you're explaining code rather than ideas, it is OK to have a longer comment.
- If outputing a *table*, rather than raw code 
- When adding a heading, start a new cell.  Any heading should either be on
  its own in a cell, or be the first line in a cell.

## 9. Figures

- Load the `dataviz` skill before writing the first line of chart code.
- Validate any categorical palette with that skill's script; record the result
  in a comment beside the colours.
- Sequential data: one hue, light to dark. Diverging: two hues with a neutral
  midpoint. Never a rainbow.
- Prefer a picture of the actual objects (pixels, arrays, fields) over an
  abstract chart. A small table beats a bar chart of four numbers.
- Use object-oriented matplotlib: `fig, ax = plt.subplots()`, then `ax` methods.
- **Render every figure and look at it** before delivering. Check for hidden
  lines, collisions and overflow. The validator checks colour, not layout.

## 10. Mathematics

- LaTeX format mathematical notation and symbols ($a = b^2 + c$, $\alpha$,
  rather than typewriter font (`a = b ** 2 + c`) or Unicode.
- Where practical use Sympy to prove short mathematical proofs.  Where the
  proofs are longer than 10 lines, split out into a separate notebook and
  refer back.  If the proof is longer than 20 lines, report, and request
  confirmation.

## 11. Tables

- Prefer to generate tables using Pandas, rather than printing to the console,
  to give better in-notebook and rendered display.  But ensure that the
  generated display is accurate.  If you can't be sure, print to the console.

## 12. Verify before delivering

Execute every cell. A notebook that has not run is worthless.

```bash
MPLBACKEND=Agg python3 - << 'PY'
import re, sys, time, pathlib, traceback
import matplotlib.pyplot as plt
cells = re.findall(r'^```\{code-cell\} ipython3\n(.*?)^```$',
                   pathlib.Path('on_topic.md').read_text(), re.M | re.S)
ns, t0, figs = {'__name__': '__main__'}, time.time(), 0
for n, code in enumerate(cells, 1):
    try:
        exec(compile(code, f'<cell {n}>', 'exec'), ns)
    except Exception:
        print(f'cell {n} FAILED'); traceback.print_exc(limit=2); sys.exit(1)
    figs += len(plt.get_fignums()); plt.close('all')
print(f'OK: {len(cells)} cells, {figs} figures, {time.time()-t0:.0f}s')
PY
```

Then check the format round-trips: `jupytext.read('on_topic.md')`.

Keep the whole notebook under about 1 minute. If a corpus loop dominates,
shrink it and say in the notebook that you did. Report the cell count, figure
count and runtime when you hand it over.

## 13. Honesty

- First, if possible, try to prove a claim must be true mathematically.  Test
  the proof carefully with implementation.
- Next, measure. Do not assert without measurement. "They differ" or "they are
  the same" needs a number and a corpus.
- A difference is not a defect. To call something a bug, show a promise it
  breaks: a documented contract, an assertion in the project's own tests, or a
  reference construction that both candidates miss.
- Say where your argument depends on interpretation, and give the strongest
  version of the opposing reading.
- State the limits plainly at the end: corpus, ranges, versions, what was not
  compared.
- If a fix changes results, say how much and for what fraction of inputs.
- These are the standards. Section 12 is the pass that checks you met them:
  a measurement beside a sentence is not evidence *for* that sentence until
  you can say which cell would change if the sentence were false.

## 14. Claim audit

Section 13 says do not assert without measurement. That rule is necessary and
it is not sufficient, because the commonest way to write a false sentence in
one of these notebooks is to write it *next to a real measurement that does not
bear on it*. Every wrong claim worth remembering has had a correct number
sitting beside it.

So run one pass over the finished draft with a question that has a yes or no
answer:

> **For each load-bearing sentence, name the cell whose printed output would be
> different if that sentence were false.**

If you cannot name one, you have three honest options: add a cell that
discriminates, weaken the sentence to what was actually measured, or delete it.
"It is obviously true" is not one of them. The claims that survive self-review
are exactly the ones that look obvious.

### The four ways these go wrong

Only the first looks like asserting without measurement. The other three all
have measurements.

| class | what it looks like |
| --- | --- |
| No cell at all | A mechanism stated in prose, with nothing that would break if the mechanism were different |
| A cell measuring something adjacent | The number is real; it is a number about a neighbouring quantity, or against a denominator that hides the effect |
| A cell whose output contradicts the prose | The evidence is right there, printed, and nobody re-read it |
| A contradiction with an earlier section | The document argues against itself in two places, and only one of them is right |

### The pass

1. **Mark the load-bearing sentences.** Those containing *because, so,
   therefore, never, always, only, cannot, every, all* — causal and universal
   claims. Descriptive sentences and quoted measurements are not the risk and
   do not need auditing. Expect a handful per section, not dozens.
2. **Name the discriminating cell for each.** Not a cell that confirms the
   claim; one whose output *changes* under its negation.
3. **Prefer the falsifying case to the confirming one.** Run the other parity,
   the other direction, the other sign, the reversed argument order, a second
   image. A claim about "odd sizes" needs a scan over sizes, not one odd size
   that works.
4. **Compute every literal that appears in a figure.** A hand-written pixel
   set, highlighted band, threshold or annotation is an unaudited claim that
   renders beautifully and executes without error. If the caption says "the
   pixels it touches", the code must compute which pixels it touches.
5. **Every percentage names its denominator in the sentence**, not only in the
   code. "38% out" is not yet a claim. Beware an aggregate over elements when
   one element carries all the error: the maximum over a set is the wrong
   denominator whenever most of the set is exactly right.
6. **Read each cell's real output against the sentence that introduces it —
   figures included.** Mechanical, cheap, and it catches the whole third class.
   Do this after the execution pass of section 10, on the executed outputs, not
   on the source. Section 9 already asks you to look at every figure; that is a
   check on layout. This is a different question about the same image: does the
   picture show the *mechanism* the caption claims? A panel can be beautifully
   laid out and demonstrate the opposite of the sentence above it.
7. **Search the document for the negation before asserting a limitation.**
   Before writing "there is no way to do X", grep for X. A long document
   contradicting itself is the failure that survives longest, because each half
   looks fine on its own.

### When the conclusion holds but the reason does not

This is the case worth naming separately, because it is the most seductive.
A measurement confirms the conclusion; a plausible mechanism is then written
down beside it; the mechanism is wrong and nothing tests it.

The remedy is to say only what was measured:

> Integer endpoints never reach the guard — measured, 50400 of 50400 pairs.

and to add the mechanism only once a cell distinguishes it from the
alternatives. A conclusion with no stated reason is honest. A conclusion with
an invented reason is worse than either, because it teaches the reader
something false while looking better evidenced.

### Limits

This pass costs real time and it does not catch everything. It is weakest
against a claim that is wrong in a way no available measurement addresses — one
about the intent of the original author, say, or about what users typically do.
For those, say plainly that the claim is a reading rather than a result.

It is also not a substitute for a second reader. Review by a different agent or
person reliably finds things this pass does not, particularly in the first and
fourth classes. Run the audit to raise the floor, not to skip the review.

Rules 5, 6 and 7 apply to any document that mixes measurement with explanation,
including plans and reviews, not only to notebooks.

## 15. Reference papers

- `library` is a symlink that may point to local, private copies of the
  reference papers, describing implementations.  Look there for relevant
  papers, but do not copy significant text content from those papers, which
  have specific copyright.  You can reproduce equations, and short quotes, but
  nothing longer than that.  Do not reproduce figures.

- The team keeps shared PDFs in the Zotero group library:
  https://www.zotero.org/groups/6683409/skimage/items. Search it for a paper
  before you look elsewhere.

- When the notebook cites a paper that has no public full text (no publisher
  open access, no arXiv or other preprint, no author copy), tell me to upload
  the PDF to the Zotero group library. Say which paper, and give the citation.
  Do not upload it yourself.

## 16. Reflection

- As you write the notebook, I will give you feedback.
- When the feedback is me asking for more explanation, there is no need to
  refer to my question, or your previous answer, in your explanation, when writing to the notebook.  Refer to my question and the previous answer in your model output in the UI, but not in the notebook.

## 16. Prose style

Write in ASD-STE100 Simplified Technical English. Short sentences, one idea in
each. Active voice, present tense. One term for one concept, used again and
again — do not change the word to make the paragraph more elegant.

The failure to guard against is fluff: prose that describes the document, or
announces the value of what comes next, instead of saying the thing. It reads
as fluent, it survives review, and it carries no information.

**Do not announce.** Delete any sentence whose job is to tell the reader that
the next sentence is worth reading.

| no | yes |
| --- | --- |
| "Worth seeing rather than asserting. Three pixels of one scene:" | "Three pixels of one scene:" |
| "which is worth checking rather than taking on trust" | "Check it:" |
| "Everything below is about eigenvalues of the Hessian, and it is worth being precise about what matrix that is" | "The eigenvalues are those of one matrix per pixel." |
| "The step that makes α intelligible is in the appendix." | "The appendix gives the step:" |

**Do not describe the document.** The reader can see its shape. An
introduction states the question and the convention, in two or three sentences,
and stops. It does not list the sections, explain what each one contributes, or
rank itself against another notebook. A cross-reference gives the section and
what is there, and nothing else: "`on_meijering.md` §5.2a measures the cost on
test images" — not "…and is the place for the defect argument; this notebook is
the explanation behind it."

**Do not grade your sources.** Cite and quote. "Lindeberg's §5.6.2 puts it
plainly" tells the reader about Lindeberg's writing; delete the praise and keep
the quotation, which does the work.

**Ban these openers:** *it is worth noting*, *note that*, *in fact*, *of
course*, *simply*, *just*, *interestingly*, *importantly*, *the key insight is*,
*what is left is*, *which makes it*. Each one is either empty or is doing a job
the following clause already does.

**Avoid rhetorical shape.** No triples for rhythm ("three images, three
different lies"), no one-line paragraph landed for effect, no metaphor where a
plain name exists. A figure of speech that names a real property once is
acceptable; repeating it as if it were terminology is not.

**The test.** Delete the sentence and read the paragraph again. If nothing is
lost, it was fluff. Run this over the whole draft, in the same pass as
section 14.
