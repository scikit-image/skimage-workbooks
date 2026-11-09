// MyST directives for the notebook header: `related` and `changelog`.
// Each body line holds fields separated by " | ". The last field is MyST
// Markdown, so it may hold links and cross-references.
//
//   :::{related}
//   https://github.com/scikit-image/scikit-image/pull/7890#issuecomment-1 | Asks why the border is reflected; see [§5](#sec-border)
//   :::
//
//   :::{changelog}
//   2026-11-09 | claude-code:claude-opus-5-5 | Add explanation of per-pixel Hessian
//   2026-09-22 | Jane Doe | Create workbook
//   :::

function splitLines(body, name, nFields) {
  return body
    .split('\n')
    .map((s) => s.trim())
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(' | ');
      if (parts.length < nFields) {
        throw new Error(`${name}: expected ${nFields} fields separated by " | ", got "${line}"`);
      }
      // The last field may itself contain " | ".
      return [...parts.slice(0, nFields - 1), parts.slice(nFields - 1).join(' | ')].map((s) =>
        s.trim(),
      );
    });
}

function inline(ctx, text) {
  if (!text) return [];
  const root = ctx.parseMyst(text);
  const first = root.children?.[0];
  return first?.type === 'paragraph' ? first.children : root.children;
}

const cell = (children, header = false) => ({ type: 'tableCell', header, children });
const text = (value) => ({ type: 'text', value });

function box(cls, title, headers, rows) {
  return [
    {
      type: 'div',
      class: `wb-meta ${cls}`,
      children: [
        { type: 'paragraph', class: 'wb-meta-title', children: [text(title)] },
        {
          type: 'table',
          children: [
            { type: 'tableRow', children: headers.map((h) => cell([text(h)], true)) },
            ...rows.map((r) => ({ type: 'tableRow', children: r.map((c) => cell(c)) })),
          ],
        },
      ],
    },
  ];
}

// Short link label: "scikit-image#7890 PR, comment" for GitHub, else the host.
function linkLabel(url) {
  const m = url.match(
    /^https:\/\/github\.com\/[^/]+\/([^/]+)\/(pull|issues|discussions)\/(\d+)(?:\/[^#]*)?(#.*)?$/,
  );
  if (!m) return new URL(url).hostname;
  const [, repo, kind, num, frag] = m;
  const what = { pull: 'PR', issues: 'issue', discussions: 'discussion' }[kind];
  const sub =
    frag && /^#(issuecomment|discussion_r|pullrequestreview|discussioncomment)/.test(frag);
  return `${repo}#${num} ${what}${sub ? ', comment' : ''}`;
}

const related = {
  name: 'related',
  doc: 'Pull requests, issues and discussions related to this notebook, one "<url> | <note>" per line.',
  body: { type: String, required: true },
  run(data, vfile, ctx) {
    const rows = splitLines(data.body, 'related', 2).map(([url, note]) => [
      [{ type: 'link', url, children: [text(linkLabel(url))] }],
      inline(ctx, note),
    ]);
    return box('wb-related', 'Related', ['Link', 'Note'], rows);
  },
};

const changelog = {
  name: 'changelog',
  doc: 'Changes to this notebook, newest first, one "<date> | <who> | <comment>" per line.',
  body: { type: String, required: true },
  run(data, vfile, ctx) {
    const rows = splitLines(data.body, 'changelog', 3).map(([date, who, comment]) => {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
        throw new Error(`changelog: expected a YYYY-MM-DD date, got "${date}"`);
      }
      // "<harness>:<model>" for an agent, a plain name for a person.
      const whoNodes = who
        .split(/,\s*/)
        .flatMap((w, i) => [
          ...(i ? [text(', ')] : []),
          w.includes(':') ? { type: 'inlineCode', value: w } : text(w),
        ]);
      return [[text(date)], whoNodes, inline(ctx, comment)];
    });
    return box('wb-changelog', 'Changelog', ['Date', 'By', 'Change'], rows);
  },
};

export default { name: 'Workbook metadata directives', directives: [related, changelog] };
