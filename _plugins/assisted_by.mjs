// MyST directive `assisted-by`: render the agents that worked on a notebook
// as a small table. One agent per body line, in the commit-tag form
// `<harness>:<model>`, optionally followed by a note.
//
//   :::{assisted-by}
//   claude-code:claude-fable-5-1
//   claude-code:claude-opus-5  figures and corpus section
//   :::

function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function parseLine(line) {
  const [tag, ...rest] = line.split(/\s+/);
  const i = tag.indexOf(':');
  if (i < 0) {
    throw new Error(`assisted-by: expected "<harness>:<model>", got "${line}"`);
  }
  return { harness: tag.slice(0, i), model: tag.slice(i + 1), note: rest.join(' ') };
}

const style = {
  box:
    'display:inline-block;margin:0 0 1.5em 0;padding:0.5em 0.9em;' +
    'background:#eaf3fc;border:1px solid #c5dcf3;border-radius:6px;' +
    'font-size:0.85em;color:#2b3a4a;',
  title: 'font-weight:600;margin:0 0 0.3em 0;',
  table: 'border-collapse:collapse;margin:0;',
  th: 'text-align:left;padding:0.1em 1.2em 0.1em 0;font-weight:500;color:#52514e;',
  td: 'padding:0.1em 1.2em 0.1em 0;font-family:monospace;',
  note: 'padding:0.1em 0;font-family:inherit;',
};

const assistedBy = {
  name: 'assisted-by',
  doc: 'Agents that worked on this notebook, one "<harness>:<model>" per line.',
  body: {
    type: String,
    required: true,
    doc: 'One agent per line: <harness>:<model>, then an optional note.',
  },
  run(data) {
    const rows = data.body
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean)
      .map(parseLine);
    const hasNote = rows.some((r) => r.note);
    const head =
      `<tr><th style="${style.th}">Harness</th><th style="${style.th}">Model</th>` +
      (hasNote ? `<th style="${style.th}">Note</th>` : '') +
      '</tr>';
    const body = rows
      .map(
        (r) =>
          `<tr><td style="${style.td}">${escapeHtml(r.harness)}</td>` +
          `<td style="${style.td}">${escapeHtml(r.model)}</td>` +
          (hasNote ? `<td style="${style.note}">${escapeHtml(r.note)}</td>` : '') +
          '</tr>',
      )
      .join('');
    const html =
      `<div class="assisted-by" style="${style.box}">` +
      `<div style="${style.title}">Assisted by</div>` +
      `<table style="${style.table}">${head}${body}</table></div>`;
    return [{ type: 'html', value: html }];
  },
};

const plugin = { name: 'Assisted-by directive', directives: [assistedBy] };

export default plugin;
