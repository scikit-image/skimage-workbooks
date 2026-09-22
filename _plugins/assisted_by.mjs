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

// Styling lives in _plugins/assisted_by.css, loaded via site.options.style.

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
      '<tr><th>Harness</th><th>Model</th>' + (hasNote ? '<th>Note</th>' : '') + '</tr>';
    const body = rows
      .map(
        (r) =>
          `<tr><td>${escapeHtml(r.harness)}</td><td>${escapeHtml(r.model)}</td>` +
          (hasNote ? `<td class="ab-note">${escapeHtml(r.note)}</td>` : '') +
          '</tr>',
      )
      .join('');
    const html =
      '<div class="assisted-by">' +
      '<div class="ab-title">Assisted by</div>' +
      `<table>${head}${body}</table></div>`;
    return [{ type: 'html', value: html }];
  },
};

const plugin = { name: 'Assisted-by directive', directives: [assistedBy] };

export default plugin;
