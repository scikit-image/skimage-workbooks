"""Project nbconvert config: LaTeX/PDF template for pandoc longtables."""

from pathlib import Path

c = get_config()  # noqa: F821

_TEMPLATES = str(Path(__file__).resolve().parent / "_nbconvert_templates")

c.TemplateExporter.extra_template_basedirs = [_TEMPLATES]
c.LatexExporter.template_name = "latex_pandoc_tables"
c.PDFExporter.template_name = "latex_pandoc_tables"
