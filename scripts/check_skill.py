"""Checks the skill against the Agent Skills rules and that its local links resolve.

- SKILL.md has frontmatter with a valid name that matches the folder, and a description
  of 1 to 1024 characters.
- Only fields that every Agent Skills host accepts are used.
- SKILL.md stays under 500 lines.
- Every relative markdown link in the skill and the rules points at a file that exists.
- No em dashes, curly quotes or other typographic characters in any markdown file.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "ai-evals"
PORTABLE = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
errors = []

text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
if not match:
    errors.append("SKILL.md: missing frontmatter")
else:
    front = match.group(1)
    keys = {line.split(":", 1)[0] for line in front.splitlines() if line and not line.startswith(" ")}
    if keys - PORTABLE:
        errors.append(f"SKILL.md: non-portable frontmatter keys {sorted(keys - PORTABLE)}")
    name = re.search(r"^name:\s*(\S+)\s*$", front, re.M)
    if not name or name.group(1) != SKILL_DIR.name:
        errors.append("SKILL.md: name must match the folder name")
    elif not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name.group(1)) or len(name.group(1)) > 64:
        errors.append("SKILL.md: name must be lowercase letters, digits and single hyphens, at most 64 chars")
    desc = re.search(r"^description:\s*(.+)$", front, re.M)
    if not desc or not 1 <= len(desc.group(1).strip()) <= 1024:
        errors.append("SKILL.md: description must be 1 to 1024 characters")
if len(text.splitlines()) >= 500:
    errors.append("SKILL.md: keep it under 500 lines")

files = sorted(p for p in ROOT.rglob("*.md*") if "node_modules" not in p.parts and ".git" not in p.parts)
for path in files:
    body = path.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#]+)\)", body):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (path.parent / target).resolve().exists():
            errors.append(f"{path.relative_to(ROOT)}: broken link {target}")
    banned = {0x2014: "em dash", 0x2013: "en dash", 0x201C: "curly quote", 0x201D: "curly quote",
              0x2018: "curly apostrophe", 0x2019: "curly apostrophe", 0x2026: "ellipsis character",
              0x00A0: "non-breaking space"}
    for code, label in banned.items():
        if chr(code) in body:
            errors.append(f"{path.relative_to(ROOT)}: contains an {label}")

if errors:
    print("\n".join(errors))
    sys.exit(1)
print(f"ok: {len(files)} files checked")
