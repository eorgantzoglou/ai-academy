"""
AI Academy content validator.

Run:  python tools/validate.py
Checks every curriculum file for the mistakes that actually break the app:
  1. JS syntax          (via node --check, if node is available)
  2. stray backticks    (they terminate the JS template literal)
  3. stray ${           (JS interpolation inside a template literal)
  4. unbalanced ~~~     fences
  5. unbalanced :::     callouts
  6. required fields    on every track and topic
"""
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUR = os.path.join(ROOT, "js", "curriculum")

RED = "\033[31m"; GRN = "\033[32m"; YEL = "\033[33m"; OFF = "\033[0m"


def have_node():
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


NODE = have_node()


def check_file(path):
    name = os.path.basename(path)
    errors = []
    warns = []
    src = io.open(path, encoding="utf-8").read()

    # --- 1. JS syntax ------------------------------------------------
    if NODE:
        r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        if r.returncode != 0:
            first = [l for l in r.stderr.splitlines() if "Error" in l or ".js:" in l]
            errors.append("JS syntax: " + (first[0] if first else r.stderr[:200]))

    # --- 2. backticks ------------------------------------------------
    # A backtick is legitimate ONLY as a template-literal delimiter, i.e.
    #     intro: `        /  md: `        /  blurb: `
    #     `        /  `,   /  `}   (the closing delimiter, alone on its line)
    # Anywhere else it silently ends the string and breaks the file.
    tick = chr(96)
    open_re = re.compile(r"^\s*(intro|md|blurb|goal)\s*:\s*" + tick + r"\s*$")
    close_re = re.compile(r"^\s*" + tick + r"\s*[,}\]]*\s*$")
    for i, line in enumerate(src.split("\n"), 1):
        if tick not in line:
            continue
        if open_re.match(line) or close_re.match(line):
            continue
        errors.append("line %d: stray backtick - use ~~~ fences or ~inline~  |  %s"
                      % (i, line.strip()[:60]))

    # --- 3. ${ ------------------------------------------------------
    # A bare ${ is JS interpolation and will either crash the file or silently
    # substitute a value. Escaped as \${ it renders as a literal dollar-brace,
    # which is what Python f-strings like f"${x:,.0f}" actually need.
    bs = chr(92)
    for i, line in enumerate(src.split("\n"), 1):
        for j, ch in enumerate(line):
            if ch == "$" and j + 1 < len(line) and line[j + 1] == "{":
                if j == 0 or line[j - 1] != bs:
                    errors.append("line %d: bare ${ - escape it as %s${  |  %s"
                                  % (i, bs, line.strip()[:60]))
                    break

    # --- 4. fences --------------------------------------------------
    fences = [i for i, l in enumerate(src.split("\n"), 1) if l.startswith("~~~")]
    if len(fences) % 2 != 0:
        errors.append("odd number of ~~~ fences (%d); last at line %d"
                      % (len(fences), fences[-1] if fences else 0))

    # --- 5. callouts -------------------------------------------------
    # NOTE: [^\S\n] means "horizontal whitespace only" - plain \s would match
    # newlines and let a closing ::: pair up with text two lines below it.
    opens = len(re.findall(r"^:::[^\S\n]*[A-Za-z]", src, re.M))
    closes = len(re.findall(r"^:::[^\S\n]*$", src, re.M))
    if opens != closes:
        errors.append("callout mismatch: %d opened, %d closed" % (opens, closes))

    # --- 6. structure (parse the object the cheap way, via node) -----
    stats = {}
    if NODE and not errors:
        probe = (
            "global.window={};"
            "require(%s);"
            "const t=global.window.CURRICULUM[0];"
            "const need=['id','title','icon','level','blurb','topics'];"
            "const miss=need.filter(k=>!(k in t));"
            "if(miss.length)throw new Error('track missing: '+miss);"
            "let lv=0;"
            "t.topics.forEach(p=>{"
            "  ['id','title','summary','levels'].forEach(k=>{"
            "    if(!(k in p))throw new Error('topic '+p.id+' missing '+k);});"
            "  if(!Array.isArray(p.levels)||!p.levels.length)"
            "    throw new Error('topic '+p.id+' has no levels');"
            "  p.levels.forEach(l=>{if(!l.name||!l.md)"
            "    throw new Error('bad level in '+p.id);});"
            "  lv+=p.levels.length;"
            "  (p.quiz||[]).forEach((q,i)=>{"
            "    if(typeof q.answer!=='number'||!q.options||q.answer>=q.options.length)"
            "      throw new Error('bad quiz '+i+' in '+p.id);});"
            "});"
            "console.log(JSON.stringify({id:t.id,topics:t.topics.length,levels:lv,"
            "quizzes:t.topics.reduce((a,p)=>a+((p.quiz||[]).length),0)}));"
            % json.dumps(path.replace("\\", "/"))
        )
        r = subprocess.run(["node", "-e", probe], capture_output=True, text=True)
        if r.returncode != 0:
            errors.append("structure: " + r.stderr.strip().split("\n")[0][:180])
        else:
            try:
                stats = json.loads(r.stdout.strip())
            except Exception:
                pass

    return name, errors, warns, stats


def main():
    if not os.path.isdir(CUR):
        print("no curriculum directory yet")
        return 0

    files = sorted(f for f in os.listdir(CUR) if f.endswith(".js"))
    if not files:
        print("no curriculum files yet")
        return 0

    total_e = 0
    t_topics = t_levels = t_quiz = 0
    print("=" * 74)
    print("AI ACADEMY CONTENT VALIDATION" + ("" if NODE else "   (node not found - syntax check skipped)"))
    print("=" * 74)

    for f in files:
        name, errors, warns, stats = check_file(os.path.join(CUR, f))
        if errors:
            total_e += len(errors)
            print("%sFAIL%s  %s" % (RED, OFF, name))
            for e in errors[:8]:
                print("        - " + e)
            if len(errors) > 8:
                print("        ... and %d more" % (len(errors) - 8))
        else:
            t_topics += stats.get("topics", 0)
            t_levels += stats.get("levels", 0)
            t_quiz += stats.get("quizzes", 0)
            print("%s ok %s  %-32s %2d topics %3d levels %3d quiz questions"
                  % (GRN, OFF, name, stats.get("topics", 0),
                     stats.get("levels", 0), stats.get("quizzes", 0)))

    print("-" * 74)
    print("TOTAL: %d files | %d topics | %d levels | %d quiz questions"
          % (len(files), t_topics, t_levels, t_quiz))
    if total_e:
        print("%s%d problem(s) found.%s" % (RED, total_e, OFF))
        return 1
    print("%sAll content valid.%s" % (GRN, OFF))
    return 0


if __name__ == "__main__":
    sys.exit(main())
