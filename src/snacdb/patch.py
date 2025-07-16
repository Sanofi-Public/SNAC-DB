def run_patch():
    """Locate installed ANARCI and apply the amino_acids fix."""
    import os, re, subprocess

    try:
        out = subprocess.check_output(["pip", "show", "anarci"]).decode()
    except subprocess.CalledProcessError:
        print("⚠️  ANARCI not found; skipping patch")
        return

    pkg_dir = next(
        line.split(":",1)[1].strip()
        for line in out.splitlines()
        if line.startswith("Location:")
    )
    pyfile = os.path.join(pkg_dir, "anarci", "anarci.py")

    if not os.path.isfile(pyfile):
        print(f"⚠️  {pyfile} missing; skipping patch")
        return

    text = open(pyfile).read()
    patched = re.sub(
        r'amino_acids\s*=\s*sorted\(list\("[^"]*"\)\)',
        'amino_acids = sorted(list("QWERTYIPASDFGHKLCVNMX"))',
        text
    )
    if text == patched:
        print("ℹ️  Already patched")
    else:
        open(pyfile + ".bak","w").write(text)
        open(pyfile, "w").write(patched)
        print(f"✅  Patched {pyfile}")

if __name__ == "__main__":
    run_patch()
