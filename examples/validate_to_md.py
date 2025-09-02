import io, contextlib
from examples.validate_simple import run
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    run()
text = buf.getvalue()
open("VALIDATION.md","w",encoding="utf-8").write(
    "# Validation Report\n\n```\n"+text+"\n```\n"
)
print("Wrote VALIDATION.md")
