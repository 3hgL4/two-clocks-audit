import inspect
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tools")

import model as M
import train as T

print("T.Policy is M.Policy:", T.Policy is M.Policy)
print("score qualname:", T.Policy.score.__qualname__, "| module:", T.Policy.__module__)


def patched(self, p, c, *, requires_grad=False, fp32=True):
    print("PATCHED CALLED", flush=True)
    return M._orig_score(self, p, c, requires_grad=requires_grad, fp32=fp32)


M.Policy.score = patched
print("after patch, T.Policy.score is patched:", T.Policy.score.__name__)

src = inspect.getsource(T.main)
for ln in src.splitlines():
    if "score(" in ln:
        print("MAIN-LINE:", ln.strip())
