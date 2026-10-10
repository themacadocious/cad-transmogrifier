"""Drive stl2step's Tk GUI without a human: fake the file picker, press the button,
wait for "done.", print the log. Run under a display (tests/test_gui.py uses xvfb-run)."""
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stl2step  # noqa: E402

inputs = tuple(sys.argv[1:])
filedialog.askopenfilenames = lambda **k: inputs
orig_mainloop = tk.Tk.mainloop


def walk(w):
    yield w
    for c in w.winfo_children():
        yield from walk(c)


def mainloop(self, n=0):
    btn = next(w for w in walk(self) if isinstance(w, ttk.Button))
    log = next(w for w in walk(self) if w.winfo_class() == "Text")

    def poll(t=0):
        content = log.get("1.0", "end")
        if "done." in content or t > 240:
            print(content)
            self.destroy()
            return
        self.after(500, poll, t + 1)

    self.after(500, btn.invoke)
    self.after(1000, poll)
    orig_mainloop(self, n)


tk.Tk.mainloop = mainloop
stl2step.gui()
