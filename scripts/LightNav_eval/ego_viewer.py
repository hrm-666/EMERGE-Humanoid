"""Display the original simulation's live ego RGB stream without controlling it."""

from __future__ import annotations

import argparse
import io
import math
import time
from pathlib import Path

from PIL import Image
from PIL import ImageTk

from robot.vla.lightnav_sonic.official01_observation import Official01Subscriber


class EgoViewer:
    """Independent read-only subscriber; never takes focus from keyboard control."""

    def __init__(self, root, subscriber, args: argparse.Namespace) -> None:
        import tkinter as tk

        self.root = root
        self.subscriber = subscriber
        self.args = args
        self.photo = None
        self.last_timestamp = None
        root.title("LightNav · Robot ego camera (read only)")
        root.resizable(False, False)
        canvas = tk.Frame(root, width=640, height=480)
        canvas.pack()
        canvas.pack_propagate(False)
        self.image = tk.Label(canvas, background="#202020")
        self.image.pack(fill="both", expand=True)
        self.status = tk.Label(root, anchor="w", padx=8, pady=8)
        self.status.pack(fill="x")
        root.protocol("WM_DELETE_WINDOW", root.destroy)

    def refresh(self) -> None:
        frame = self.subscriber.latest(self.args.max_age_s)
        if frame is None:
            # Hide frozen pixels: an old frame must not look like a live feed.
            self.image.configure(image="")
            self.photo = None
            self.last_timestamp = None
            self.status.configure(
                text=
                f"WAITING / STALE · {self.subscriber.error or 'no fresh camera frame'}",
                foreground="#b00020",
            )
        else:
            if frame.wall_time_s != self.last_timestamp:
                with Image.open(io.BytesIO(frame.jpeg)) as rgb:
                    self.photo = ImageTk.PhotoImage(rgb, master=self.root)
                self.image.configure(image=self.photo)
                self.last_timestamp = frame.wall_time_s
                if self.args.snapshot_path is not None:
                    path = self.args.snapshot_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + ".tmp")
                    temporary.write_bytes(frame.jpeg)
                    temporary.replace(path)
            age_ms = max(0.0, time.time() - frame.wall_time_s) * 1000
            self.status.configure(
                text=f"LIVE · head_camera · 640 × 480 · age {age_ms:.0f} ms "
                f"· sim {frame.sim_time_s:.2f} s",
                foreground="#176b35",
            )
        self.root.after(round(1000 / self.args.refresh_hz), self.refresh)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-endpoint",
                        default="tcp://10.254.240.2:5560")
    parser.add_argument("--max-age-s", type=float, default=1.0)
    parser.add_argument("--refresh-hz", type=float, default=20.0)
    parser.add_argument("--snapshot-path",
                        type=Path,
                        default=Path("outputs/tabletop_ego/latest.jpg"))
    return parser


def main() -> None:
    import tkinter as tk

    parser = build_parser()
    args = parser.parse_args()
    for name in ("max_age_s", "refresh_hz"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(
                f"--{name.replace('_', '-')} must be finite and positive")
    if args.refresh_hz > 100:
        parser.error("--refresh-hz must not exceed 100")
    subscriber = Official01Subscriber(args.observation_endpoint)
    try:
        root = tk.Tk()
        viewer = EgoViewer(root, subscriber, args)
        viewer.refresh()
        print(
            "EGO_VIEWER=READY (read only; waiting for original camera stream)",
            flush=True)
        root.mainloop()
    finally:
        subscriber.close()


if __name__ == "__main__":
    main()
