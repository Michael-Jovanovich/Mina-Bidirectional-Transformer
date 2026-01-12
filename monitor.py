"""Training monitor for Mina bidirectional transformer."""
import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import ttk


CONTROL_FILE = Path("training_control.json")
LN2 = math.log(2)

# Sentinel for "no best yet" that is JSON-safe and displays nicely
SENTINEL_BEST_LOSS = 1e9


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize values to be JSON-serializable (handles inf/nan)."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj):
            return SENTINEL_BEST_LOSS if obj > 0 else -SENTINEL_BEST_LOSS
        if math.isnan(obj):
            return 0.0
        return obj
    return obj


@dataclass
class TrainingState:
    """Current training state with per-position improvement metrics."""
    epoch: int = 0
    batch: int = 0
    total_batches: int = 0
    # Loss components
    fwd_loss: float = 0.0
    bwd_loss: float = 0.0
    corr_loss: float = 0.0
    warmth_pred_loss: float = 0.0
    improvement_loss: float = 0.0
    # Validation
    val_fwd_loss: float = 0.0
    val_bwd_loss: float = 0.0
    val_corr_loss: float = 0.0
    # Per-position improvement metrics
    improvement_rate: float = 0.0
    val_improvement_rate: float = 0.0
    # General
    best_val_loss: float = SENTINEL_BEST_LOSS
    tokens_per_sec: float = 0.0
    elapsed_sec: float = 0.0


def request_stop() -> None:
    """Signal the training loop to stop and save."""
    CONTROL_FILE.write_text(json.dumps({'command': 'stop_and_save'}))


def check_stop_requested() -> bool:
    """Check if stop was requested."""
    if not CONTROL_FILE.exists():
        return False
    try:
        data = json.loads(CONTROL_FILE.read_text())
        if data.get('command') == 'stop_and_save':
            CONTROL_FILE.unlink()
            return True
    except (json.JSONDecodeError, KeyError):
        pass
    return False


class TrainingMonitor:
    """Writes training metrics to JSON with per-position improvement tracking."""

    def __init__(self, log_path: str = "training_log.json", history_path: str = "training_history.jsonl"):
        self.log_path = Path(log_path)
        self.history_path = Path(history_path)
        self.state = TrainingState()
        self.history: list[dict] = []
        self.start_time = time.time()
        self._batch_start = time.time()

    def update(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        fwd_loss: float,
        bwd_loss: float,
        corr_loss: float = 0.0,
        warmth_pred_loss: float = 0.0,
        improvement_loss: float = 0.0,
        improvement_rate: float = 0.0,
        tokens: int = 0,
    ) -> None:
        """Update training state after a batch."""
        self.state.epoch = epoch
        self.state.batch = batch
        self.state.total_batches = total_batches
        self.state.fwd_loss = fwd_loss
        self.state.bwd_loss = bwd_loss
        self.state.corr_loss = corr_loss
        self.state.warmth_pred_loss = warmth_pred_loss
        self.state.improvement_loss = improvement_loss
        self.state.improvement_rate = improvement_rate
        self.state.elapsed_sec = time.time() - self.start_time

        elapsed = time.time() - self._batch_start
        if elapsed > 0:
            self.state.tokens_per_sec = tokens / elapsed
        self._batch_start = time.time()

        if batch % 100 == 0:
            self._save_state()

    def update_validation(self, val_metrics: dict) -> None:
        """Update validation metrics."""
        self.state.val_fwd_loss = val_metrics['forward']
        self.state.val_bwd_loss = val_metrics['backward']
        self.state.val_corr_loss = val_metrics['correction']
        self.state.val_improvement_rate = val_metrics.get('improvement_rate', 0.0)

        if val_metrics['forward'] < self.state.best_val_loss:
            self.state.best_val_loss = val_metrics['forward']

        record = {
            'epoch': self.state.epoch,
            'train_fwd': self.state.fwd_loss,
            'train_bwd': self.state.bwd_loss,
            'train_corr': self.state.corr_loss,
            'train_improvement_rate': self.state.improvement_rate,
            'val_fwd': val_metrics['forward'],
            'val_bwd': val_metrics['backward'],
            'val_corr': val_metrics['correction'],
            'val_improvement_rate': val_metrics.get('improvement_rate', 0.0),
            'val_mean_improvement': val_metrics.get('mean_improvement', 0.0),
            'val_fwd_bpc': val_metrics['forward'] / LN2,
            'val_corr_bpc': val_metrics['correction'] / LN2 if val_metrics['correction'] > 0 else 0.0,
            'elapsed_sec': self.state.elapsed_sec,
            'timestamp': time.time(),
        }
        self.history.append(record)

        with open(self.history_path, 'a') as f:
            f.write(json.dumps(sanitize_for_json(record)) + '\n')

        self._save_state()

    def _save_state(self) -> None:
        """Save current state to JSON."""
        data = {
            'state': asdict(self.state),
            'history': self.history[-100:],
        }
        # Sanitize to handle any inf/nan values before JSON serialization
        self.log_path.write_text(json.dumps(sanitize_for_json(data), indent=2))


class MonitorGUI:
    """Tkinter GUI for training with per-position improvement display."""

    def __init__(self, log_path: str = "training_log.json"):
        self.log_path = Path(log_path)
        self.running = True

        self.root = tk.Tk()
        self.root.title("Mina Bidirectional Training Monitor")
        self.root.geometry("800x550")
        self.root.configure(bg='#1e1e1e')

        self._setup_ui()
        self._start_update_loop()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background='#1e1e1e', foreground='#ffffff', font=('Consolas', 10))
        style.configure('Header.TLabel', font=('Consolas', 14, 'bold'))
        style.configure('TFrame', background='#1e1e1e')
        style.configure('Improvement.TLabel', foreground='#4aff9e')

        main = ttk.Frame(self.root, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main, text="Mina Bidirectional Transformer", style='Header.TLabel').pack(pady=(0, 15))

        # Metrics frame
        metrics = ttk.Frame(main)
        metrics.pack(fill=tk.X)

        self.labels = {}
        metric_names = [
            ('epoch', 'Epoch'),
            ('batch', 'Batch'),
            ('elapsed', 'Elapsed'),
            ('impr_rate', 'Impr Rate'),
            ('fwd_bpc', 'FWD BPC'),
            ('bwd_bpc', 'BWD BPC'),
            ('corr_bpc', 'CORR BPC'),
            ('best', 'Best FWD BPC'),
            ('val_fwd_bpc', 'Val FWD BPC'),
            ('val_bwd_bpc', 'Val BWD BPC'),
            ('val_corr_bpc', 'Val CORR BPC'),
            ('val_impr', 'Val Impr'),
            ('best_corr', 'Best CORR BPC'),
            ('fwd_bwd_gap', 'FWD-BWD'),
        ]

        for i, (key, name) in enumerate(metric_names):
            row = i // 4
            col = i % 4

            frame = ttk.Frame(metrics)
            frame.grid(row=row, column=col, padx=8, pady=4, sticky='w')

            ttk.Label(frame, text=f"{name}:").pack(side=tk.LEFT)
            if 'impr' in key.lower():
                self.labels[key] = ttk.Label(frame, text="--", style='Improvement.TLabel')
            else:
                self.labels[key] = ttk.Label(frame, text="--")
            self.labels[key].pack(side=tk.LEFT, padx=(5, 0))

        # Progress bar
        self.progress = ttk.Progressbar(main, length=650, mode='determinate')
        self.progress.pack(pady=15)

        # Stop button
        style.configure('Stop.TButton', font=('Consolas', 11, 'bold'))
        self.stop_btn = ttk.Button(
            main,
            text="Stop & Save",
            style='Stop.TButton',
            command=self._on_stop_save
        )
        self.stop_btn.pack(pady=8)

        self.status_label = ttk.Label(main, text="", foreground='#ff9944')
        self.status_label.pack()

        # Improvement gauge
        self.gauge_frame = ttk.Frame(main)
        self.gauge_frame.pack(fill=tk.X, pady=10)

        ttk.Label(self.gauge_frame, text="Correction Wins:").pack(side=tk.LEFT)
        self.gauge_bar = tk.Canvas(self.gauge_frame, width=500, height=30, bg='#2d2d2d', highlightthickness=0)
        self.gauge_bar.pack(side=tk.LEFT, padx=10)

        # Loss comparison bar
        self.loss_frame = ttk.Frame(main)
        self.loss_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.loss_frame, text="Loss (FWD/CORR):").pack(side=tk.LEFT)
        self.loss_bar = tk.Canvas(self.loss_frame, width=500, height=25, bg='#2d2d2d', highlightthickness=0)
        self.loss_bar.pack(side=tk.LEFT, padx=10)

    def _start_update_loop(self) -> None:
        self.root.after(100, self._update_tick)

    def _update_tick(self) -> None:
        if not self.running:
            return
        self._update_display()
        self.root.after(1000, self._update_tick)

    def _update_display(self) -> None:
        if not self.log_path.exists():
            return

        try:
            data = json.loads(self.log_path.read_text())
            state = data['state']

            self.labels['epoch'].config(text=f"{state['epoch']}")
            self.labels['batch'].config(text=f"{state['batch']}/{state['total_batches']}")
            self.labels['impr_rate'].config(text=f"{state.get('improvement_rate', 0):.1%}")

            # Training BPC
            fwd_bpc = state['fwd_loss'] / LN2
            bwd_bpc = state.get('bwd_loss', 0) / LN2
            corr_bpc = state.get('corr_loss', 0) / LN2
            self.labels['fwd_bpc'].config(text=f"{fwd_bpc:.3f}")
            self.labels['bwd_bpc'].config(text=f"{bwd_bpc:.3f}")
            self.labels['corr_bpc'].config(text=f"{corr_bpc:.3f}")

            # Validation BPC
            val_fwd_bpc = state['val_fwd_loss'] / LN2
            val_bwd_bpc = state.get('val_bwd_loss', 0) / LN2
            val_corr_bpc = state.get('val_corr_loss', 0) / LN2
            self.labels['val_fwd_bpc'].config(text=f"{val_fwd_bpc:.3f}")
            self.labels['val_bwd_bpc'].config(text=f"{val_bwd_bpc:.3f}")
            self.labels['val_corr_bpc'].config(text=f"{val_corr_bpc:.3f}")
            self.labels['val_impr'].config(text=f"{state.get('val_improvement_rate', 0):.1%}")

            # Best metrics
            self.labels['best'].config(text=f"{state['best_val_loss'] / LN2:.3f}")

            # FWD-BWD gap (validation)
            gap = val_fwd_bpc - val_bwd_bpc
            self.labels['fwd_bwd_gap'].config(text=f"{gap:+.3f}")

            # Track best correction BPC from history
            history = data.get('history', [])
            if history:
                best_corr = min(h.get('val_corr_bpc', 999) for h in history)
                self.labels['best_corr'].config(text=f"{best_corr:.3f}")
            else:
                self.labels['best_corr'].config(text="--")

            elapsed = state['elapsed_sec']
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.labels['elapsed'].config(text=f"{hours}:{minutes:02d}:{seconds:02d}")

            if state['total_batches'] > 0:
                progress = (state['batch'] / state['total_batches']) * 100
                self.progress['value'] = progress

            self._draw_improvement_gauge(state.get('improvement_rate', 0))
            self._draw_loss_comparison(state['fwd_loss'], state.get('corr_loss', 0))

        except (json.JSONDecodeError, KeyError):
            pass

    def _draw_improvement_gauge(self, rate: float) -> None:
        """Draw gauge showing % of positions where correction beats forward."""
        self.gauge_bar.delete('all')

        width = 500
        height = 30
        fill_width = width * rate

        self.gauge_bar.create_rectangle(0, 0, width, height, fill='#663333', outline='')
        if fill_width > 0:
            self.gauge_bar.create_rectangle(0, 0, fill_width, height, fill='#4aff9e', outline='')

        self.gauge_bar.create_text(width / 2, height / 2, text=f"{rate:.1%}", fill='white', font=('Consolas', 10, 'bold'))

        marker_x = width * 0.5
        self.gauge_bar.create_line(marker_x, 0, marker_x, height, fill='#ffffff', width=2, dash=(4, 2))

    def _draw_loss_comparison(self, fwd: float, corr: float) -> None:
        """Draw bar comparing forward vs correction loss."""
        self.loss_bar.delete('all')

        total = fwd + corr
        if total < 1e-8:
            return

        width = 500
        height = 25
        fwd_width = width * (fwd / total)

        self.loss_bar.create_rectangle(0, 0, fwd_width, height, fill='#4a9eff', outline='')
        self.loss_bar.create_rectangle(fwd_width, 0, width, height, fill='#ff9944', outline='')

        if fwd_width > 50:
            self.loss_bar.create_text(fwd_width / 2, height / 2, text=f"FWD Loss {fwd:.3f}", fill='white', font=('Consolas', 8))
        corr_width = width - fwd_width
        if corr_width > 50:
            self.loss_bar.create_text(fwd_width + corr_width / 2, height / 2, text=f"CORR Loss {corr:.3f}", fill='white', font=('Consolas', 8))

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_stop_save(self) -> None:
        request_stop()
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Stop requested - saving checkpoint...")

    def _on_close(self) -> None:
        self.running = False
        self.root.destroy()


# Aliases for backwards compatibility with train.py imports
TrainingMonitorV4 = TrainingMonitor
MonitorGUIV4 = MonitorGUI
TrainingStateV4 = TrainingState


def main() -> None:
    parser = argparse.ArgumentParser(description='Mina training monitor GUI')
    parser.add_argument('--log', default='training_log.json', help='Log file path')
    args = parser.parse_args()

    gui = MonitorGUI(args.log)
    gui.run()


if __name__ == '__main__':
    main()
