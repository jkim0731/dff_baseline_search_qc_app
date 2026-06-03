"""Entry point for the production DFF QC app (dff-qc-production)."""

from __future__ import annotations

import sys


def main():
    from PyQt5.QtWidgets import QApplication
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from .app import MainWindow
    from .curation import DEFAULT_PATH
    from .data import DATA_DIR

    win = MainWindow(data_dir=DATA_DIR, curation_path=DEFAULT_PATH)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
