"""Entry point for the Historic Map Pipeline GUI application."""

from gui import PipelineGUI


def main() -> None:
    app = PipelineGUI()
    app.run()


if __name__ == "__main__":
    main()

