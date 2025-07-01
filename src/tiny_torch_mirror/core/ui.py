from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import Button, DataTable, Footer, Header, Static


class PackageViewerApp(App):
    """TUI application for viewing and updating package mirrors."""

    CSS = """
    Screen {
        background: $surface;
    }

    #package-info {
        height: 3;
        background: $primary;
        color: $text;
        padding: 1;
        text-align: center;
        text-style: bold;
    }

    DataTable {
        height: 1fr;
    }

    DataTable:focus {
        border: tall $accent;
    }

    #legend {
        height: 5;
        padding: 1;
        background: $panel;
    }

    #summary {
        height: 3;
        padding: 1;
        background: $panel;
        color: $warning;
        text-style: bold;
    }

    #buttons {
        height: 5;
        padding: 1;
        align: center middle;
    }

    Button {
        margin: 0 1;
        min-width: 24;
        height: 3;
    }

    #update-btn {
        background: $primary;
        color: $text;
    }

    #cancel-btn {
        background: $error;
        color: $text;
    }
    """

    BINDINGS = [
        Binding("p", "prev_package", "Previous Package", key_display="p"),
        Binding("n", "next_package", "Next Package", key_display="n"),
        Binding("j", "prev_package", "Previous Package", show=False),  # vim-style
        Binding("k", "next_package", "Next Package", show=False),  # vim-style
        Binding("u", "update_all", "Update All"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, packages, available_wheels):
        super().__init__()
        self.packages = packages
        self.available_wheels = available_wheels
        self.all_to_be_updated = set()
        self.current_package_idx = 0
        self.package_names = sorted(packages.keys())
        self.confirmed = False
        self._scan_all_packages()

    def _scan_all_packages(self):
        """Scan all packages to find wheels that need updating."""
        self.all_to_be_updated.clear()

        for package_name in self.packages:
            for variant in self.packages[package_name]:
                for version in self.packages[package_name][variant]:
                    info = self.packages[package_name][variant][version]
                    if info["available"] and not info["installed"]:
                        self.all_to_be_updated.add(info["wheel_name"])

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="package-info")
        yield DataTable(cursor_type="row", zebra_stripes=True)
        yield Container(
            Static(
                "[green]✓✓[/green] = Available & Installed    "
                "[yellow]✓[/yellow] = Available (not installed)    "
                "[blue]✓[/blue] = Installed (not in index)    "
                "[dim]-[/dim] = Not available",
                id="legend",
            ),
            Static("", id="summary"),
            Horizontal(
                Button("Update All", variant="primary", id="update-btn"),
                Button("Cancel", variant="error", id="cancel-btn"),
                id="buttons",
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        # Set button labels explicitly after mount
        self.query_one("#update-btn", Button).label = "Update All"
        self.query_one("#cancel-btn", Button).label = "Cancel"

        self.load_package_data()
        self.update_summary()

    def load_package_data(self) -> None:
        """Load data for current package into the table."""
        table = self.query_one(DataTable)
        table.clear(columns=True)

        if not self.package_names:
            return

        package_name = self.package_names[self.current_package_idx]

        info_widget = self.query_one("#package-info", Static)
        info_widget.update(
            f"Package: {package_name} [{self.current_package_idx + 1}/{len(self.package_names)}] "
            f"(press [bold]p[/bold]/[bold]n[/bold] to navigate to previous/next package)"
        )

        # Get all versions and variants
        all_versions = set()
        all_variants = sorted(self.packages[package_name].keys())

        for variant in all_variants:
            all_versions.update(self.packages[package_name][variant].keys())

        all_versions = sorted(all_versions, reverse=True)

        # Add columns
        table.add_column("Version", width=20)

        # Add variant columns with better width management
        col_width = max(12, min(20, 100 // len(all_variants))) if all_variants else 12

        for idx, variant in enumerate(all_variants):
            table.add_column(variant, width=col_width)

        # Add rows
        for version in all_versions:
            row_data = [version]

            for variant in all_variants:
                if version in self.packages[package_name][variant]:
                    info = self.packages[package_name][variant][version]
                    if info["available"] and info["installed"]:
                        row_data.append("[green]✓✓[/green]")
                    elif info["available"]:
                        row_data.append("[yellow]✓[/yellow]")
                    elif info["installed"]:
                        row_data.append("[blue]✓[/blue]")
                else:
                    row_data.append("[dim]-[/dim]")

            table.add_row(*row_data)

    def update_summary(self) -> None:
        """Update the summary display."""
        summary = self.query_one("#summary", Static)
        summary.update(f"Total wheels to download: {len(self.all_to_be_updated)}")

        # Update button state and label - ensure text is set
        update_btn = self.query_one("#update-btn", Button)
        update_btn.disabled = len(self.all_to_be_updated) == 0

        if len(self.all_to_be_updated) == 0:
            update_btn.label = "✓ Up to date"
        else:
            update_btn.label = f"Update {len(self.all_to_be_updated)} wheels"

    def action_next_package(self) -> None:
        """Navigate to next package."""
        if self.current_package_idx < len(self.package_names) - 1:
            self.current_package_idx += 1
            self.load_package_data()
            self.notify(
                f"Package {self.current_package_idx + 1}/{len(self.package_names)}"
            )

    def action_prev_package(self) -> None:
        """Navigate to previous package."""
        if self.current_package_idx > 0:
            self.current_package_idx -= 1
            self.load_package_data()
            self.notify(
                f"Package {self.current_package_idx + 1}/{len(self.package_names)}"
            )

    def action_update_all(self) -> None:
        """Trigger update for all packages."""
        if self.all_to_be_updated:
            self.confirmed = True
            self.exit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id
        if button_id == "update-btn" and self.all_to_be_updated:
            self.confirmed = True
            self.exit()
        elif button_id == "cancel-btn":
            self.exit()
