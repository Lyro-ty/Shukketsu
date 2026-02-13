"""Tests for WCL CLI argument parsing."""

from code.shukketsu.apis.wcl.ingest import _build_parser


class TestWCLCLI:
    """Tests for _build_parser argument parsing."""

    def test_sync_characters_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--sync-characters"])
        assert args.sync_characters is True

    def test_rankings_with_zone(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--rankings", "--zone", "1052"])
        assert args.rankings is True
        assert args.zone == 1052

    def test_report_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--report", "TNtKz3G1H9kVAQr4"])
        assert args.report == "TNtKz3G1H9kVAQr4"

    def test_full_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--full"])
        assert args.full is True

    def test_rate_limit_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--rate-limit"])
        assert args.rate_limit is True

    def test_stats_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--stats"])
        assert args.stats is True

    def test_dry_run_combined(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--dry-run", "--full"])
        assert args.dry_run is True
        assert args.full is True
