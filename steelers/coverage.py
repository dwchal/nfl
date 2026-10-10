"""Shared historical dataset checks, separate from pregame state coverage."""

MIN_GAMES = 200
MIN_COVERAGE = .95


def training_coverage(rows, years, require_pbp=False):
    coverage = {}
    for year in years:
        selected = [r for r in rows if r["season"] == year and r["kind"] == "REG"]
        counts = {"games": len(selected), "covered": sum(r["covered"] for r in selected),
                  "pregame_weekly": 0, "pregame_qb": 0}
        if require_pbp:
            counts.update(pbp=sum(r.get("pbp_covered", False) for r in selected), pregame_pbp=0)
        for row in selected:
            teams = list(row.get("input_coverage", {}).get("teams", {}).values())
            counts["pregame_weekly"] += len(teams) == 2 and all(t["weekly"]["games"] > 0 for t in teams)
            counts["pregame_qb"] += len(teams) == 2 and all(t["quarterback"]["dropbacks"] > 0 for t in teams)
            if require_pbp:
                counts["pregame_pbp"] += len(teams) == 2 and all(t["pbp"]["games"] > 0 for t in teams)
        coverage[str(year)] = counts
    return coverage


def sufficient_history(coverage, require_pbp=False):
    return bool(coverage) and all(
        c["games"] >= MIN_GAMES and c["covered"] / c["games"] >= MIN_COVERAGE
        and (not require_pbp or c["pbp"] / c["games"] >= MIN_COVERAGE)
        for c in coverage.values())
