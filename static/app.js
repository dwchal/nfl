"use strict";

const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
const pct = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const signed = value => `${value > 0 ? "+" : ""}${value}`;
const record = row => `${row.wins}–${row.losses}${row.ties ? `–${row.ties}` : ""}`;
const color = value => value > 0 ? "positive" : value < 0 ? "negative" : "";
const gameLabel = game => game.kind === "REG" ? `Week ${game.week}` : ({WC:"Wild Card", DIV:"Divisional", CON:"Conference", SB:"Super Bowl"}[game.kind] || game.kind);
const dateLabel = value => new Date(`${value}T12:00:00`).toLocaleDateString(undefined, {month:"short", day:"numeric"});
let dashboard = null;
let scheduleFilter = "all";
let sortKey = "rating";
let sortDirection = -1;
let requestNumber = 0;

async function load(refresh = false) {
  const current = ++requestNumber;
  $("refresh").disabled = true;
  $("retry").disabled = true;
  $("phase").disabled = true;
  $("season").disabled = true;
  $("team").disabled = true;
  $("model").disabled = true;
  $("loading").hidden = Boolean(dashboard);
  $("error").hidden = true;
  $("refresh").querySelector("span").textContent = "Updating…";
  const query = new URLSearchParams({phase: $("phase").value, team: $("team").value, model: $("model").value});
  if (dashboard) query.set("season", $("season").value);
  try {
    const response = await fetch(`${refresh ? "/api/refresh" : "/api/dashboard"}?${query}`, {method:refresh ? "POST" : "GET"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The schedule could not be loaded.");
    if (current !== requestNumber) return;
    if (dashboard && dashboard.team.code !== result.team.code) $("team-search").value = "";
    dashboard = result;
    $("team").value = result.team.code;
    try { localStorage.setItem("nfl-dashboard-team", result.team.code); } catch { /* Storage may be disabled. */ }
    $("season").innerHTML = result.seasons.map(s => `<option value="${s}">${s}</option>`).join("");
    $("season").value = String(result.season);
    $("phase").value = result.include_playoffs ? "all" : "regular";
    $("model").value = result.model.choice;
    try { localStorage.setItem("nfl-dashboard-model", result.model.choice); } catch { /* Storage may be disabled. */ }
    render();
    $("dashboard").hidden = false;
  } catch (error) {
    if (current !== requestNumber) return;
    // Hide an older dashboard when controls no longer describe its data.
    $("dashboard").hidden = true;
    $("notice").hidden = true;
    $("error-text").textContent = error.message;
    $("error").hidden = false;
  } finally {
    if (current === requestNumber) {
      $("loading").hidden = true;
      $("refresh").disabled = false;
      $("retry").disabled = false;
      $("phase").disabled = false;
      $("season").disabled = !dashboard;
      $("team").disabled = false;
      $("model").disabled = false;
      $("refresh").querySelector("span").textContent = "Refresh data";
    }
  }
}

function render() {
  const d = dashboard, s = d.team_stats, team = d.team;
  document.body.dataset.team = team.code;
  document.title = `${team.short_name} · Season Dashboard`;
  $("brand-team").textContent = team.short_name.toUpperCase();
  $("brand-mark").textContent = team.short_name[0];
  $("brand").setAttribute("aria-label", `${team.short_name} dashboard home`);
  $("favicon").href = team.code === "MIN" ? "/favicon-vikings.svg" : "/favicon.svg";
  $("team-tagline").textContent = team.tagline;
  $("intro-copy").textContent = `Every result. The next matchup. A clearer view of ${team.city}’s season.`;
  $("metrics").setAttribute("aria-label", `${team.short_name} season statistics`);
  $("division-heading").textContent = team.division;
  $("schedule-heading").textContent = `The ${team.short_name} schedule`;
  $("rankings-heading").textContent = `How ${team.city} stacks up`;
  $("team-footer").textContent = `Made for the ${team.code === "MIN" ? "purple" : "black"} & gold.`;
  $("season-heading").textContent = `${team.city.toUpperCase()} / ${d.season} ${d.include_playoffs ? "SEASON + PLAYOFFS" : "REGULAR SEASON"}`;
  $("form").innerHTML = d.form.length ? d.form.map(r => `<span class="form-badge ${r}">${r}</span>`).join("") : '<span class="muted">NO RESULTS YET</span>';
  $("metrics").innerHTML = [
    ["SEASON RECORD", record(s), `${s.games} games played`],
    ["WINNING PERCENTAGE", pct(s.win_pct), `${s.ties} ${s.ties === 1 ? "tie" : "ties"} · ties count as half a win`],
    ["POINT DIFFERENTIAL", signed(s.differential), `${s.pf} scored · ${s.pa} allowed`],
    ["ELO POWER RANK", `#${s.rank}`, `${s.rating.toFixed(1)} rating · of ${d.rankings.length} teams`],
  ].map(([label, value, detail]) => `<article class="metric"><span class="muted">${label}</span><div class="value">${value}</div><small>${detail}</small></article>`).join("");
  const warning = d.data.warning;
  const calendarYear = new Date().getFullYear() - (new Date().getMonth() < 2 ? 1 : 0);
  const latest = Math.max(...d.seasons);
  const availability = latest < calendarYear ? `The latest season available from the provider is ${latest}.` : "";
  $("notice").textContent = [warning, availability].filter(Boolean).join(" ");
  $("notice").hidden = !$("notice").textContent;
  $("division-phase").textContent = d.include_playoffs ? "Regular + playoffs" : "Regular season";
  renderNext();
  renderProjection();
  renderModelReport();
  renderTrend();
  renderDivision();
  renderSchedule();
  renderRankings();
  const saved = new Date(d.data.downloaded_at).toLocaleString(undefined, {month:"short", day:"numeric", hour:"numeric", minute:"2-digit"});
  $("data-status").textContent = `${d.data.source} · downloaded ${saved}${d.latest_result_date ? ` · latest result ${dateLabel(d.latest_result_date)}` : " · no completed games"}`;
}

function renderNext() {
  const g = dashboard.next_game;
  $("next-week").hidden = !g;
  if (!g) {
    $("next-game").innerHTML = `<div class="empty">No unplayed ${esc(dashboard.team.short_name)} games are listed in this view. Try another season or include playoffs.</div>`;
    return;
  }
  $("next-week").textContent = gameLabel(g);
  const overdue = g.date < new Date().toLocaleDateString("en-CA");
  $("next-game").innerHTML = `<p class="muted">${esc(dashboard.team.name.toUpperCase())} ${g.venue === "Away" ? "AT" : "VS"}</p>
    <div class="opponent">${esc(g.opponent_name)}</div>
    <p class="game-meta">${esc(dateLabel(g.date))}${g.kickoff ? ` · ${esc(g.kickoff)} ET` : " · Time TBD"} · ${esc(g.venue)}${g.stadium ? `<br>${esc(g.stadium)}` : ""}</p>
    <div class="probability"><strong>${pct(g.win_probability)}</strong><span>${esc(dashboard.team.short_name)} win probability</span></div>
    <div class="prob-track"><div class="prob-fill" id="matchup-fill"></div></div>
    <div class="matchup-detail"><span>${esc(dashboard.team.code)} ${pct(g.win_probability)}</span><span>${esc(g.opponent)} ${pct(1 - g.win_probability)}</span></div>
    <p class="caption">${esc(dashboard.model.name)} · ${g.venue === "Neutral" ? "no home-field adjustment" : "home-field advantage"}.${overdue ? " This date has passed; the provider has not supplied a final score." : ""}</p>`;
  // SVGs and width attributes avoid inline styles under the app's CSP.
  const fill = $("matchup-fill");
  fill.replaceWith(svg(`<rect width="${(g.win_probability * 100).toFixed(2)}" height="7" fill="var(--accent)"/>`, "0 0 100 7", `${dashboard.team.short_name} estimated chance of winning`, "none"));
}

function svg(content, viewBox, label, aspect = "xMidYMid meet") {
  const wrapper = document.createElement("div");
  wrapper.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" role="img" aria-label="${esc(label)}" preserveAspectRatio="${aspect}">${content}</svg>`;
  return wrapper.firstElementChild;
}

function renderProjection() {
  const p = dashboard.projection;
  $("projection").innerHTML = `<div class="outlook-head"><strong>${p.expected_wins.toFixed(1)}</strong><span>projected final wins</span></div>
    <p class="projection-sub">${p.remaining ? `${p.remaining} games left · Middle 80% of simulations: ${p.low}–${p.high} wins` : "Regular season complete · final record shown below"}</p><div class="chart" id="distribution"></div><p class="caption">${p.remaining ? "Regular season only. Future ties are not simulated." : "No remaining regular-season games to simulate."}</p>`;
  const dist = p.distribution;
  const max = Math.max(...dist.map(r => r.probability));
  const width = 460, height = 170, left = 35, usable = 410, baseline = 136;
  const spacing = usable / dist.length;
  const peak = dist.reduce((a, b) => a.probability > b.probability ? a : b).wins;
  const bars = dist.map((row, i) => {
    const x = left + i * spacing + spacing * .14;
    const h = Math.max(1, row.probability / max * 94);
    return `<rect class="bar ${row.wins === peak ? "peak" : ""}" x="${x}" y="${baseline - h}" width="${spacing * .72}" height="${h}" rx="2"><title>${record(row)}: ${pct(row.probability)}</title></rect><text x="${x + spacing * .36}" y="${baseline + 18}" text-anchor="middle">${row.wins}</text>${row.wins === peak ? `<text x="${x + spacing * .36}" y="${baseline - h - 9}" text-anchor="middle">${pct(row.probability)}</text>` : ""}`;
  }).join("");
  $("distribution").append(svg(`${bars}<text x="14" y="${baseline + 18}">W</text>`, `0 0 ${width} ${height}`, `Projected final wins distribution. Expected wins ${p.expected_wins}; middle 80 percent range ${p.low} to ${p.high}.`));
}

function renderModelReport() {
  const model = dashboard.model, report = model.evaluation;
  $("model-summary").textContent = `Active: ${model.name}. ${model.use_margin ? "Final score margins help measure team strength." : "Wins and losses determine rating changes."}`;
  if (report.status !== "evaluated") {
    $("model-report").innerHTML = `<p class="caption">${esc(report.reason)} Original Elo is active for this season.</p>`;
    return;
  }
  const baseline = report.baseline, candidate = report.challenger_metrics;
  const years = `${report.test_seasons[0]}–${report.test_seasons.at(-1)}`;
  const improvement = report.brier_improvement_pct;
  const scope = report.by_team[dashboard.team.code];
  const teamDelta = scope?.challenger && scope?.baseline ? scope.challenger.brier - scope.baseline.brier : 0;
  const teamComparison = Math.abs(teamDelta) < 1e-12 ? "the same" : teamDelta < 0 ? "better" : "worse";
  const headline = Math.abs(improvement) < 1e-10 ? "No change in league-wide probability error" : `${Math.abs(improvement).toFixed(1)}% ${improvement >= 0 ? "lower" : "higher"} league-wide probability error`;
  $("model-report").innerHTML = `<p class="model-evidence"><strong>${headline}</strong><span>${esc(years)} · ${baseline.games} regular-season games</span></p>
    <div class="table-wrap"><table><caption class="sr-only">League-wide historical prediction comparison</caption><thead><tr><th>Historical evaluation</th><th>Original Elo</th><th>${esc(report.challenger.name)}</th></tr></thead><tbody>
    <tr><td>Correct winner picks</td><td>${pct(baseline.accuracy)}</td><td>${pct(candidate.accuracy)}</td></tr>
    <tr><td>Probability error (Brier) ↓</td><td>${baseline.brier.toFixed(4)}</td><td>${candidate.brier.toFixed(4)}</td></tr>
    <tr><td>Log loss ↓</td><td>${baseline.log_loss.toFixed(4)}</td><td>${candidate.log_loss.toFixed(4)}</td></tr></tbody></table></div>
    ${scope?.baseline && scope?.challenger ? `<p class="team-evidence ${color(-teamDelta)}">${esc(dashboard.team.short_name)} subset: probability error ${scope.baseline.brier.toFixed(4)} → ${scope.challenger.brier.toFixed(4)} across ${scope.baseline.games} games. The candidate performed ${teamComparison} on this team’s sample.</p>` : ""}
    <p class="caption">Settings selected on ${report.tuning_seasons[0]}–${report.tuning_seasons.at(-1)}; later seasons used for evaluation and the default-model decision. Lower error values are better. Same settings for all teams. Ties are excluded from pick accuracy.</p>
    <p class="caption">${esc(report.reason)} These are historical results, not a guarantee of future accuracy.</p>`;
}

function renderTrend() {
  const results = dashboard.schedule.filter(g => g.result);
  if (!results.length) {
    $("trend").innerHTML = '<div class="empty">The scoring trend will appear after the first result.</div>';
    return;
  }
  const values = [0, ...results.map(g => g.cumulative)];
  const low = Math.min(...values, -10), high = Math.max(...values, 10);
  const y = value => 22 + (high - value) / (high - low) * 143;
  const x = i => 46 + i / (values.length - 1) * 450;
  const grid = [high, 0, low].map(v => `<line class="${v === 0 ? "zero" : "grid"}" x1="46" y1="${y(v)}" x2="496" y2="${y(v)}"/><text x="37" y="${y(v) + 4}" text-anchor="end">${signed(v)}</text>`).join("");
  const points = values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const dots = results.map((g, i) => `<circle class="dot" cx="${x(i + 1)}" cy="${y(g.cumulative)}" r="3.5"><title>${esc(gameLabel(g))} vs ${esc(g.opponent)}: ${signed(g.cumulative)}</title></circle>`).join("");
  $("trend").replaceChildren(svg(`${grid}<polyline class="line" points="${points}"/>${dots}<text x="46" y="192">Start</text><text x="496" y="192" text-anchor="end">${esc(gameLabel(results.at(-1)))}</text>`, "0 0 520 205", `${dashboard.team.short_name} cumulative point differential, ending at ${signed(results.at(-1).cumulative)}.`));
}

function renderDivision() {
  $("division").innerHTML = `<table><caption class="sr-only">${esc(dashboard.team.division)} comparison</caption><thead><tr><th>Team</th><th>Record</th><th>Win %</th><th>+ / −</th></tr></thead><tbody>${dashboard.division.map(r => `<tr class="${r.team === dashboard.team.code ? "selected-team" : ""}"><td class="team">${esc(r.team)}</td><td>${record(r)}</td><td>${pct(r.win_pct)}</td><td class="${color(r.differential)}">${signed(r.differential)}</td></tr>`).join("")}</tbody></table>`;
}

function renderSchedule() {
  const games = dashboard.schedule.filter(g => scheduleFilter === "all" || (scheduleFilter === "completed" ? g.result : !g.result));
  if (!games.length) {
    $("schedule").innerHTML = '<p class="empty">No games match this filter.</p>';
    return;
  }
  $("schedule").innerHTML = `<table><caption class="sr-only">${esc(dashboard.team.short_name)} schedule and results</caption><thead><tr><th>Game</th><th>Date</th><th>Opponent</th><th>Venue</th><th>Result / ${esc(dashboard.team.code)} score</th><th>+ / −</th><th>${esc(dashboard.team.code)} win chance</th></tr></thead><tbody>${games.map(g => `<tr><td>${esc(gameLabel(g))}</td><td>${esc(dateLabel(g.date))}</td><td class="team opponent-cell">${esc(g.opponent_name)}<span>${g.kickoff ? `${esc(g.kickoff)} ET` : "Time TBD"}</span></td><td>${esc(g.venue)}</td><td>${g.result ? `<span class="result-badge ${g.result}">${g.result}</span>${g.scored}–${g.allowed}` : "Upcoming"}</td><td class="${color(g.differential)}">${g.differential == null ? "—" : signed(g.differential)}</td><td>${pct(g.result ? g.pregame_probability : g.win_probability)}${g.result ? ' <span class="muted">PRE</span>' : ""}</td></tr>`).join("")}</tbody></table>`;
}

function renderRankings() {
  const term = $("team-search").value.trim().toLowerCase();
  const rows = dashboard.rankings.filter(r => `${r.name} ${r.team}`.toLowerCase().includes(term));
  rows.sort((a, b) => {
    if (sortKey === "name") return sortDirection * a.name.localeCompare(b.name);
    // Missing winning percentages remain last regardless of direction.
    if (a[sortKey] == null) return b[sortKey] == null ? 0 : 1;
    if (b[sortKey] == null) return -1;
    return sortDirection * (a[sortKey] - b[sortKey]);
  });
  const cols = [["rank", "Elo rank"], ["name", "Team"], ["wins", "W"], ["losses", "L"], ["ties", "T"], ["win_pct", "Win %"], ["pf", "PF"], ["pa", "PA"], ["differential", "+ / −"], ["rating", "Elo"]];
  $("rankings").innerHTML = `<table><caption class="sr-only">NFL team records and Elo rankings</caption><thead><tr>${cols.map(([key, label]) => `<th${sortKey === key ? ` aria-sort="${sortDirection === 1 ? "ascending" : "descending"}"` : ""}><button data-sort="${key}">${label}${sortKey === key ? (sortDirection === 1 ? " ↑" : " ↓") : ""}</button></th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr class="${r.team === dashboard.team.code ? "selected-team" : ""}"><td>${r.rank}</td><td class="team">${esc(r.name)}</td><td>${r.wins}</td><td>${r.losses}</td><td>${r.ties}</td><td>${pct(r.win_pct)}</td><td>${r.pf}</td><td>${r.pa}</td><td class="${color(r.differential)}">${signed(r.differential)}</td><td>${r.rating.toFixed(1)}</td></tr>`).join("") || '<tr><td colspan="10">No matching teams.</td></tr>'}</tbody></table>`;
}

$("refresh").addEventListener("click", () => load(true));
$("retry").addEventListener("click", () => load(true));
$("season").addEventListener("change", () => load());
$("phase").addEventListener("change", () => load());
$("team").addEventListener("change", () => load());
$("model").addEventListener("change", () => load());
$("team-search").addEventListener("input", () => dashboard && renderRankings());
document.querySelector(".segmented").addEventListener("click", event => {
  const button = event.target.closest("button[data-filter]");
  if (!button || !dashboard) return;
  scheduleFilter = button.dataset.filter;
  document.querySelectorAll("[data-filter]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
  renderSchedule();
});
$("rankings").addEventListener("click", event => {
  const button = event.target.closest("button[data-sort]");
  if (!button) return;
  sortDirection = sortKey === button.dataset.sort ? -sortDirection : (button.dataset.sort === "name" || button.dataset.sort === "rank" ? 1 : -1);
  sortKey = button.dataset.sort;
  renderRankings();
});
try {
  const savedTeam = localStorage.getItem("nfl-dashboard-team");
  if (["PIT", "MIN"].includes(savedTeam)) $("team").value = savedTeam;
  const savedModel = localStorage.getItem("nfl-dashboard-model");
  if (["auto", "baseline"].includes(savedModel)) $("model").value = savedModel;
} catch { /* The dashboard also works without browser storage. */ }
load();
