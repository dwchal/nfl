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
let scenarioQB = "", scenarioOpponentQB = "";

async function load(refresh = false) {
  if (refresh) { scenarioQB = ""; scenarioOpponentQB = ""; }
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
  if (scenarioQB) query.set("qb", scenarioQB);
  if (scenarioOpponentQB) query.set("opponent_qb", scenarioOpponentQB);
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
  renderMatchup();
  renderProjection();
  renderModelReport();
  renderMatchupReport();
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

function renderMatchup() {
  const m = dashboard.matchup, weather = dashboard.weather;
  if (!m) {
    $("matchup-lab").innerHTML = `<p class="caption">${dashboard.next_game ? "QB scenarios need enough historical player and team statistics. Refresh data when online to load them." : "Choose a season with an upcoming matchup to explore quarterback scenarios."}</p>`;
  } else {
    const options = (rows, recent, selected) => `<option value="">${esc(recent?.status || "Recent passer")}: ${esc(recent?.name || "Unknown")}</option>${rows.map(q => `<option value="${esc(q.id)}"${q.id === selected ? " selected" : ""}>${esc(q.name)}${q.prior_dropbacks < 100 ? " · limited history" : ""}</option>`).join("")}`;
    const delta = 100 * (m.probability - m.default_probability);
    const scenario = Boolean(m.selected_qb || m.opponent_qb);
    const contributions = [...m.contributions].sort((a,b) => Math.abs(b.change) - Math.abs(a.change));
    $("matchup-lab").innerHTML = `<div class="scenario-grid"><div><h3>Who plays quarterback?</h3><div class="scenario-controls"><label>${esc(dashboard.team.short_name)} QB<select id="scenario-qb">${options(m.team_qbs, m.assumed_qbs[dashboard.team.code], m.selected_qb)}</select></label><label>${esc(dashboard.next_game.opponent_name)} QB<select id="scenario-opponent-qb">${options(m.opponent_qbs, m.assumed_qbs[dashboard.next_game.opponent], m.opponent_qb)}</select></label></div>
      <div class="scenario-result"><strong id="scenario-probability">${pct(m.probability)}</strong><span>${esc(dashboard.team.short_name)} chance · experimental matchup model</span></div><p class="caption">${scenario ? `${signed(Number(delta.toFixed(1)))} percentage points versus recent-passer assumptions. This scenario does not alter the schedule, season outlook, or saved forecasts.` : `Compare quarterbacks to explore a possible lineup change. Elo alone: ${pct(m.elo_probability)}.`}</p><p class="caption">${esc(m.note)}</p></div>
      <div><h3>What changes the estimate?</h3><div class="table-wrap"><table><caption class="sr-only">Sequential contributions to the experimental matchup forecast</caption><thead><tr><th>Signal</th><th>Probability change</th></tr></thead><tbody>${contributions.map(c => `<tr><td>${esc(c.label)}</td><td class="${color(c.change)}">${signed(Number((100*c.change).toFixed(1)))} pp</td></tr>`).join("")}</tbody></table></div><p class="caption">Contributions add from Elo to the recent-passer forecast in a fixed order. They describe the model, not proven causes. QB selections above are shown separately.</p></div></div>
      <details class="injury-details"><summary>Reported availability · ${m.injuries.length} entries for this matchup</summary>${m.injuries.length ? `<div class="table-wrap"><table><thead><tr><th>Team</th><th>Player</th><th>Status</th><th>Injury</th></tr></thead><tbody>${m.injuries.map(r => `<tr><td>${esc(r.team)}</td><td>${esc(r.name)} · ${esc(r.position)}</td><td>${esc(r.status)}</td><td>${esc(r.injury)}</td></tr>`).join("")}</tbody></table></div>` : '<p class="caption">No reports for this game week are available in the saved feed. That does not establish that every player is healthy.</p>'}</details>`;
  }
  renderAdvancedContext(m);
  $("game-weather").innerHTML = weather ? `<div class="weather-panel"><h3>Kickoff conditions</h3>${weather.status === "forecast" ? `<p>${weather.temperature_f}°F · wind up to ${weather.wind_mph} mph · gusts ${weather.gust_mph} mph · ${weather.precipitation_inches}″ precipitation</p><p class="caption">Three-hour game window. Forecast saved ${esc(new Date(weather.retrieved_at).toLocaleString())}. ${esc(weather.warning || "")}</p>` : `<p class="caption">${esc(weather.reason)}</p>`}<p class="caption">${esc(weather.note)} Weather by <a href="https://open-meteo.com/" target="_blank" rel="noreferrer">Open-Meteo</a>.</p></div>` : "";
}

function renderAdvancedContext(m) {
  const root = $("advanced-context");
  if (!m?.advanced) { root.innerHTML = ""; return; }
  const c = m.advanced_context, support = dashboard.model.matchup_evaluation.support;
  const teams = [dashboard.team.code, dashboard.next_game.opponent];
  root.innerHTML = `<h3>Pregame evidence</h3><div class="table-wrap"><table><thead><tr><th>Team</th><th>Quarterback</th><th>Evidence</th><th>Team-local kickoff</th><th>Travel</th></tr></thead><tbody>${teams.map(t => {
    const q = c.quarterbacks[t], travel = c.travel.teams[t];
    return `<tr><td>${esc(t)}</td><td>${esc(q.name)}</td><td>${esc(q.status)} · ${esc(q.source)}${q.available_at ? `<br>${esc(new Date(q.available_at).toLocaleString())}` : ""}</td><td>${esc(travel.body_clock || "Unknown")}<br>${esc(travel.time_zone || "")}</td><td>${travel.miles == null ? "Unknown" : `${travel.miles} mi`} · ${travel.away_streak} consecutive away</td></tr>`;
  }).join("")}</tbody></table></div><p class="caption">${esc(c.travel.note)}</p><p class="caption">Weather interactions: ${support.weather_enabled ? "trained" : "collecting history"} (${support.weather_games} earlier forecasts). Availability adjustments: ${support.availability_enabled ? "trained" : "collecting history"} (${support.availability_games} earlier snapshots). ${c.weather ? "A pre-kickoff weather forecast is available for this estimate." : "No eligible outdoor weather snapshot is available for this estimate."}</p>
  <details><summary>Record a confirmed starting quarterback</summary><p class="caption">Choose a quarterback in the scenario controls, then record the announcement source below. Saving applies to the advanced forecast and its archive; a scenario selection alone does not.</p><label>Confirmation source<input id="lineup-source" type="text" maxlength="500" placeholder="Team announcement or source URL"></label><div class="scenario-controls"><button data-confirm-lineup="${esc(teams[0])}" ${m.selected_qb ? "" : "disabled"}>Confirm ${esc(teams[0])} selection</button><button data-confirm-lineup="${esc(teams[1])}" ${m.opponent_qb ? "" : "disabled"}>Confirm ${esc(teams[1])} selection</button></div><p id="lineup-message" class="caption" aria-live="polite"></p></details>`;
}

function renderMatchupReport() {
  const r = dashboard.model.matchup_evaluation;
  const warnings = dashboard.features?.sources.filter(s => s.warning) || [];
  if (!r || r.status !== "evaluated") {
    $("matchup-report").innerHTML = `<p class="caption">${esc(r?.reason || "Extra player and team data are unavailable in this data-file mode.")}</p>`;
  } else {
    const b = r.baseline, c = r.challenger_metrics, scope = r.by_team[dashboard.team.code];
    const advanced = r.version === "advanced-v2";
    $("matchup-report").innerHTML = `<h3>Does the matchup model help?</h3><p class="caption">${advanced ? `Each comparison season refits using earlier years only. Comparison: ${r.test_seasons[0]}–${r.test_seasons.at(-1)}. Current coefficients use ${r.tuning_seasons[0]}–${r.tuning_seasons.at(-1)}.` : `Frozen corrections trained on ${r.tuning_seasons[0]}–${r.tuning_seasons.at(-1)}, compared on ${r.test_seasons[0]}–${r.test_seasons.at(-1)}.`}</p><div class="table-wrap"><table><thead><tr><th>${b.games} games</th><th>Selected Elo</th><th>${advanced ? "Advanced matchup" : "QB-aware matchup"}</th></tr></thead><tbody><tr><td>Correct winner picks</td><td>${pct(b.accuracy)}</td><td>${pct(c.accuracy)}</td></tr><tr><td>Brier error ↓</td><td>${b.brier.toFixed(4)}</td><td>${c.brier.toFixed(4)}</td></tr><tr><td>Log loss ↓</td><td>${b.log_loss.toFixed(4)}</td><td>${c.log_loss.toFixed(4)}</td></tr></tbody></table></div><p class="caption">${esc(r.reason)}</p>${scope?.baseline && scope?.challenger ? `<p class="caption">${esc(dashboard.team.short_name)} Brier error: ${scope.baseline.brier.toFixed(4)} → ${scope.challenger.brier.toFixed(4)} across ${scope.baseline.games} games.</p>` : ""}<details><summary>${advanced ? "Feature comparisons on later seasons" : "Feature checks on the older validation season"}</summary><div class="table-wrap"><table><thead><tr><th>Included signals</th><th>Brier error ↓</th><th>Log loss ↓</th></tr></thead><tbody>${r.ablations.map(a=>`<tr><td>${esc(a.name)}</td><td>${(a.validation || a.metrics).brier.toFixed(4)}</td><td>${(a.validation || a.metrics).log_loss.toFixed(4)}</td></tr>`).join("")}</tbody></table></div><p class="caption">${advanced ? "Each group refits on earlier seasons before each comparison year. These diagnostics do not select the final model." : "Each smaller model trains on the first two tuning seasons and validates on the third. These checks do not choose the final challenger."}</p></details>`;
  }
  if (warnings.length) $("matchup-report").innerHTML += `<p class="caption">Player/team feeds: ${warnings.length} saved or unavailable files. Forecasts may use older player information. Refresh while online to update.</p>`;
  const forward = dashboard.forward_evaluation;
  $("forward-report").innerHTML = forward ? `<h3>Saved before kickoff</h3><p class="caption">${forward.snapshots} snapshots covering ${forward.games} games for this team and model choice. ${forward.evaluated ? `${forward.evaluated.games} completed games · Brier ${forward.evaluated.brier.toFixed(4)} · log loss ${forward.evaluated.log_loss.toFixed(4)}.` : "No completed games to evaluate yet."}</p><p class="caption">${esc(forward.policy)}</p>` : "";
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
  if (model.matchup_active && model.matchup_evaluation?.status === "evaluated") {
    const evidence = model.matchup_evaluation;
    $("model-report").innerHTML = `<p class="model-evidence"><strong>${pct(evidence.challenger_metrics.accuracy)} correct winner picks</strong><span>${evidence.test_seasons[0]}–${evidence.test_seasons.at(-1)} · ${evidence.baseline.games} regular-season games</span></p><p class="caption">Compared with ${pct(evidence.baseline.accuracy)} for the Elo baseline on the same games. Historical development results; prospective accuracy remains to be measured.</p>`;
    return;
  }
  if (report.status !== "evaluated") {
    $("model-report").innerHTML = `<p class="caption">${esc(report.reason)} Original Elo is active for this season.</p>`;
    return;
  }
  const baseline = report.baseline, candidate = report.active_metrics;
  const years = `${report.test_seasons[0]}–${report.test_seasons.at(-1)}`;
  const improvement = 100 * (baseline.brier - candidate.brier) / baseline.brier;
  const scope = report.by_team[dashboard.team.code];
  const teamDelta = scope?.active && scope?.baseline ? scope.active.brier - scope.baseline.brier : 0;
  const teamComparison = Math.abs(teamDelta) < 1e-12 ? "the same" : teamDelta < 0 ? "better" : "worse";
  const headline = Math.abs(improvement) < 1e-10 ? "No change in league-wide probability error" : `${Math.abs(improvement).toFixed(1)}% ${improvement >= 0 ? "lower" : "higher"} league-wide probability error`;
  $("model-report").innerHTML = `<p class="model-evidence"><strong>${headline}</strong><span>${esc(years)} · ${baseline.games} regular-season games</span></p>
    <div class="table-wrap"><table><caption class="sr-only">League-wide historical prediction comparison</caption><thead><tr><th>Historical evaluation</th><th>Original Elo</th><th>${esc(report.active.name)}</th></tr></thead><tbody>
    <tr><td>Correct winner picks</td><td>${pct(baseline.accuracy)}</td><td>${pct(candidate.accuracy)}</td></tr>
    <tr><td>Probability error (Brier) ↓</td><td>${baseline.brier.toFixed(4)}</td><td>${candidate.brier.toFixed(4)}</td></tr>
    <tr><td>Log loss ↓</td><td>${baseline.log_loss.toFixed(4)}</td><td>${candidate.log_loss.toFixed(4)}</td></tr></tbody></table></div>
    ${scope?.baseline && scope?.active ? `<p class="team-evidence ${color(-teamDelta)}">${esc(dashboard.team.short_name)} subset: probability error ${scope.baseline.brier.toFixed(4)} → ${scope.active.brier.toFixed(4)} across ${scope.baseline.games} games. The selected default performed ${teamComparison} on this team’s sample.</p>` : ""}
    ${calibrationEvidence(report.calibration)}
    <p class="caption">Settings selected on ${report.tuning_seasons[0]}–${report.tuning_seasons.at(-1)}; later seasons used for evaluation and the default-model decision. Lower error values are better. Same settings for all teams. Ties are excluded from pick accuracy.</p>
    <p class="caption">${esc(report.reason)} These are historical results, not a guarantee of future accuracy.</p>`;
}

function calibrationEvidence(report) {
  if (!report) return "";
  const b = report.baseline, c = report.challenger_metrics;
  return `<h3>Confidence and rest adjustment</h3><p class="caption">Compared with the previous default: Brier ${b.brier.toFixed(5)} → ${c.brier.toFixed(5)}; log loss ${b.log_loss.toFixed(5)} → ${c.log_loss.toFixed(5)}. ${esc(report.reason)}</p><p class="caption">Learned from older seasons, with missing rest treated as unknown. Small historical gains may be noise; these comparison seasons also inform the default-model decision.</p>`;
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
$("season").addEventListener("change", () => { scenarioQB = ""; scenarioOpponentQB = ""; load(); });
$("phase").addEventListener("change", () => load());
$("team").addEventListener("change", () => { scenarioQB = ""; scenarioOpponentQB = ""; load(); });
$("model").addEventListener("change", () => load());
$("matchup-lab").addEventListener("change", event => {
  if (event.target.id === "scenario-qb") scenarioQB = event.target.value;
  else if (event.target.id === "scenario-opponent-qb") scenarioOpponentQB = event.target.value;
  else return;
  load();
});
$("advanced-context").addEventListener("click", async event => {
  const button = event.target.closest("button[data-confirm-lineup]");
  if (!button || !dashboard?.matchup?.advanced) return;
  const team = button.dataset.confirmLineup;
  const player = team === dashboard.team.code ? dashboard.matchup.selected_qb : dashboard.matchup.opponent_qb;
  const source = $("lineup-source").value.trim();
  if (!source) { $("lineup-message").textContent = "Enter the confirmation source."; return; }
  button.disabled = true;
  try {
    const response = await fetch("/api/lineup", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({game_id: dashboard.next_game.id, team, player_id: player, source})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not save the lineup.");
    scenarioQB = ""; scenarioOpponentQB = "";
    await load();
  } catch (error) {
    $("lineup-message").textContent = error.message;
    button.disabled = false;
  }
});
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
  if (["auto", "baseline", "elo", "matchup", "advanced"].includes(savedModel)) $("model").value = savedModel;
} catch { /* The dashboard also works without browser storage. */ }
load();
