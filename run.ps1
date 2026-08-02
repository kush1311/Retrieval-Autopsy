<#
.SYNOPSIS
  Windows task runner — the same targets as the Makefile, without needing make.

.DESCRIPTION
  The Makefile is what CI and Unix use. `make` is not installed by default on
  Windows, so this mirrors it. Both call the same CLI underneath; neither has logic
  of its own, so they cannot drift apart in behaviour.

  Every target sets PYTHONPATH=. because the packages are not installed into
  site-packages during development.

.EXAMPLE
  .\run.ps1 help
  .\run.ps1 query "what does KLV-4021 mean"
  .\run.ps1 demo-query          # the baseline-vs-ablated comparison
  .\run.ps1 test
#>

[CmdletBinding()]
param(
  [Parameter(Position = 0)][string]$Target = "help",
  [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot

# Load .env into the process environment.
#
# The Python code deliberately does not auto-load a dotenv file — a library quietly
# switching to a paid provider because a file happened to be on disk is a bill waiting
# to happen. A task runner is different: it is an explicit entry point you invoked, and
# it is exactly where environment setup belongs. `make` does the same thing.
#
# Without this, `.\run.ps1 api` silently ran the offline simulator against an index built
# with real embeddings, and the mismatch surfaced as a per-chunk error mid-query.
# Overridden tracks names where a pre-existing shell value beat .env, so the banner below
# can say so. Not cosmetic: a stale `$env:AUTOPSY_PROVIDER` from an earlier command in the
# same terminal outlives every edit to .env, and the launch line read "provider=groq" while
# .env plainly said offline. The value was right and the *source* was invisible.
$Overridden = @{}
if (Test-Path (Join-Path $PSScriptRoot ".env")) {
  Get-Content (Join-Path $PSScriptRoot ".env") | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
      $name = $Matches[1]
      $value = $Matches[2].Trim().Trim('"').Trim("'")
      # Do not clobber a value already set in the shell. Standard dotenv semantics, and
      # load-bearing here: overriding meant `$env:AUTOPSY_VECTOR_BACKEND="local"` before
      # calling this script was silently reverted to `qdrant` from .env, so the documented
      # way to dodge the embedded-Qdrant single-process lock did not work.
      $existing = [Environment]::GetEnvironmentVariable($name)
      if ($name -ne "PYTHONPATH" -and [string]::IsNullOrEmpty($existing)) {
        Set-Item -Path "env:$name" -Value $value
      } elseif ($name -ne "PYTHONPATH" -and $existing -ne $value) {
        $Overridden[$name] = @{ shell = $existing; file = $value }
      }
    }
  }
}

# Echo what actually resolved. A provider surprise should be visible at launch, not
# inferred from a stack trace.
function Show-Resolved {
  $p = if ($env:AUTOPSY_PROVIDER) { $env:AUTOPSY_PROVIDER } else { "offline (default)" }
  $e = if ($env:AUTOPSY_EMBEDDER) { $env:AUTOPSY_EMBEDDER } else { "follows provider" }
  $v = if ($env:AUTOPSY_VECTOR_BACKEND) { $env:AUTOPSY_VECTOR_BACKEND } else { "local (default)" }
  Write-Host "  provider=$p  embedder=$e  store=$v" -ForegroundColor DarkGray

  # Name every value that came from the shell instead of .env. Editing .env and seeing the
  # old value survive is otherwise indistinguishable from the edit not having saved.
  foreach ($k in $Overridden.Keys) {
    $o = $Overridden[$k]
    Write-Host "  $k=$($o.shell) came from your shell, NOT .env (which says '$($o.file)')" -ForegroundColor Yellow
    Write-Host "    to use .env instead:  Remove-Item env:$k" -ForegroundColor DarkGray
  }

  # A provider that cannot authenticate should fail here, not four stages into the first
  # question. `groq` with an empty key reaches the network before it complains.
  $needsKey = @{ groq = "GROQ_API_KEY"; openai = "OPENAI_API_KEY" }
  $prov = "$env:AUTOPSY_PROVIDER".Trim().ToLower()
  if ($needsKey.ContainsKey($prov)) {
    $keyName = $needsKey[$prov]
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($keyName))) {
      Write-Host "`nprovider='$prov' needs $keyName, which is empty." -ForegroundColor Red
      Write-Host "The keys previously in .env were removed: they had been shared outside this" -ForegroundColor Yellow
      Write-Host "machine and must be treated as compromised. Rotate before reusing this provider." -ForegroundColor Yellow
      Write-Host "  groq   : https://console.groq.com/keys" -ForegroundColor DarkGray
      Write-Host "  openai : https://platform.openai.com/api-keys" -ForegroundColor DarkGray
      Write-Host "`nOr run free and keyless - real embeddings, simulated answers:" -ForegroundColor Cyan
      Write-Host "  Remove-Item env:AUTOPSY_PROVIDER   # let .env's 'offline' apply" -ForegroundColor Cyan
      exit 1
    }
  }
}

$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  Write-Host "No virtualenv found. Run:  .\run.ps1 install" -ForegroundColor Yellow
  if ($Target -ne "install") { exit 1 }
  $Py = "python"
}

function Invoke-Step([string]$Label, [scriptblock]$Body) {
  Write-Host "`n$Label" -ForegroundColor Cyan
  & $Body
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {

  "install" {
    Write-Host "creating .venv and installing" -ForegroundColor Cyan
    python -m venv .venv
    & (Join-Path $PSScriptRoot ".venv\Scripts\python.exe") -m pip install -U pip
    & (Join-Path $PSScriptRoot ".venv\Scripts\python.exe") -m pip install -e ".[dev,llm]"
  }

  "ingest"    { Invoke-Step "building the index" { & $Py -m autopsy.cli ingest } }
  "test"      { Invoke-Step "tests"              { & $Py -m pytest tests -q } }
  "eval"      { Invoke-Step "eval suites"        { & $Py -m autopsy.cli eval } }
  "ablate"    { Invoke-Step "ablation sweep"     { & $Py -m autopsy.cli ablate -n 220 } }
  "calibrate" { Invoke-Step "judge calibration"  { & $Py -m autopsy.cli calibrate --derive -n 120 } }

  # Free: retrieval-only, no model calls. Derives gate.threshold and gate.reads from the
  # corpus instead of inheriting a number tuned for a different embedding model.
  "calibrate-gate" { Invoke-Step "gate calibration" { & $Py -m autopsy.cli calibrate-gate -n 300 } }

  # Free: the context-width sensitivity curve. An ablation cannot change the answer unless
  # it changes what reaches the generator, and comparing context sets costs no tokens.
  "sensitivity" { Invoke-Step "context sensitivity" { & $Py -u scripts\sensitivity_sweep.py 40 } }

  # Everything that needs no model calls at all. Safe to run any time, any quota.
  "free" {
    Invoke-Step "1/3 tests"              { & $Py -m pytest tests -q }
    Invoke-Step "2/3 gate calibration"   { & $Py -m autopsy.cli calibrate-gate -n 300 }
    Invoke-Step "3/3 context sensitivity" { & $Py -u scripts\sensitivity_sweep.py 40 }
  }

  # The report refresh: everything whose committed numbers are stale. Costs model calls.
  "refresh" {
    Invoke-Step "1/3 eval suites"       { & $Py -m autopsy.cli eval }
    Invoke-Step "2/3 judge calibration" { & $Py -m autopsy.cli calibrate --derive -n 120 }
    Invoke-Step "3/3 ablation study"    { & $Py -m autopsy.cli ablate --core -n 25 }
  }
  "schema"    { Invoke-Step "regenerating types" { & $Py -m autopsy.cli schema } }
  "demo-traces" { Invoke-Step "freezing traces"  { & $Py -m autopsy.cli demo } }

  "eval-baseline"   { Invoke-Step "recording baseline" { & $Py -m autopsy.cli eval --update-baseline } }
  "ablate-snapshot" { Invoke-Step "pinning snapshot"   { & $Py -m autopsy.cli ablate -n 220 --snapshot } }

  "query" {
    if (-not $Rest) { Write-Host 'usage: .\run.ps1 query "your question" [-- --tenant tenant_atlas]' -ForegroundColor Yellow; exit 1 }
    & $Py -m autopsy.cli query @Rest
  }

  "demo-query" {
    # The three-step comparison from the README, run back to back.
    $q = "what does KLV-4021 mean"
    Write-Host "`n=== baseline =========================================" -ForegroundColor Cyan
    & $Py -m autopsy.cli query $q --tenant tenant_kelvin
    Write-Host "`n=== no_lexical (wrong, but flagged) ==================" -ForegroundColor Yellow
    & $Py -m autopsy.cli query $q --tenant tenant_kelvin --ablation no_lexical
    Write-Host "`n=== no_lexical + no_discriminator_guard (silent) =====" -ForegroundColor Red
    & $Py -m autopsy.cli query $q --tenant tenant_kelvin --ablation no_lexical --ablation no_discriminator_guard
  }

  "api" {
    Write-Host "inspector on http://localhost:8000  (API docs at /docs)" -ForegroundColor Cyan
    Show-Resolved

    # Clear an orphaned worker before binding. `--reload` runs the app in a multiprocessing
    # child; if the reloader parent dies badly the child survives, keeps the *inherited*
    # listening socket, and keeps the exclusive Qdrant lock. It is close to invisible:
    # netstat attributes the socket to the dead parent PID, so `taskkill` on that PID says
    # "process not found" while the port still answers 200 and every new start fails on the
    # lock. Found by walking Win32_Process for a child whose ParentProcessId is gone.
    $stale = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
      $_.CommandLine -match 'multiprocessing-fork' -and $_.CommandLine -match 'parent_pid=(\d+)' -and
      -not (Get-Process -Id ([int]$Matches[1]) -ErrorAction SilentlyContinue)
    }
    foreach ($s in $stale) {
      Write-Host "clearing orphaned worker PID $($s.ProcessId) (its reloader parent is gone)" -ForegroundColor Yellow
      Stop-Process -Id $s.ProcessId -Force -ErrorAction SilentlyContinue
    }
    if ($stale) { Start-Sleep -Seconds 2 }
    # Watch only the source directories, and only Python. The default watches the whole
    # repo, so editing run.ps1, a report, or .env restarted the server — and with embedded
    # Qdrant every restart races the outgoing worker for an exclusive file lock. That is
    # how this died once already: exit 255, clean log, no traceback.
    #
    # inspector.html is deliberately not watched. It is re-read from disk on every request,
    # so a refresh in the browser already picks up edits without a restart.
    & $Py -m uvicorn api.main:app --port 8000 --reload `
        --reload-dir autopsy --reload-dir api --reload-include "*.py"
  }

  "web" {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
      Write-Host "npm not found. The inspector UI needs Node 20+: https://nodejs.org" -ForegroundColor Yellow
      Write-Host "Everything else in this project runs without it." -ForegroundColor Yellow
      exit 1
    }
    Push-Location web
    try { npm install; npm run dev } finally { Pop-Location }
  }

  "reports" {
    Invoke-Step "eval"      { & $Py -m autopsy.cli eval }
    Invoke-Step "ablate"    { & $Py -m autopsy.cli ablate -n 220 }
    Invoke-Step "calibrate" { & $Py -m autopsy.cli calibrate --derive -n 120 }
  }

  "all" {
    Invoke-Step "1/4 ingest" { & $Py -m autopsy.cli ingest }
    Invoke-Step "2/4 test"   { & $Py -m pytest tests -q }
    Invoke-Step "3/4 eval"   { & $Py -m autopsy.cli eval }
    Invoke-Step "4/4 ablate" { & $Py -m autopsy.cli ablate -n 220 }
    Write-Host "`nreports/ is up to date" -ForegroundColor Green
  }

  default {
    Write-Host @"

Retrieval Autopsy — Windows task runner

  SETUP
    .\run.ps1 install          create .venv and install everything
    .\run.ps1 ingest           generate the corpus and build the index (idempotent)

  SEE IT WORK
    .\run.ps1 demo-query       baseline vs ablated, side by side  <- start here
    .\run.ps1 query "..."      one query, full trace
    .\run.ps1 api              inspector UI + API on http://localhost:8000

  COSTS NOTHING (no model calls, run at any quota)
    .\run.ps1 test             133 tests
    .\run.ps1 calibrate-gate   derive gate.threshold from the corpus
    .\run.ps1 sensitivity      context-width curve -> reports/context-sensitivity.md
    .\run.ps1 free             all three of the above

  COSTS MODEL CALLS
    .\run.ps1 eval             isolation + silent-failure   (~50 calls)
    .\run.ps1 ablate           counterfactual sweep         (~100 calls)
    .\run.ps1 calibrate        judge calibration            (~20 calls)
    .\run.ps1 refresh          all three, to un-stale reports/  (~170 calls)

  OTHER
    .\run.ps1 schema           regenerate the frontend's types from the models
    .\run.ps1 demo-traces      freeze pre-recorded traces for the keyless demo
    .\run.ps1 web              inspector UI (needs Node 20+)

  No API key is needed for any of the above.

"@ -ForegroundColor Gray
  }
}
