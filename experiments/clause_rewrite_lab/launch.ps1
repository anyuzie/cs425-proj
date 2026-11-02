$ErrorActionPreference = 'Stop'
$configuration = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'runtime.local.json') -Raw | ConvertFrom-Json
& $configuration.python (Join-Path $PSScriptRoot 'run.py') @args
exit $LASTEXITCODE
