param([string]$TargetPlatform = 'android-arm64')
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$previousDirectory = Get-Location
try {
    Set-Location -LiteralPath $project
    if (-not (Test-Path -LiteralPath 'assets/benchmark/manifest.json')) {
        throw 'Prepare the benchmark bundle with tools/prepare.py first.'
    }
    $changes = git -c "safe.directory=$project" status --porcelain
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect project Git status.' }
    if ($changes) { throw 'Commit source changes before building a reproducible release APK.' }
    $revision = git -c "safe.directory=$project" rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read project commit.' }
    flutter build apk --release --target-platform $TargetPlatform "--dart-define=APP_COMMIT=$revision"
    if ($LASTEXITCODE -ne 0) { throw 'Flutter release build failed.' }
    & .venv/Scripts/python tools/apk_size.py build/app/outputs/flutter-apk/app-release.apk --output artifacts/apk-size.json
    if ($LASTEXITCODE -ne 0) { throw 'APK inspection failed.' }
} finally {
    Set-Location -LiteralPath $previousDirectory
}
