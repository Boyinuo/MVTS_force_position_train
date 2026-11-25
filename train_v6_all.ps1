# Run the ten v6 experiments sequentially. Stop on the first training failure.
# Preview commands: powershell -ExecutionPolicy Bypass -File .\train_v6_all.ps1 -DryRun
# Run only the four new image-only jobs: add -ImageOnlyOnly.
# Skip the first two completed audiovisual jobs: add -StartFrom 3.
param(
    [switch]$DryRun,
    [switch]$ImageOnlyOnly,
    [ValidateRange(1, 10)]
    [int]$StartFrom = 1
)

$ErrorActionPreference = 'Stop'

$commonArguments = @(
    '--trim-stationary-xy-mm', '0.029',
    '--filter-xy-range',
    '--device', 'cuda',
    '--epochs', '30',
    '--batch-size', '8',
    '--workers', '10'
)

$jobs = @(
    @{ Model = 'model_triple_head'; Data = @('slide') },
    @{ Model = 'model_triple_head'; Data = @('slide', 'polyline') },
    @{ Model = 'model_triple_head'; Data = @('slide', 'press') },
    @{ Model = 'model_triple_head'; Data = @('slide', 'polyline', 'press') },
    @{ Model = 'model_contact_gated'; Data = @('slide', 'press') },
    @{ Model = 'model_contact_gated'; Data = @('slide', 'polyline', 'press') },
    # train.py currently supports --image-only only for model_triple_head.
    @{ Model = 'model_triple_head'; Data = @('slide'); ImageOnly = $true },
    @{ Model = 'model_triple_head'; Data = @('slide', 'polyline'); ImageOnly = $true },
    @{ Model = 'model_triple_head'; Data = @('slide', 'press'); ImageOnly = $true },
    @{ Model = 'model_triple_head'; Data = @('slide', 'polyline', 'press'); ImageOnly = $true }
)

if ($ImageOnlyOnly) {
    $jobs = @($jobs | Where-Object { $_.ImageOnly })
}
if ($StartFrom -gt $jobs.Count) {
    throw "StartFrom must be at most $($jobs.Count) for the selected jobs."
}

Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not $DryRun) {
        $null = Get-Command uv -ErrorAction Stop
    }

    for ($index = $StartFrom - 1; $index -lt $jobs.Count; $index++) {
        $job = $jobs[$index]
        $modelName = $job.Model -replace '^model_', ''
        $dataName = $job.Data -join '_'
        if ($job.ImageOnly) {
            # Pure vision has no audio input; mic selection is mutually exclusive.
            $modeArguments = @('--image-only')
            $inputModeName = 'image_only'
        } else {
            $modeArguments = @('--disable-mic-channels', '5')
            $inputModeName = 'no_mic5'
        }
        $output = "runs/v6_${modelName}_${dataName}_${inputModeName}_trim0p029_xyfilter"
        $roots = @($job.Data | ForEach-Object { ".cache/data_final_v6_${_}_contact" })
        $trainArguments = @('run', 'train.py', '--model', $job.Model, '--prepared-root') +
            $roots + $modeArguments + $commonArguments + @('--output', $output)

        Write-Host "[$($index + 1)/$($jobs.Count)] $output"
        Write-Host ('uv ' + ($trainArguments -join ' '))
        if ($DryRun) {
            continue
        }

        # Native commands run synchronously: the next job waits for uv to exit.
        & uv @trainArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed with exit code $LASTEXITCODE. Stopped at $output."
        }
    }

    if ($DryRun) {
        Write-Host 'Preview complete. No training was started.'
    } else {
        Write-Host "All $($jobs.Count - $StartFrom + 1) selected training runs completed successfully."
    }
} finally {
    Pop-Location
}
