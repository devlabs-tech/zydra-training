param(
    [string]$Prompt,
    [string]$Model = "output/zydra-v2-hf",
    [int]$MaxNewTokens = 96,
    [switch]$FreeForm,
    [double]$Temperature = 0.8,
    [double]$TopP = 0.95
)

if ([string]::IsNullOrWhiteSpace($Prompt)) {
    $Prompt = Read-Host "Masukkan prompt untuk Zydra"
}

if ([string]::IsNullOrWhiteSpace($Prompt)) {
    Write-Error "Prompt tidak boleh kosong."
    exit 1
}

$arguments = @(
    "test_zydra.py",
    "--model", $Model,
    "--max-new-tokens", "$MaxNewTokens",
    "--prompt", $Prompt,
    "--temperature", "$Temperature",
    "--top-p", "$TopP"
)
if ($FreeForm) {
    $arguments += "--free-form"
}

python @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
