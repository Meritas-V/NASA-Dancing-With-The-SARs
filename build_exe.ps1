# Build the Windows app. Run from the project folder inside the activated virtual environment:
#   .\build_exe.ps1            -> dist\DancingWithTheSARs\ (folder; zip it to share)
#   .\build_exe.ps1 -OneFile   -> dist\DancingWithTheSARs.exe (single file, slower to start)
param([switch]$OneFile)
python -m pip install --upgrade pyinstaller
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
if ($OneFile) { $env:ONEFILE = "1" } else { Remove-Item Env:ONEFILE -ErrorAction SilentlyContinue }
pyinstaller --noconfirm --clean DancingWithTheSARs.spec
if ($OneFile) {
    Write-Host "`nDone: dist\DancingWithTheSARs.exe"
} else {
    Compress-Archive -Path dist\DancingWithTheSARs -DestinationPath dist\DancingWithTheSARs-windows.zip -Force
    Write-Host "`nDone: dist\DancingWithTheSARs\DancingWithTheSARs.exe and dist\DancingWithTheSARs-windows.zip"
}
