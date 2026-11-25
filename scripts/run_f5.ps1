$env:OMP_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'
$py = 'C:\Program Files\Python312\python.exe'
$ps = @()
foreach ($s in 1..18) {
  $out = "C:\work\esphys\f5_log_" + $s + ".txt"
  $ps += Start-Process -FilePath $py -ArgumentList @('C:\work\esphys\f5_gA.py', "$s", '2') -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError ($out + ".err")
}
$ps | Wait-Process
Get-Content C:\work\esphys\f5_log_*.txt
