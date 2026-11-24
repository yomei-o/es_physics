$env:LAT = '12'; $env:OMP_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
$py = 'C:\Program Files\Python312\python.exe'
$ps = @()
foreach ($b in @('2.2', '2.3', '2.4')) {
  foreach ($s in 1..2) {
    $out = "C:\work\esphys\flow_" + $b + "_" + $s + ".txt"
    $ps += Start-Process -FilePath $py -ArgumentList @('C:\work\esphys\f4b_flow_scale.py', $b, "$s") -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError ($out + ".err")
  }
}
$ps | Wait-Process
Get-Content C:\work\esphys\flow_*.txt
Get-Content C:\work\esphys\flow_*.err
