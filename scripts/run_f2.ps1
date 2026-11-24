$env:LAT = '12'; $env:RMAX = '5'; $env:OMP_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
$py = 'C:\Program Files\Python312\python.exe'
$ps = @()
foreach ($c in @('SU2 2.2', 'SU2 2.4', 'U1 1.5')) {
  foreach ($s in 1..4) {
    $a = $c.Split(' ')
    $out = "C:\work\esphys\out_" + $a[0] + "_" + $a[1] + "_" + $s + ".txt"
    $ps += Start-Process -FilePath $py -ArgumentList @('C:\work\esphys\f2_confinement.py', $a[0], $a[1], "$s") -NoNewWindow -PassThru -RedirectStandardOutput $out
  }
}
$ps | Wait-Process
Get-Content C:\work\esphys\out_*.txt
