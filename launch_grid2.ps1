$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$p = Start-Process -FilePath python -ArgumentList 'C:\SZL\cram-study\grid2.py' -WindowStyle Hidden -PassThru
Write-Output ('GRID2_PID=' + $p.Id)
