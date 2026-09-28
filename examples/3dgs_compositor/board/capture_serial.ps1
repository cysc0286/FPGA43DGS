param([Parameter(Mandatory=$true)][string]$OutputFile,[int]$Seconds=100)
$ErrorActionPreference='Stop'
$serial=[IO.Ports.SerialPort]::new('COM3',115200,[IO.Ports.Parity]::None,8,[IO.Ports.StopBits]::One)
$serial.Handshake=[IO.Ports.Handshake]::None
$serial.DtrEnable=$false
$serial.RtsEnable=$false
$writer=[IO.StreamWriter]::new($OutputFile,$false,[Text.UTF8Encoding]::new($false))
$writer.AutoFlush=$true
try{
    $serial.Open()
    Write-Output 'SERIAL_CAPTURE_READY'
    $timer=[Diagnostics.Stopwatch]::StartNew()
    while($timer.Elapsed.TotalSeconds -lt $Seconds){
        if($serial.BytesToRead -gt 0){$writer.Write($serial.ReadExisting())}
        Start-Sleep -Milliseconds 100
    }
}finally{$writer.Dispose();if($serial.IsOpen){$serial.Close()};$serial.Dispose()}
Write-Output "SERIAL_CAPTURE_SAVED=$OutputFile"
