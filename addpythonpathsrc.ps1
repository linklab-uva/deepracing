if (([string]::IsNullOrEmpty($ENV:PYTHONPATH)))
{
    $ENV:PYTHONPATH=$PSScriptRoot+"\deepracing_py"
}
else
{
    $ENV:PYTHONPATH=$PSScriptRoot+"\deepracing_py;"+$ENV:PYTHONPATH
}
$ENV:PYTHONPATH=$PSScriptRoot+"\DCNN-Pytorch;"+$ENV:PYTHONPATH