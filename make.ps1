Write-Host "---------- Reformating with black... ----------" -ForegroundColor Blue
black .\ml_rest_fastapi .\tests
Write-Host "---------- Analysing with pylint... -----------" -ForegroundColor Blue
pylint --recursive=y .\ml_rest_fastapi .\tests
Write-Host "---------- Validating with mypy... ------------" -ForegroundColor Blue
mypy --pretty --config-file=mypy.ini .\ml_rest_fastapi
