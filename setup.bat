@echo off
cls

echo Setup: [92mSTARTED[0m
echo.

if exist .venv\ (

  echo Step 1 of 3: Activating virtual environment...

  call .venv\Scripts\activate.bat > NUL 2> NUL
  if errorlevel 1 (
    call .venv\Scripts\Activate.ps1
  )
  if errorlevel 1 goto :cantactivate

  echo Step 2 of 3: Upgrading pip...

  call python --version >NUL
  if errorlevel 1 goto :nopython
  call python -m pip install --upgrade pip >NUL

  echo Step 3 of 3: Checking the requirements...

  call python -m pip install -r .requirements >NUL

  goto :success

) else (

  echo Step 1 of 4: Creating virtual environment...

  call python --version >NUL
  if errorlevel 1 goto :nopython
  call python -m venv .venv

  echo Step 2 of 4: Activating virtual environment......

  call .venv\Scripts\activate.bat > NUL 2> NUL
  if errorlevel 1 (
    call .venv\Scripts\Activate.ps1
  )
  if errorlevel 1 goto :cantactivate

  echo Step 3 of 4: Upgrading pip...

  call python -m pip install --upgrade pip >NUL

  echo Step 4 of 4: Installing requirements. This may take a while...

  call python -m pip install -r .requirements --use-pep517 >NUL

  goto :success

)

:success
echo.
echo Setup: [92mFINISHED[0m
goto :end

:nopython
echo Cannot find command 'python'. Make sure it's installed and added to PATH.
echo.
echo Setup: [91mFAILED[0m

:cantactivate
echo Cannot activate virtual environment. Make sure you use a command prompt.

:end
