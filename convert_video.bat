@echo off
echo Converting video...
echo.

:: Find the most recent output*.avi file
for /f "delims=" %%a in ('dir /b /od output_*.avi') do set "latest_output=%%a"

if not defined latest_output (
    echo No output_*.avi files found.
    echo Make sure to run the script first to generate output video files.
    goto :end
)

echo Found latest output file: %latest_output%
set mp4_output=%latest_output:.avi=.mp4%
echo Will convert to: %mp4_output%

:: Try conda ffmpeg first
set CONDA_FFMPEG="%CONDA_PREFIX%\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg.exe"
if exist %CONDA_FFMPEG% (
    echo Found ffmpeg in conda environment at %CONDA_FFMPEG%
    %CONDA_FFMPEG% -i "%latest_output%" -c:v libx264 -preset medium -crf 23 "%mp4_output%"
    if %ERRORLEVEL% == 0 (
        echo Conversion successful! File saved as %mp4_output%
        goto :end
    ) else (
        echo Conversion failed with error code %ERRORLEVEL%
    )
)

:: Try the larger ffmpeg executable
set CONDA_FFMPEG_LARGE="%CONDA_PREFIX%\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
if exist %CONDA_FFMPEG_LARGE% (
    echo Found larger ffmpeg in conda environment at %CONDA_FFMPEG_LARGE%
    %CONDA_FFMPEG_LARGE% -i "%latest_output%" -c:v libx264 -preset medium -crf 23 "%mp4_output%"
    if %ERRORLEVEL% == 0 (
        echo Conversion successful! File saved as %mp4_output%
        goto :end
    ) else (
        echo Conversion failed with error code %ERRORLEVEL%
    )
)

:: Try WinGet ffmpeg as a last resort
set FFMPEG_PATH="%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget\ffmpeg.exe"
if exist %FFMPEG_PATH% (
    echo Found ffmpeg at %FFMPEG_PATH%
    %FFMPEG_PATH% -i "%latest_output%" -c:v libx264 -preset medium -crf 23 "%mp4_output%"
    if %ERRORLEVEL% == 0 (
        echo Conversion successful! File saved as %mp4_output%
    ) else (
        echo Conversion failed with error code %ERRORLEVEL%
    )
) else (
    echo No ffmpeg found at any expected location.
    echo Please download ffmpeg manually and retry.
)

:end
pause 