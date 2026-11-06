@echo off
rem Uncharted VR test switches: starts the launcher with the settings below, for this session only.
rem Starting Uncharted.exe normally runs the game without any of them.
rem To use a switch, remove "rem " from the start of its line and save this file.
rem After the run, send user\last_run.log from this folder (or from your own saves folder).

cd /d "%~dp0"
if not exist "Uncharted.exe" (
    echo Put this file in the Uncharted folder, next to Uncharted.exe.
    pause
    exit /b 1
)

rem Timing statistics in last_run.log every 5 seconds: frame rate, shader compiles, GPU waits.
set BB_FRAME_STATS=1

rem Logs each step of the headset's loop that takes longer than this many milliseconds.
set BB_VR_STALL_LOG=50

rem Frame pacing. Default: one game frame per headset frame.
rem   adaptive = the pacing of earlier versions (vr-0.2)
rem   0        = no pacing at all
rem set BB_VR_CAP=adaptive
rem set BB_VR_CAP=0

rem Copies to the headset on the game's own GPU queue instead of a separate one.
rem set BB_VR_SHARED_QUEUE=1

rem Hands frames to the headset without first checking that the GPU has finished them (an older way).
rem set BB_VR_WAIT_FRAMES=1

rem Logs the first frames' pose tags, and any frame that finds no tag.
rem set BB_VR_TAGLOG=1

start "" "Uncharted.exe"
