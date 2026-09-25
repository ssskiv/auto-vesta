# Demo script for Windows
successfully tested with Dualshock 4, unsuccessfully with F310
## get started
1. turn vehicle on, set Neutral gear, check handbrake is **engaged**
2. turn on PCAN and OSCC modules and connect gamepad
    #### *once per computer:* 
    * install [PCAN drivers](https://www.peak-system.com/support/downloads/drivers/)
    * run `setup_oscc_win.bat` with administrator rights to check if all drivers installed and working and install necessary python packages
3. run from terminal `py ds4_teleop_win.py --list-can` to check if CAN bus working
4. run `py ds4_teleop_win.py --no-can` to explore controls  

now you are ready to drive.  

drive carefully and **DO NOT TOUCH** the steering wheel while script is running. It will harm yourself if OSCC will steer the wheels.
## IF YOU ARE READY TO DRIVE WITH GAMEPAD
1. turn vehicle on, set Neutral gear, check handbrake is **engaged**
2. turn on PCAN and OSCC modules and connect gamepad
3. run `py ds4_teleop_win.py` 
4. change gear to D(A in Lada Vesta) and disengage handbrake
4. now you are controlling the car with gamepad. If something unusual is happenning, push the Brake pedal immediatelly.  
