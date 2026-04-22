//gcc -Wall -O3 -lm -shared -o master.so -fPIC master.c
//gcc -Wall -O3 -lm -shared -o fnc/master.so -fPIC C/master.c

#include "fnc.c"

#include "frame/frame_fnc.c"
#include "frame/ICRS_esf_to_FSR_dex_car.c"
#include "frame/FSR_dex_car_to_ICRS_esf.c"
