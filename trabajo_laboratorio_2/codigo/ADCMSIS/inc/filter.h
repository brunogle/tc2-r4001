
/*
===============================================================================
 Name        : filter.h
 Author      : Israel Pavelek, Cesar Fuoco
 Version     : 1.2
 Copyright   : $(copyright)
 Description : main definition
===============================================================================
*/

#ifndef LOWPASS_H_
#define LOWPASS_H_

#include <stdint.h>
#include "arm_math.h"

#define FIR_TAP_NUM 151
#define IIR_TAP_NUM 20

#define SAMPLES_PER_BLOCK 1024

extern int32_t fir_taps[];
extern int32_t iir_taps[];
extern float32_t float_fir_taps[];
extern float32_t float_iir_taps[];


typedef enum{
	NO_PROCESAR,
	PROCESAR_A,
	PROCESAR_B,

}procesar_type_t;


#endif /* LOWPASS_H_ */
