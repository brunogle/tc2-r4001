/*
===============================================================================
 Name        : ADCMSIS.c
 Author      : Israel Pavelek, Cesar Fuoco
 Version     : 1.2
 Copyright   : $(copyright)
 Description : main definition
===============================================================================
*/

#include <filter.h>
#include "adc.h"
#include "dac.h"
#include "arm_math.h"

typedef enum{
	TALKTHROUGH,
	FIR,
	IIR,
}filter_type_t;

filter_type_t filter = IIR; //Función que ejecuta

extern procesar_type_t procesar;

//Estructuras
arm_fir_instance_q31 SFIR;
arm_biquad_casd_df1_inst_q31 SIIR;

//Variables de estado
q31_t fir_state[FIR_TAP_NUM+SAMPLES_PER_BLOCK-1];
q31_t iir_state[IIR_TAP_NUM+SAMPLES_PER_BLOCK-1];

//Ping Pong Buffers
q31_t InputA[SAMPLES_PER_BLOCK];
q31_t InputB[SAMPLES_PER_BLOCK];
q31_t OutputA[SAMPLES_PER_BLOCK];
q31_t OutputB[SAMPLES_PER_BLOCK];

int main(void) {


	uint32_t index;
	float32_t max;
    SystemCoreClockUpdate();
    Board_Init();
    Board_LED_Set(0, true);

    //Conversion float to q31
    arm_float_to_q31(float_fir_taps, fir_taps,FIR_TAP_NUM);

    arm_max_f32(float_iir_taps,IIR_TAP_NUM,&max,&index);
    //max=;
    if(max>=1)arm_scale_f32(float_iir_taps, ((float32_t)1/((1<<(uint32_t)max))),float_iir_taps,IIR_TAP_NUM );
    arm_float_to_q31(float_iir_taps, iir_taps, IIR_TAP_NUM);

    //Inicializa los filtros
	arm_fir_init_q31(&SFIR,FIR_TAP_NUM,fir_taps,fir_state,SAMPLES_PER_BLOCK);
    arm_biquad_cascade_df1_init_q31(&SIIR,(uint32_t)IIR_TAP_NUM/5,iir_taps,iir_state,(uint32_t)max);


    //Inicializa ADC y DAC
    adcInit();
    dacInit();

    while(true){
		if(procesar){
			switch (filter){
				case TALKTHROUGH:
						for(uint16_t i=0;i<SAMPLES_PER_BLOCK;i++){
							if(procesar==PROCESAR_A)OutputA[i]=InputA[i];
							else OutputB[i]=InputB[i];
						}
						break;

				case IIR:
						if(procesar==PROCESAR_A){
							arm_biquad_cascade_df1_q31  (&SIIR, InputA, OutputA, SAMPLES_PER_BLOCK);
						}else {
							arm_biquad_cascade_df1_q31  (&SIIR, InputB, OutputB, SAMPLES_PER_BLOCK);
						}
						break;

				case FIR:
						if(procesar==PROCESAR_A){
							arm_fir_q31(&SFIR,InputA, OutputA, SAMPLES_PER_BLOCK);
						}else {
							arm_fir_q31(&SFIR,InputB, OutputB , SAMPLES_PER_BLOCK);
						}
						break;
			}
			procesar=NO_PROCESAR;
		}
	}
}

