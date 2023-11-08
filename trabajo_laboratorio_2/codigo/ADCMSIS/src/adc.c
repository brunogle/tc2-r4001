/*
 ===============================================================================
 Name        : adc.c
 Authors     : Israel Pavelek, Cesar Fuoco
 Version     : 1.2
 Copyright   : $(copyright)
 Description : main definition
 ===============================================================================
 */

#include "board.h"
#include "filter.h"
#include "arm_math.h"
#include "dac.h"

uint8_t estado = 0;
procesar_type_t procesar=NO_PROCESAR;

extern q31_t InputA[SAMPLES_PER_BLOCK];
extern q31_t InputB[SAMPLES_PER_BLOCK];
extern q31_t OutputA[SAMPLES_PER_BLOCK];
extern q31_t OutputB[SAMPLES_PER_BLOCK];

#define SAMPLE_RATE 44100
#define CARGANDO_A false


void adcInit(void) {
	ADC_CLOCK_SETUP_T adc;

	Chip_ADC_Init(LPC_ADC, &adc);
	Chip_ADC_SetSampleRate(LPC_ADC, &adc, SAMPLE_RATE);

	Chip_ADC_EnableChannel(LPC_ADC, ADC_CH0, ENABLE);
	Chip_ADC_Int_SetChannelCmd(LPC_ADC, ADC_CH0, ENABLE);
	Chip_ADC_SetBurstCmd(LPC_ADC, ENABLE);

	NVIC_EnableIRQ(ADC_IRQn);
}



void ADC_IRQHandler(void) {
	static uint16_t data;
	static uint16_t index = 0;

	Chip_ADC_ReadValue(LPC_ADC, ADC_CH0, &data);

	if (estado==CARGANDO_A){
			InputA[index] = data >> 2 ;
			dacWrite(OutputA[index]);
	}
	else {
		InputB[index] = data >> 2 ;
		dacWrite( OutputB[index]);
	}

	index++;

	if (index == SAMPLES_PER_BLOCK) {
		index = 0;
		if(estado==CARGANDO_A)procesar=PROCESAR_A;
		else procesar=PROCESAR_B;
		estado ^= 1;
	}
}

