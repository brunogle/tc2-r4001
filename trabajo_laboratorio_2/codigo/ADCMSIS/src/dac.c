/*
===============================================================================
 Name        : dac.c
 Author      : Israel Pavelek, Cesar Fuoco
 Version     : 1.2
 Copyright   : $(copyright)
 Description : main definition
===============================================================================
*/

#include "board.h"

/* P0.26 -> AOUT */
void dacInit(void)
{
	Chip_DAC_Init(LPC_DAC);
}

void dacWrite(uint32_t v)
{
	Chip_DAC_UpdateValue(LPC_DAC, v);
}
