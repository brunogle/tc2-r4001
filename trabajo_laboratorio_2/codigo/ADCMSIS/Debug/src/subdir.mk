################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/ADCMSIS.c \
../src/adc.c \
../src/cr_startup_lpc175x_6x.c \
../src/crp.c \
../src/dac.c \
../src/filter.c \
../src/sysinit.c 

C_DEPS += \
./src/ADCMSIS.d \
./src/adc.d \
./src/cr_startup_lpc175x_6x.d \
./src/crp.d \
./src/dac.d \
./src/filter.d \
./src/sysinit.d 

OBJS += \
./src/ADCMSIS.o \
./src/adc.o \
./src/cr_startup_lpc175x_6x.o \
./src/crp.o \
./src/dac.o \
./src/filter.o \
./src/sysinit.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c src/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: MCU C Compiler'
	arm-none-eabi-gcc -DDEBUG -D__CODE_RED -DCORE_M3 -D__USE_LPCOPEN -D__USE_CMSIS_DSPLIB=CMSIS_DSPLIB_CM3 -D__LPC17XX__ -D__REDLIB__ -I"C:\Users\Bruno\dev\utn\test\workspace\lpc_board_nxp_lpcxpresso_1769\inc" -I"C:\Users\Bruno\dev\utn\test\workspace\ADCMSIS\inc" -I"C:\Users\Bruno\dev\utn\test\workspace\lpc_chip_175x_6x\inc" -I"C:\Users\Bruno\dev\utn\test\workspace\CMSIS_DSPLIB_CM3\inc" -O0 -fno-common -g -Wall -c -fmessage-length=0 -fno-builtin -ffunction-sections -fdata-sections -fmerge-constants -fmacro-prefix-map="$(<D)/"= -mcpu=cortex-m3 -mthumb -D__REDLIB__ -fstack-usage -specs=redlib.specs -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.o)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-src

clean-src:
	-$(RM) ./src/ADCMSIS.d ./src/ADCMSIS.o ./src/adc.d ./src/adc.o ./src/cr_startup_lpc175x_6x.d ./src/cr_startup_lpc175x_6x.o ./src/crp.d ./src/crp.o ./src/dac.d ./src/dac.o ./src/filter.d ./src/filter.o ./src/sysinit.d ./src/sysinit.o

.PHONY: clean-src

