################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include src/subdir.mk
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := ADCMSIS
BUILD_ARTIFACT_EXTENSION := axf
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all:
	+@$(MAKE) --no-print-directory main-build && $(MAKE) --no-print-directory post-build

# Main-build Target
main-build: ADCMSIS.axf

# Tool invocations
ADCMSIS.axf: $(OBJS) $(USER_OBJS) makefile $(OPTIONAL_TOOL_DEPS)
	@echo 'Building target: $@'
	@echo 'Invoking: MCU Linker'
	arm-none-eabi-gcc -nostdlib -L"C:\Users\Bruno\dev\utn\test\workspace\lpc_board_nxp_lpcxpresso_1769\Debug" -L"C:\Users\Bruno\dev\utn\test\workspace\lpc_chip_175x_6x\Debug" -L"C:\Users\Bruno\dev\utn\test\workspace\CMSIS_DSPLIB_CM3\lib" -Xlinker -Map="ADCMSIS.map" -Xlinker --cref -Xlinker --gc-sections -Xlinker -print-memory-usage -mcpu=cortex-m3 -mthumb -T ADCMSIS_Debug.ld -o "ADCMSIS.axf" $(OBJS) $(USER_OBJS) $(LIBS) -llpc_board_nxp_lpcxpresso_1769 -llpc_chip_175x_6x -lCMSIS_DSPLIB_CM3
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) ADCMSIS.axf
	-@echo ' '

post-build:
	-@echo 'Performing post-build steps'
	-arm-none-eabi-size "ADCMSIS.axf"; # arm-none-eabi-objcopy -v -O binary "ADCMSIS.axf" "ADCMSIS.bin" ; # checksum -p LPC1769 -d "ADCMSIS.bin";
	-@echo ' '

.PHONY: all clean dependents main-build post-build

-include ../makefile.targets
