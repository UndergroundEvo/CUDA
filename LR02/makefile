CC = g++
NVCC = nvcc

CFLAGS = -Wall -g

CCFILES :=  LR02_1C LR02_1C_2
CUFILES :=  LR02_1G LR02_1G_2

all: $(CCFILES) $(CUFILES)

$(CCFILES): %: src/%.cpp
	$(CC) $(CFLAGS) -o $@ $<

$(CUFILES): %: src/%.cu
	$(NVCC) -o $@ $<


clean:
	$(RM) $(CCFILES) $(CUFILES)

help:
	@echo " • С таймером: time(./prog)"
	@echo " • Одно ядро:  taskset -c 0 ./prog"
	@echo " • Все ядра:   taskset -c cpu-core-number(s) ./prog"
	@echo " • Имба:       time(taskset -c 0,1 ./lab10_3)"
