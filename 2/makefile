.PHONY: all
all:
	nvcc lab01_1G.cu -o lab01_1G
	g++  lab01_1C.cpp -o lab01_1C
	g++  lab01_2.cpp -o lab01_2


.PHONY: clean
clean:
	rm lab01_1G
	rm lab01_1C

.PHONY: zabul
zabul:
	@echo "c таймером: time(./prog)"
	@echo "одно ядро: taskset -c 0 ./prog"
	@echo "все ядра: taskset -c cpu-core-number(s) ./prog"
	@echo "имба: time(taskset -c 0,1 ./lab10_3)"