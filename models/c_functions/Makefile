# Makefile for building c_compute_paths.so

# Compiler and flags
CC := gcc

# Parallel Computation
CFLAGS := -Wall -Wextra -fPIC -O2 -fopenmp
LDFLAGS := -shared -lgomp -lm -O2 -fopenmp

# Serial Computation
# CFLAGS := -Wall -Wextra -fPIC -O2 
# LDFLAGS := -shared -lgomp -lm -O2 

# Source files
SRC := c_anomaly_score.c c_compute_paths.c c_make_importance.c

# Object files
OBJ := $(SRC:.c=.o)

# Target shared library
TARGET := c_compute_paths.so c_anomaly_score.so c_make_importance.so

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)
