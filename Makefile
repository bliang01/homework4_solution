# compiler settings
CC=mpicc
CFLAGS=-c -Wall -fPIC
LDFLAGS=-shared

# target
LIBTARGET=libhomework4.so

# directories
SRC=src

LIB=lib
INCLUDE=include

# sources
LIBSOURCES=$(SRC)/heat.c
LIBOBJECTS=$(LIBSOURCES:.c=.o)

.PHONY: all lib clean

default: all

all: lib

lib: $(LIBTARGET)

test: lib
	mpiexec -n 3 python test_homework4.py

$(LIBTARGET): $(LIBOBJECTS)
	$(CC) -I$(INCLUDE) $(LDFLAGS) $(LIBOBJECTS) -o $(LIB)/$(LIBTARGET)

%.o : %.c
	$(CC) -I$(INCLUDE) $(CFLAGS) $< -o $@

clean:
	@rm -fr $(SRC)/*.o

clobber: clean
	@rm -fd $(LIB)/*.so
