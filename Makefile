#******************************************************#
# Distributed UDDSketch                                #
#                                                      #
# Coded by Catiuscia Melle                             #
#                                                      #
# April 8, 2021                                        #
#                                                      #
#******************************************************#


NAME=$(shell uname -s)

ifeq ($(NAME),Linux)
	CC=icc
	CFLAGS=-std=c++14 -O3
else
	CC=clang++
	CFLAGS=-std=c++14 -Os
endif

SRCDIR=.
LIBPATH=/usr/local/igraph-install



IFLAGS=-I$(SRCDIR) -I$(LIBPATH)/include/igraph
LFLAGS=-L$(LIBPATH)/lib
LIBS=-ligraph


DEPS=$(SRCDIR)/DistributedSketch.cc $(SRCDIR)/Coordinator.cc $(SRCDIR)/Quantiles.cc $(SRCDIR)/Event.cc $(SRCDIR)/GraphNet.cc $(SRCDIR)/Graphs.cc $(SRCDIR)/Sketch.cc $(SRCDIR)/InputSet.cc $(SRCDIR)/Base.cc


INPUT=-DPOSI# -DALL#


REQUIRED_FOLDERS="./CSV ./Logs ./Dots"

TARGET=DistributedSketch



all: $(TARGET) $(_MKDIR)


_MKDIR:=$(shell for d in $(REQUIRED_FOLDERS);do [[ -d $$d ]] || mkdir -p $$d; done)

$(TARGET):$(DEPS)
	@echo "Compiling for" $(NAME)
	$(CC) $(CFLAGS) -o $@ $(DEPS) $(IFLAGS) $(LFLAGS) $(LIBS) $(INPUT) $(DISTINCT)


.PHONY: clean allclean distclean


clean:
	rm -f $(TARGET)
	rm -rf $(TARGET).dSYM

allclean:
	rm -f Dots/*
	rm -f Logs/*
	rm -f CSV/*


distclean:
	rm -f $(TARGET)
	rm -f Dots/*
	rm -f Logs/*
	rm -f CSV/*
	rmdir ./Dots ./Logs ./CSV

