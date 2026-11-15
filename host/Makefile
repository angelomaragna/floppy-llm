CC ?= cc
CFLAGS = -Os -std=c99 -Wall

floppy: floppy.c vocab.c
	$(CC) $(CFLAGS) -o floppy floppy.c vocab.c -lm

strip: floppy
	strip floppy

.PHONY: strip
