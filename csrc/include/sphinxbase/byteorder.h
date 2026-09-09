/* -*- c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ====================================================================
 * Copyright (c) 1999-2001 Carnegie Mellon University.  All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * This work was supported in part by funding from the Defense Advanced
 * Research Projects Agency and the National Science Foundation of the
 * United States of America, and the CMU Sphinx Speech Consortium.
 *
 * THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
 * NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ====================================================================
 *
 */

/*
 * byteorder.h -- Byte swapping ordering macros.
 *
 * **********************************************
 * CMU ARPA Speech Project
 *
 * Copyright (c) 1996 Carnegie Mellon University.
 * ALL RIGHTS RESERVED.
 * **********************************************
 *
 * HISTORY
 *
 * 2026-09-05: Use unsigned byte-swap intermediates after UBSan reported a
 * signed shift at s3io.c:701 while reading a big-endian v8_seg header.
 *
 * 2026-09-09: Swap float storage through memcpy rather than through an
 * integer lvalue.  See the comment above SWAP_FLOAT32.
 *
 * $Log: byteorder.h,v $
 * Revision 1.8  2005/09/01 21:09:54  dhdfu
 * Really, actually, truly consolidate byteswapping operations into
 * byteorder.h.  Where unconditional byteswapping is needed, SWAP_INT32()
 * and SWAP_INT16() are to be used.  The WORDS_BIGENDIAN macro from
 * autoconf controls the functioning of the conditional swap macros
 * (SWAP_?[LW]) whose names and semantics have been regularized.
 * Private, adhoc macros have been removed.
 *
 */

#ifndef __S2_BYTEORDER_H__
#define __S2_BYTEORDER_H__	1

#include <string.h>

#include <sphinxbase/prim_type.h>

/* Unsigned intermediates keep shifts defined for every input bit pattern. */
/* Macro to byteswap an int16 variable.  x = ptr to variable */
#define SWAP_INT16(x)	*(x) = (uint16)((((uint32)*(x) & 0x00ffU) << 8) | \
					 (((uint32)*(x) & 0xff00U) >> 8))

/* Macro to byteswap an int32 variable.  x = ptr to variable */
#define SWAP_INT32(x)	*(x) = (uint32)((((uint32)*(x) & 0x000000ffU) << 24) | \
					 (((uint32)*(x) & 0x0000ff00U) << 8) | \
					 (((uint32)*(x) & 0x00ff0000U) >> 8) | \
					 (((uint32)*(x) & 0xff000000U) >> 24))

/*
 * The float swaps used to reach their storage through an integer lvalue
 * -- SWAP_INT32((int32 *) x) -- which reads and writes a float object as
 * though its effective type were int32.  C leaves that undefined, and
 * nothing in this build turns strict aliasing off, so an optimizer is
 * entitled to assume the integer store cannot change any float the
 * caller later reads, and to reorder or drop accesses accordingly.
 *
 * Copy the bytes into a local integer instead, swap them there, and copy
 * them back.  memcpy between two local objects is defined however the
 * compiler chooses to implement it, and it is the form every compiler
 * folds back into a plain load, byte reverse, and store: on the
 * compilers this project builds with, the emitted code for both the
 * scalar swap and the per-frame loops in feat.c and sphinx_fe.c is the
 * same as the punning version produced, vectorization included.
 *
 * The intermediate is copied back into a float local rather than written
 * straight to *x, because keeping the outer accesses float-typed is what
 * lets the vectorizer disambiguate the loops; a memcpy directly onto *x
 * costs the vectorization.
 *
 * A union member would also be defined type punning in C, but not in
 * C++, and this header is installed.
 *
 * These now take what their names always said they took: SWAP_FLOAT32
 * wants a float32 *, SWAP_FLOAT64 a float64 *.  The punning version
 * accepted any 4-byte pointer, and also lost the argument parentheses,
 * so SWAP_FLOAT32(p + i) expanded to (int32 *) p + i and only worked
 * because int32 and float32 are the same width.
 */

/* Macro to byteswap a float32 variable.  x = ptr to variable */
#define SWAP_FLOAT32(x)							\
    do {								\
	float32 _swap_value_ = *(x);					\
	uint32 _swap_bits_;						\
	memcpy(&_swap_bits_, &_swap_value_, sizeof(_swap_bits_));	\
	SWAP_INT32(&_swap_bits_);					\
	memcpy(&_swap_value_, &_swap_bits_, sizeof(_swap_bits_));	\
	*(x) = _swap_value_;						\
    } while (0)

/* Macro to byteswap a float64 variable.  x = ptr to variable */
#define SWAP_FLOAT64(x)							\
    do {								\
	float64 _swap_value_ = *(x);					\
	uint32 _swap_low_, _swap_high_;					\
	memcpy(&_swap_low_, (char *)&_swap_value_,			\
	       sizeof(_swap_low_));					\
	memcpy(&_swap_high_, (char *)&_swap_value_ + sizeof(_swap_low_),\
	       sizeof(_swap_high_));					\
	SWAP_INT32(&_swap_low_);					\
	SWAP_INT32(&_swap_high_);					\
	memcpy((char *)&_swap_value_, &_swap_high_,			\
	       sizeof(_swap_high_));					\
	memcpy((char *)&_swap_value_ + sizeof(_swap_high_),		\
	       &_swap_low_, sizeof(_swap_low_));			\
	*(x) = _swap_value_;						\
    } while (0)

#ifdef WORDS_BIGENDIAN
#define SWAP_BE_64(x)
#define SWAP_BE_32(x)
#define SWAP_BE_16(x)
#define SWAP_LE_64(x) SWAP_FLOAT64(x)
#define SWAP_LE_32(x) SWAP_INT32(x)
#define SWAP_LE_16(x) SWAP_INT16(x)
#else
#define SWAP_LE_64(x)
#define SWAP_LE_32(x)
#define SWAP_LE_16(x)
#define SWAP_BE_64(x) SWAP_FLOAT64(x)
#define SWAP_BE_32(x) SWAP_INT32(x)
#define SWAP_BE_16(x) SWAP_INT16(x)
#endif

#endif
