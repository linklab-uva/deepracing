
#ifndef PACK
	#ifdef _MSC_VER
		#define PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop) )
	#else
		#define PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
	#endif
#endif

#ifndef F1_DATALOGGER_TYPEDEFS_H
#define F1_DATALOGGER_TYPEDEFS_H
#include <cstdint>

namespace deepf1{

typedef uint8_t uint8; //Unsigned 8-bit integer

typedef int8_t int8; //Signed 8-bit integer

typedef uint16_t uint16; //Unsigned 16-bit integer

typedef int16_t int16; //Signed 16-bit integer

typedef uint32_t uint; //Unsigned 32-bit integer

typedef uint32_t uint32; //Another name for an unsigned 32-bit integer

typedef uint64_t uint64; //Unsigned 64-bit integer

typedef float F1scalar; //Single-precision (32-bit) floating point

}

#endif