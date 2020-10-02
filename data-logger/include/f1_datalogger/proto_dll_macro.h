#ifndef F1_DATALOGGER_PROTO_DLL_MACRO__H_
#define F1_DATALOGGER_PROTO_DLL_MACRO__H_


#ifndef  F1_DATALOGGER_PROTO_DLL_MACRO
    #if defined _WIN32 || defined __CYGWIN__
        #ifdef __GNUC__
            #define F1_DATALOGGER_PROTO_DLL_MACRO __attribute__ ((dllimport))
        #else
            #define F1_DATALOGGER_PROTO_DLL_MACRO __declspec(dllimport)
        #endif
    #else
        #define F1_DATALOGGER_PROTO_DLL_MACRO __attribute__ ((visibility("default")))
    #endif
#endif

#endif