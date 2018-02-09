
#include <stdio.h>
//#include "envH.h"
#include <iostream>
#include <fstream>
#include <deepf1_gsoap/deepf1_gsoap.nsmap>
int main(int argc, char** argv)
{

	struct soap* soap = soap_new();
	soap_set_mode(soap, SOAP_XML_INDENT);
	std::fstream* fs = new std::fstream("cpu_times.xml", std::fstream::out);
	deepf1::cpu_times* cpu_times = deepf1::soap_new_cpu_times(soap);
	cpu_times->system = 123;
	cpu_times->user = 456;
	cpu_times->wall = 789;
	soap->os = fs;
	if (deepf1::soap_write_cpu_times(soap, cpu_times) != SOAP_OK)
	{
		std::cerr << "Error writing cpu_times.xml file" << std::endl;
		soap_stream_fault(soap, std::cerr);
		exit(1);
	}
	fs->close();
	delete fs;
	// Delete instances
	soap_destroy(soap);
	// Delete data
	soap_end(soap);
	// Free soap struct engine context
	soap_free(soap);

	return 0;
}
