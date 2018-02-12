
#include <stdio.h>
//#include "envH.h"
#include <iostream>
#include <fstream>
#include <deepf1_gsoap/deepf1_gsoap.nsmap>
int main(int argc, char** argv)
{

	soap* soap = soap_new();
	soap_set_mode(soap, SOAP_XML_INDENT | SOAP_XML_STRICT | SOAP_XML_DEFAULTNS);
	deepf1_gsoap::ground_truth_sample* ground_truth_out = deepf1_gsoap::soap_new_ground_truth_sample(soap);
	ground_truth_out->image_file = std::string("asdf.jkl;");
	ground_truth_out->sample.m_steer = 1.0;
	ground_truth_out->sample.m_brake = 0.5;
	ground_truth_out->sample.m_throttle = 0.48;
	std::fstream* out_fs = new std::fstream("ground_truth_sample.xml", std::fstream::out);
	soap->os = out_fs;
	if (deepf1_gsoap::soap_write_ground_truth_sample(soap, ground_truth_out) != SOAP_OK)
	{
		std::cerr << "Error writing ground_truth_sample.xml file" << std::endl;
		soap_stream_fault(soap, std::cerr);
		exit(1);
	}
	out_fs->close();
	delete out_fs;
   /*

	std::fstream* in_fs = new std::fstream("input.xml", std::fstream::in);
	deepf1::ground_truth_label* ground_truth_in = deepf1::soap_new_ground_truth_label(soap);
	soap->is = in_fs;
	if (deepf1::soap_read_ground_truth_label(soap, ground_truth_in) != SOAP_OK) {
		std::cerr << "Error getting input.xml file" << std::endl;
		soap_stream_fault(soap, std::cerr);
		exit(1);
	}
	printf("Ground truth read from file, file_name: %s, steering: %f, brake: %f, throttle: %f\n", ground_truth_in->file_name.c_str(),
		ground_truth_in->m_steer, ground_truth_in->m_brake, ground_truth_in->m_throttle);
	in_fs->close();

	delete in_fs;


	*/
	// Delete instances
	soap_destroy(soap);
	// Delete data
	soap_end(soap);
	// Free soap struct engine context
	soap_free(soap);

	return 0;
}
