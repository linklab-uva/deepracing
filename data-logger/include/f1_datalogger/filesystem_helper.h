#ifndef INCLUDE_F1_DATALOGGER_FILESYSTEM_HELPER_H_
#define INCLUDE_F1_DATALOGGER_FILESYSTEM_HELPER_H_
#ifdef BOOST_FILESYSTEM
  #include <boost/filesystem.hpp>
  namespace fs = boost::filesystem;
#else
  #include <filesystem>
  namespace fs = std::filesystem;
#endif
#endif