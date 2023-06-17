#include <f1_datalogger/pcl_types.h>
namespace f1_datalogger
{
   
  std::ostream& 
  operator << (std::ostream& os, const PointXYZLapdistance& p)
  {
    os << "(";
    os << p.x << "," << p.y << "," << p.z << "," << p.lapdistance << ")";
    return (os);
  }
} 
