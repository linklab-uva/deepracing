#include <f1_datalogger/pcl_types.h>
namespace f1_datalogger
{
   
  std::ostream& 
  operator << (std::ostream& os, const PointXYZLapdistance& p)
  {
    os << "(";
    os << p.x << "," << p.y << "," << p.z << "," << p.lapdistance;
    os << ")";
    return (os);
  }

  std::ostream& 
  operator << (std::ostream& os, const PointXYZTime& p)
  {
    os << "(";
    os << p.x << "," << p.y << "," << p.z << "," << p.time;
    os << ")";
    return (os);
  }

  
  std::ostream& 
  operator << (std::ostream& os, const PointXYZSpeed& p)
  {
    os << "(";
    os << p.x << "," << p.y << "," << p.z << "," << p.speed;
    os << ")";
    return (os);
  }

  
  std::ostream& 
  operator << (std::ostream& os, const PointXYZArclength& p)
  {
    os << "(";
    os << p.x << "," << p.y << "," << p.z << "," << p.arclength;
    os << ")";
    return (os);
  }


} 
