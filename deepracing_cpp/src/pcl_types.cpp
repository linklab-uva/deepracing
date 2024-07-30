#include <deepracing/pcl_types.hpp>
namespace deepracing
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

  
  std::ostream& 
  operator << (std::ostream& os, const PointXYZTAL& p)
  {
    os << "(";
    os << p.x << "," << p.y << "," << p.z << "," << p.time << "," << p.arclength << "," << p.lapdistance;
    os << ")";
    return (os);
  }

  std::ostream& 
  operator << (std::ostream& os, const PointWidthMap& p)
  {
    os << "(";
    os << p.x << "," << p.y << "," << p.z << "," << p.i << "," << p.j << "," << p.k << "," << p.w << "," << p.ib_distance << "," << p.ob_distance;
    os << ")";
    return (os);
  }

} 
