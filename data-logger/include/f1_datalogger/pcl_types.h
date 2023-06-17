#ifndef F1_DATALOGGER_PCL_POINT_TYPES_H_
#define F1_DATALOGGER_PCL_POINT_TYPES_H_
#include <pcl/point_types.h>
#include <f1_datalogger/visibility_control.h>

namespace f1_datalogger
{
    struct EIGEN_ALIGN16 _PointXYZLapdistance
    {
        PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])
        union
        {
        struct
        {
            float lapdistance;
        };
        float data_c[4];
        };
        PCL_MAKE_ALIGNED_OPERATOR_NEW
    };
    
    struct PointXYZLapdistance : public _PointXYZLapdistance
    {
        inline constexpr PointXYZLapdistance (const _PointXYZLapdistance &p) : PointXYZLapdistance{p.x, p.y, p.z, p.lapdistance} {}

        inline constexpr PointXYZLapdistance (float _lapdistance = 0.f) : PointXYZLapdistance(0.f, 0.f, 0.f, _lapdistance) {}

        inline constexpr PointXYZLapdistance (float _x, float _y, float _z, float _lapdistance = 0.f) : _PointXYZLapdistance{{{_x, _y, _z, 1.0f}}, {{_lapdistance}}} {}
        
        friend std::ostream& operator << (std::ostream& os, const PointXYZLapdistance& p);
    };
    std::ostream& operator << (std::ostream& os, const PointXYZLapdistance& p);
}



#endif