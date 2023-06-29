#ifndef DEEPRACING_CPP_PCL_POINT_TYPES_HPP_
#define DEEPRACING_CPP_PCL_POINT_TYPES_HPP_
#include <pcl/point_types.h>

namespace deepracing
{
    struct EIGEN_ALIGN16 _PointXYZArclength
    {
        PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])
        union
        {
        struct
        {
            float arclength;
        };
        float data_c[4];
        };
        PCL_MAKE_ALIGNED_OPERATOR_NEW
    };
    
    struct PointXYZArclength : public _PointXYZArclength
    {
        inline constexpr PointXYZArclength (const _PointXYZArclength &p) : PointXYZArclength{p.x, p.y, p.z, p.arclength} {}

        inline constexpr PointXYZArclength (float _arclength = 0.f) : PointXYZArclength(0.f, 0.f, 0.f, _arclength) {}

        inline constexpr PointXYZArclength (float _x, float _y, float _z, float _arclength = 0.f) : _PointXYZArclength{{{_x, _y, _z, 1.0f}}, {{_arclength}}} {}
        
        friend std::ostream& operator << (std::ostream& os, const PointXYZArclength& p);
    };
    std::ostream& operator << (std::ostream& os, const PointXYZArclength& p);


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
    

    struct EIGEN_ALIGN16 _PointXYZSpeed
    {
        PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])
        union
        {
        struct
        {
            float speed;
        };
        float data_c[4];
        };
        PCL_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct PointXYZSpeed : public _PointXYZSpeed
    {
        inline constexpr PointXYZSpeed (const _PointXYZSpeed &p) : PointXYZSpeed{p.x, p.y, p.z, p.speed} {}

        inline constexpr PointXYZSpeed (float _speed = 0.f) : PointXYZSpeed(0.f, 0.f, 0.f, _speed) {}

        inline constexpr PointXYZSpeed (float _x, float _y, float _z, float _speed = 0.f) : _PointXYZSpeed{{{_x, _y, _z, 1.0f}}, {{_speed}}} {}
        
        friend std::ostream& operator << (std::ostream& os, const PointXYZSpeed& p);
    };
    std::ostream& operator << (std::ostream& os, const PointXYZSpeed& p);


    struct EIGEN_ALIGN16 _PointXYZTime
    {
        PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])
        union
        {
        struct
        {
            float time;
        };
        float data_c[4];
        };
        PCL_MAKE_ALIGNED_OPERATOR_NEW
    };
    
    struct PointXYZTime : public _PointXYZTime
    {
        inline constexpr PointXYZTime (const _PointXYZTime &p) : PointXYZTime{p.x, p.y, p.z, p.time} {}

        inline constexpr PointXYZTime (float _time = 0.f) : PointXYZTime(0.f, 0.f, 0.f, _time) {}

        inline constexpr PointXYZTime (float _x, float _y, float _z, float _time = 0.f) : _PointXYZTime{{{_x, _y, _z, 1.0f}}, {{_time}}} {}
        
        friend std::ostream& operator << (std::ostream& os, const PointXYZTime& p);
    };
    std::ostream& operator << (std::ostream& os, const PointXYZTime& p);
}

POINT_CLOUD_REGISTER_POINT_STRUCT(deepracing::PointXYZArclength,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, arclength, arclength)
)
POINT_CLOUD_REGISTER_POINT_WRAPPER(deepracing::PointXYZArclength, deepracing::_PointXYZArclength)

POINT_CLOUD_REGISTER_POINT_STRUCT(deepracing::PointXYZLapdistance,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, lapdistance, lapdistance)
)
POINT_CLOUD_REGISTER_POINT_WRAPPER(deepracing::PointXYZLapdistance, deepracing::_PointXYZLapdistance)

POINT_CLOUD_REGISTER_POINT_STRUCT(deepracing::PointXYZSpeed,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, speed, speed)
)
POINT_CLOUD_REGISTER_POINT_WRAPPER(deepracing::PointXYZSpeed, deepracing::_PointXYZSpeed)

POINT_CLOUD_REGISTER_POINT_STRUCT(deepracing::PointXYZTime,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, time, time)
)
POINT_CLOUD_REGISTER_POINT_WRAPPER(deepracing::PointXYZTime, deepracing::_PointXYZTime)

#endif