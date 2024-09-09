#include <deepracing/utils.hpp>
pcl::PointCloud<deepracing::PointXYZLapdistance> deepracing::Utils::closeBoundary(const pcl::PointCloud<deepracing::PointXYZLapdistance>& open_boundary){
    Eigen::VectorXd dL(open_boundary.size()-1);
    for(unsigned int i = 1; i < open_boundary.size(); i++)
    {
        dL[i-1] = open_boundary[i].lapdistance - open_boundary[i-1].lapdistance;
    }
    double meandL = dL.mean();
    const PointXYZLapdistance& p0 = open_boundary.at(0);
    const PointXYZLapdistance& pf = open_boundary.at(open_boundary.size()-1);
    Eigen::Vector3f pf_eigen = Eigen::Vector3f(pf.getVector3fMap());
    Eigen::Vector3f deltavec = Eigen::Vector3f(p0.getVector3fMap() - pf_eigen);
    double final_distance = deltavec.norm();
    if (final_distance<2.0*meandL)
    {
        return open_boundary;
    }
    Eigen::Vector3f direction = deltavec.normalized();
    unsigned int linspace_size = (unsigned int)std::round(final_distance/meandL);
    Eigen::VectorXd extra_ld = Eigen::VectorXd::LinSpaced(linspace_size, 0.0, final_distance);
    unsigned int extra_points = linspace_size - 2;
    pcl::PointCloud<deepracing::PointXYZLapdistance> rtn(open_boundary);
    rtn.resize(rtn.size() + extra_points);
    for (unsigned int i = 0; i < extra_points; i++)
    {
        PointXYZLapdistance& newpoint = rtn.at(open_boundary.size() + i);
        newpoint.getVector3fMap() = pf_eigen + direction*extra_ld[i+1];
        newpoint.lapdistance = pf.lapdistance + extra_ld[i+1];
    }
    return rtn;
}
std::map<std::int8_t, std::string> deepracing::Utils::trackNames()
{
    std::map<std::int8_t, std::string> rtn;
    rtn[0] = "Australia";
    rtn[1] = "Canada";
    rtn[2] = "China";
    rtn[3] = "Bahrain";
    rtn[4] = "Catalunya";
    rtn[5] = "Monaco";
    rtn[6] = "Montreal";
    rtn[7] = "Britain";
    rtn[8] = "Hockenheim";
    rtn[9] = "Hungaroring";
    rtn[10] = "Spa";
    rtn[11] = "Monza";
    rtn[12] = "Singapore";
    rtn[13] = "Suzuka";
    rtn[14] = "AbuDhabi";
    rtn[15] = "TexasF1";
    rtn[16] = "Brazil";
    rtn[17] = "Austria";
    rtn[18] = "Sochi";
    rtn[19] = "Mexico";
    rtn[20] = "Azerbaijan";
    rtn[21] = "Sakhir Short";
    rtn[22] = "Silverstone Short";
    rtn[23] = "Texas Short";
    rtn[24] = "Suzuka Short";
    rtn[25] = "Hanoi";
    rtn[26] = "Zandvoort";
    rtn[27] = "Imola";
    rtn[28] = "Portimao";
    rtn[29] = "Jeddah";
    rtn[30] = "Miami";
    rtn[31] = "VegasF1";
    rtn[32] = "Losail";
    rtn[33] = "Las Vegas Motor Speedway";
    rtn[34] = "Texas Motor Speedway";
    return rtn;
}
