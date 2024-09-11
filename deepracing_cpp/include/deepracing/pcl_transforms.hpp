 #ifndef DEEPRACING_PCL_TRANSFORMS_H
 #define DEEPRACING_PCL_TRANSFORMS_H

 #include <deepracing/pcl_types.hpp>  
 #include <pcl/common/transforms.h>
 namespace deepracing
 {
 namespace transforms
 {
    
 template <typename PointT> void
 transformPointCloudWithOrientation (const pcl::PointCloud<PointT> &cloud_in,
                                    pcl::PointCloud<PointT> &cloud_out,
                                 const Eigen::Isometry3f &transform,
                                 bool copy_all_fields = true)
 {
   std::size_t npts = cloud_in.size();
   // In order to transform the data, we need to remove NaNs
   cloud_out.is_dense = cloud_in.is_dense;
   cloud_out.header   = cloud_in.header;
   cloud_out.width    = static_cast<int> (npts);
   cloud_out.height   = 1;
   cloud_out.resize (npts);
   cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
   cloud_out.sensor_origin_      = cloud_in.sensor_origin_;
  
//    pcl::detail::Transformer<float> tf (transform.matrix());
   // If the data is dense, we don't need to check for NaN
   if (cloud_in.is_dense)
   {
     for (std::size_t i = 0; i < cloud_out.size (); ++i)
     {
       // Copy fields first, then transform
       if (copy_all_fields)
         cloud_out[i] = cloud_in[i];
       cloud_out[i].getVector3fMap () = transform*(cloud_in[i].getVector3fMap());
       deepracing::Quaternion4fMapConst qmapin = cloud_in[i].getQuaternionfMap();
       cloud_out[i].getQuaternionfMap() = Eigen::Quaternionf(transform.rotation()*qmapin).normalized();
     }
   }
   // Dataset might contain NaNs and Infs, so check for them first.
   else
   {
     for (std::size_t i = 0; i < cloud_out.size (); ++i)
     {
       // Copy fields first, then transform
       if (copy_all_fields)
         cloud_out[i] = cloud_in[i];
  
       if (!std::isfinite (cloud_in[i].x) ||
           !std::isfinite (cloud_in[i].y) ||
           !std::isfinite (cloud_in[i].z))
         continue;
       cloud_out[i].getVector3fMap () = transform*(cloud_in[i].getVector3fMap());
       cloud_out[i].getQuaternionfMap() = Eigen::Quaternionf(transform.rotation()*(cloud_in[i].getQuaternionfMap())).normalized();
     }
   }
 }
 template <typename PointT> void
 transformPointCloudWithOrientation (const pcl::PointCloud<PointT> &cloud_in,
                                    pcl::PointCloud<PointT> &cloud_out,
                                 const Eigen::Affine3f &transform,
                                 bool copy_all_fields = true)
 {
    Eigen::Isometry3f isometry;
    isometry.fromPositionOrientationScale(transform.translation(), transform.rotation(), Eigen::Vector3f::Ones());
    transformPointCloudWithOrientation<PointT>(cloud_in, cloud_out, isometry, copy_all_fields);
 }
 }
 }

  #endif