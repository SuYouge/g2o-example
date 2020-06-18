#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/solver.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/types/icp/types_icp.h"

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace g2o;

/*
T=
    0.98199    0.131236    0.135917    -0.29855
  -0.129037    0.991327  -0.0249051 -0.00876509
  -0.138007  0.00691829    0.990407  -0.0242999
          0           0           0           1
*/

void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

void bundleAdjustment_predef(
    const vector<Point3f> &pts0,
    const vector<Point3f> &pts1);

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED); // 深度图为16位无符号数，单通道图像
    Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED); // 深度图为16位无符号数，单通道图像
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts1, pts2;

    for (DMatch m : matches)
    {
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0) // bad depth
            continue;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        float dd1 = float(d1) / 5000.0; // 5000.0由RGBD相机给出
        float dd2 = float(d2) / 5000.0;
        pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }
    cout << "3d-3d pairs: " << pts1.size() << endl;
    cout << "calling bundle adjustment" << endl;
    bundleAdjustment_predef(pts1, pts2);

    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches)
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void bundleAdjustment_predef(
    const vector<Point3f> &pts0,
    const vector<Point3f> &pts1)
{
    SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    // variable-size block solver
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverX>(g2o::make_unique<LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>>()));
    optimizer.setAlgorithm(solver);
    int vertex_id = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        // set up rotation and translation for this node
        Vector3d t(0, 0, 0);
        Quaterniond q;
        q.setIdentity();
        Eigen::Isometry3d cam; // camera pose
        cam = q;
        cam.translation() = t;
        // set up node， 不采用相机顶点
        VertexSE3 *vc = new VertexSE3();
        vc->setEstimate(cam); // 初始估计，可以由SVD给出bian
        vc->setId(vertex_id); // vertex id
        // set first cam pose fixed
        if (i == 0)
            vc->setFixed(true);
        // add to optimizer
        optimizer.addVertex(vc);
        vertex_id++;
    }

    for (size_t i = 0; i < pts0.size(); i++)
    {
        VertexSE3 *vp0 =
            dynamic_cast<VertexSE3 *>(optimizer.vertices().find(0)->second);
        VertexSE3 *vp1 =
            dynamic_cast<VertexSE3 *>(optimizer.vertices().find(1)->second);

        Vector3d pt0, pt1;
        pt0 << pts0[i].x , pts0[i].y, pts0[i].z;
        pt1 << pts1[i].x , pts1[i].y, pts1[i].z;

        Vector3d nm0(pt0);
        Vector3d nm1(pt1);
        nm0.normalize(); // not necessary ?
        nm1.normalize();

        Edge_V_V_GICP *e // new edge with correct cohort for caching
            = new Edge_V_V_GICP();
        e->setVertex(0, vp0); // first viewpoint
        e->setVertex(1, vp1); // second viewpoint

        EdgeGICP meas; // class for edges between two points rigidly attached to vertices
        meas.pos0 = pt0;
        meas.pos1 = pt1;
        meas.normal0 = nm0;
        meas.normal1 = nm1;

        e->setMeasurement(meas);
        meas = e->measurement();
        e->information().setIdentity();
        optimizer.addEdge(e);
    }
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << endl;

    optimizer.setVerbose(true);

    optimizer.optimize(15);

    cout << "T=\n" << dynamic_cast<VertexSE3 *>(optimizer.vertices().find(1)->second)
            ->estimate().matrix() << endl;

}