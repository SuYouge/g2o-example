// https://www.cnblogs.com/gaoxiang12/p/5304272.html 2d2d的ba方法
// 半闲居士

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/block_solver.h>
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <sophus/se3.hpp>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

void bundleAdjustmentG2O(
    const vector<Point2f> &points_1,
    const vector<Point2f> &points_2,
    const Mat &K,
    Sophus::SE3d &pose);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "usage: pose_estimation_2d2d img1 img2" << endl;
        return 1;
    }

    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1,
             0, 521.0, 249.7,
             0, 0, 1);

    vector<Point2f> pts_1;
    vector<Point2f> pts_2;

    for (DMatch m : matches)
    {
        pts_1.push_back(keypoints_1[m.queryIdx].pt);
        pts_2.push_back(keypoints_2[m.trainIdx].pt);
    }
    cout << "2d-2d pairs: " << pts_1.size() << endl;

    Sophus::SE3d pose_g2o;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_1, pts_2, K, pose_g2o);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    t2 = chrono::steady_clock::now();

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

void bundleAdjustmentG2O(
    const vector<Point2f> &points_1,
    const vector<Point2f> &points_2,
    const Mat &K,
    Sophus::SE3d &pose)
{

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;             // pose is 6, landmark is 3
    typedef g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true);     // 打开调试输出

    cout << " g2o init done --- " << endl;

    for (int i = 0; i < 2; i++)
    {
        g2o::VertexSE3Expmap *v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if (i == 0)
            v->setFixed(true); // 第一个点固定为零
        // 预设值为单位Pose，因为我们不知道任何信息
        v->setEstimate(g2o::SE3Quat());
        optimizer.addVertex(v);
    }

    // 以第一帧为准
    for (size_t i = 0; i < points_1.size(); i++)
    {
        g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ();
        v->setId(2 + i);
        // 由于深度不知道，只能把深度设置为1了
        double z = 1;
        double x = (points_1[i].x - K.at<double>(0, 2)) * z / K.at<double>(0, 0);
        double y = (points_2[i].y - K.at<double>(1, 2)) * z / K.at<double>(1, 1);
        v->setMarginalized(true);
        v->setEstimate(Eigen::Vector3d(x, y, z));
        optimizer.addVertex(v);
    }

    // 准备相机参数
    g2o::CameraParameters *camera = new g2o::CameraParameters(K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // 准备边
    // 第一帧
    vector<g2o::EdgeProjectXYZ2UV *> edges;
    for (size_t i = 0; i < points_1.size(); i++)
    {
        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0)));
        edge->setMeasurement(Eigen::Vector2d(points_1[i].x, points_1[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        // 核函数
        // edge->setRobustKernel(new g2o::RobustKernelHuber());
        // http://docs.ros.org/fuerte/api/re_vision/html/classg2o_1_1OptimizableGraph_1_1Edge.html#a854375dd4b9ceb686e0609e4ec524fff
        // https://blog.csdn.net/hzwwpgmwy/article/details/79884070
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        edge->setRobustKernel(rk);
        rk->setDelta(1.0);
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }
    // 第二帧
    for (size_t i = 0; i < points_2.size(); i++)
    {
        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(1)));
        edge->setMeasurement(Eigen::Vector2d(points_2[i].x, points_2[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        // 核函数
        // edge->setRobustKernel(new g2o::RobustKernelHuber());
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        edge->setRobustKernel(rk);
        rk->setDelta(1.0);
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    cout << "开始优化" << endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cout << "优化完毕" << endl;

    //我们比较关心两帧之间的变换矩阵
    g2o::VertexSE3Expmap *v = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(1));
    Eigen::Isometry3d pose_cam = v->estimate();
    cout << "Pose=" << endl
         << pose_cam.matrix() << endl;

    // 以及所有特征点的位置
    for (size_t i = 0; i < points_1.size(); i++)
    {
        g2o::VertexSBAPointXYZ *v = dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(i + 2));
        cout << "vertex id " << i + 2 << ", pos = ";
        Eigen::Vector3d pos = v->estimate();
        cout << pos(0) << "," << pos(1) << "," << pos(2) << endl;
    }

    // 估计inlier的个数
    int inliers = 0;
    for (auto e : edges)
    {
        e->computeError();
        // chi2 就是 error*\Omega*error, 如果这个数很大，说明此边的值与其他边很不相符
        if (e->chi2() > 1)
        {
            cout << "error = " << e->chi2() << endl;
        }
        else
        {
            inliers++;
        }
    }
    cout << "inliers in total points: " << inliers << "/" << points_1.size() + points_2.size() << endl;
    optimizer.save("ba.g2o");
}