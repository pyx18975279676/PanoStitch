#include "ransac.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <iostream>

using namespace cv;

namespace cvv
{

    int RANSACUpdateNumIters(double p, double ep, int modelPoints, int maxIters)
    {
        if (modelPoints <= 0)
            CV_Error(Error::StsOutOfRange, "the number of model points should be positive");

        p = MAX(p, 0.);
        p = MIN(p, 1.);
        ep = MAX(ep, 0.);
        ep = MIN(ep, 1.);

        // avoid inf's & nan's
        double num = MAX(1. - p, DBL_MIN);
        double denom = 1. - std::pow(1. - ep, modelPoints);
        if (denom < DBL_MIN)
            return 0;

        num = std::log(num);
        denom = std::log(denom);

        return denom >= 0 || -num >= maxIters * (-denom) ? maxIters : cvRound(num / denom);
    }


    class RANSACPointSetRegistrator : public PointSetRegistrator
    {
    public:
        RANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb = Ptr<PointSetRegistrator::Callback>(),
            int _modelPoints = 0, double _threshold = 0, double _confidence = 0.99, int _maxIters = 1000)
            : cb(_cb), modelPoints(_modelPoints), threshold(_threshold), confidence(_confidence), maxIters(_maxIters) {}

        int findInliers(const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh, OutputArray _inlierPoints1 = noArray(), OutputArray _inlierPoints2 = noArray()) const
        {
            cb->computeError(m1, m2, model, err);
            mask.create(err.size(), CV_8U);

            CV_Assert(err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
            const float* errptr = err.ptr<float>();
            uchar* maskptr = mask.ptr<uchar>();
            float t = (float)(thresh * thresh);
            int i, n = (int)err.total(), nz = 0;

            Mat inlierMat = (Mat_<double>(n, 6));
            Mat inlierFlag = (Mat_<int>(n, 1));
            const Point3d* m1ptr = m1.ptr<Point3d>();
            const Point3d* m2ptr = m2.ptr<Point3d>();

            for (i = 0; i < n; i++)
            {
                int f = int(errptr[i] <= t);
                maskptr[i] = (uchar)f;
                inlierFlag.at<int>(i, 0) = 0;
                if (f > 0)
                {
                    inlierMat.at<double>(i, 0) = m1ptr[i].x;
                    inlierMat.at<double>(i, 1) = m1ptr[i].y;
                    inlierMat.at<double>(i, 2) = m1ptr[i].z;
                    inlierMat.at<double>(i, 3) = m2ptr[i].x;
                    inlierMat.at<double>(i, 4) = m2ptr[i].y;
                    inlierMat.at<double>(i, 5) = m2ptr[i].z;
                    inlierFlag.at<int>(i, 0) = 1;
                }
                nz += f;
            }
            if (_inlierPoints1.needed() && _inlierPoints2.needed())
            {
                if (nz <= 0)
                    return 0;
                _inlierPoints1.create(nz, 3, CV_64F);
                _inlierPoints2.create(nz, 3, CV_64F);
                Mat inlierPoints1 = _inlierPoints1.getMat();
                Mat inlierPoints2 = _inlierPoints2.getMat();

                int count = 0;
                for (int k = 0; k < inlierMat.rows; k++)
                {
                    if (inlierFlag.at<int>(k, 0) == 1)
                    {
                        inlierPoints1.at<double>(count, 0) = inlierMat.at<double>(k, 0);
                        inlierPoints1.at<double>(count, 1) = inlierMat.at<double>(k, 1);
                        inlierPoints1.at<double>(count, 2) = inlierMat.at<double>(k, 2);
                        inlierPoints2.at<double>(count, 0) = inlierMat.at<double>(k, 3);
                        inlierPoints2.at<double>(count, 1) = inlierMat.at<double>(k, 4);
                        inlierPoints2.at<double>(count, 2) = inlierMat.at<double>(k, 5);
                        count++;
                    }
                }
                inlierPoints1.copyTo(_inlierPoints1);
                inlierPoints2.copyTo(_inlierPoints2);
            }
            return nz;
        }

        bool getSubset(const Mat& m1, const Mat& m2,
            Mat& ms1, Mat& ms2, RNG& rng,
            int maxAttempts = 1000) const
        {
            cv::AutoBuffer<int> _idx(modelPoints);
            int* idx = _idx;
            int i = 0, j, k, iters = 0;
            int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
            int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
            int esz1 = (int)m1.elemSize1() * d1, esz2 = (int)m2.elemSize1() * d2;
            int count1 = m1.checkVector(d1), count2 = m2.checkVector(d2);
            const int* m1ptr = m1.ptr<int>(), * m2ptr = m2.ptr<int>();

            ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
            ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

            int* ms1ptr = ms1.ptr<int>(), * ms2ptr = ms2.ptr<int>();

            CV_Assert(count1 >= modelPoints && count1 == count2);
            CV_Assert((esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0);
            esz1 /= sizeof(int);
            esz2 /= sizeof(int);

            for (; iters < maxAttempts; iters++)
            {
                for (i = 0; i < modelPoints && iters < maxAttempts; )
                {
                    int idx_i = 0;
                    for (;;)
                    {
                        idx_i = idx[i] = rng.uniform(0, count1);
                        for (j = 0; j < i; j++)
                            if (idx_i == idx[j])
                                break;
                        if (j == i)
                            break;
                    }
                    for (k = 0; k < esz1; k++)
                        ms1ptr[i * esz1 + k] = m1ptr[idx_i * esz1 + k];
                    for (k = 0; k < esz2; k++)
                        ms2ptr[i * esz2 + k] = m2ptr[idx_i * esz2 + k];
                    i++;
                }
                if (i == modelPoints && !cb->checkSubset(ms1, ms2, i))
                    continue;
                break;
            }

            return i == modelPoints && iters < maxAttempts;
        }

        bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _inlierPoints1 = noArray(), OutputArray _inlierPoints2 = noArray(), OutputArray _mask = noArray()) const CV_OVERRIDE
        {
            bool result = false;
            Mat m1 = _m1.getMat(), m2 = _m2.getMat();
            Mat err, mask, model, bestModel, ms1, ms2;

            int iter, niters = MAX(maxIters, 1);
            int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
            int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
            int count1 = m1.checkVector(d1), count2 = m2.checkVector(d2), maxGoodCount = 0;

            RNG rng((uint64)-1);

            CV_Assert(cb);
            CV_Assert(confidence > 0 && confidence < 1);

            CV_Assert(count1 >= 0 && count2 == count1);
            if (count1 < modelPoints)
                return false;

            Mat bestMask0, bestMask;

            if (_mask.needed())
            {
                _mask.create(count1, 1, CV_8U, -1, true);
                bestMask0 = bestMask = _mask.getMat();
                CV_Assert((bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count1);
            }
            else
            {
                bestMask.create(count1, 1, CV_8U);
                bestMask0 = bestMask;
            }

            if (count1 == modelPoints)
            {
                int nmodels = cb->runKernel(m1, m2, model);
                if (nmodels <= 0)
                    return false;
                CV_Assert(model.rows % nmodels == 0);
                Size modelSize(model.cols, model.rows / nmodels);

                //std::cout << model << std::endl;

                for (int i = 0; i < nmodels; i++)
                {
                    Mat model_i = model.rowRange(i * modelSize.height, (i + 1) * modelSize.height);
                    int goodCount = findInliers(m1, m2, model_i, err, mask, threshold);

                    if (goodCount > MAX(maxGoodCount, modelPoints - 1))
                    {
                        std::swap(mask, bestMask);
                        model_i.copyTo(bestModel);
                        maxGoodCount = goodCount;
                        niters = RANSACUpdateNumIters(confidence, (double)(count1 - goodCount) / count1, modelPoints, niters);
                    }
                }

                bestModel.copyTo(_model);
                std::cout << bestModel << std::endl;
                findInliers(m1, m2, bestModel, err, mask, threshold, _inlierPoints1, _inlierPoints2);
                bestMask.setTo(Scalar::all(1));
                return true;
            }

            for (iter = 0; iter < niters; iter++)
            {
                std::cout << "iter:" << iter << std::endl;
                int i, nmodels;
                if (count1 > modelPoints)
                {
                    bool found = getSubset(m1, m2, ms1, ms2, rng, 10000);
                    if (!found)
                    {
                        if (iter == 0)
                            return false;
                        break;
                    }
                }

                nmodels = cb->runKernel(ms1, ms2, model);
                if (nmodels <= 0)
                    continue;
                CV_Assert(model.rows % nmodels == 0);
                Size modelSize(model.cols, model.rows / nmodels);

                for (i = 0; i < nmodels; i++)
                {
                    Mat model_i = model.rowRange(i * modelSize.height, (i + 1) * modelSize.height);
                    int goodCount = findInliers(m1, m2, model_i, err, mask, threshold);

                    if (goodCount > MAX(maxGoodCount, modelPoints - 1))
                    {
                        std::swap(mask, bestMask);
                        model_i.copyTo(bestModel);
                        maxGoodCount = goodCount;
                        niters = RANSACUpdateNumIters(confidence, (double)(count1 - goodCount) / count1, modelPoints, niters);
                    }
                }
            }

            if (maxGoodCount > 0)
            {
                if (bestMask.data != bestMask0.data)
                {
                    if (bestMask.size() == bestMask0.size())
                        bestMask.copyTo(bestMask0);
                    else
                        transpose(bestMask, bestMask0);
                }
                bestModel.copyTo(_model);
                findInliers(m1, m2, bestModel, err, mask, threshold, _inlierPoints1, _inlierPoints2);
                result = true;
            }
            else
                _model.release();

            return result;
        }

        void setCallback(const Ptr<PointSetRegistrator::Callback>& _cb) CV_OVERRIDE { cb = _cb; }

        Ptr<PointSetRegistrator::Callback> cb;
        int modelPoints;
        double threshold;
        double confidence;
        int maxIters;
    };

    Ptr<PointSetRegistrator> createRANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb,
        int _modelPoints, double _threshold, double _confidence, int _maxIters)
    {
        return Ptr<PointSetRegistrator>(new RANSACPointSetRegistrator(_cb, _modelPoints, _threshold, _confidence, _maxIters));
    }

} // namespace cv
