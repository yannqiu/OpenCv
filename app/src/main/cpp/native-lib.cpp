#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


extern "C"
{
    double ave(Mat roi)
    {
        int i, j;
        int sum = 0, number = 0;
        //int x = roi.width, y = roi.height;//x表示目标区域列数，y表示目标区域行数
        //int gg[1000];

        for (i = 0; i < roi.rows; i++)
        {
            uchar* data = roi.ptr<uchar>(i);//获得一行数值的指针
            for (j = 0; j < roi.cols; j++)
            {
                sum = sum + data[j];
                number++;
            }
        }
        double average = sum / number;
        return average;
    }
    //Mat中元素求方差
    double var(Mat roi, double average)
    {

        double variance = 0;
        int i, j;
        for (i = 0; i < roi.rows; i++)
        {
            uchar* data = roi.ptr<uchar>(i);
            for (j = 0; j < roi.cols; j++)
            {
                variance = variance + (data[j] - average)*(data[j] - average);
            }
        }
        variance = variance / (roi.rows*roi.cols);
        return variance;
    }
    //计算灰度共生矩阵
    void feature_texture(Mat roi, int flag, double* glcm_features)
    {
        int i, j;
        int GLCM_class = 16;//灰度量化等级
        int GLCM_dis = 1;//灰度共生矩阵统计距离
        int sum = 0;//图像中灰度对的总数

        //得到量化后的图象像素数组
        int* histImg = new int[roi.cols*roi.rows];
        for (i = 0; i <roi.rows; i++)
        {
            uchar* data = roi.ptr<uchar>(i);
            for (j = 0; j < roi.cols; j++)
            {
                histImg[i*roi.cols + j] = data[j] * GLCM_class / 256;
            }
        }

        double* GLCM = new double[GLCM_class*GLCM_class];//共生矩阵数组
        //GLCM赋初值
        for (i = 0; i < GLCM_class; i++)
        {
            for (j = 0; j < GLCM_class; j++)
            {
                GLCM[i*GLCM_class + j] = 0;
            }
        }
        int lx, mm;
        switch (flag)
        {
            case 1:
                //0度 得到水平方向GLCM
                for (i = 0; i < roi.rows; i++)
                {
                    for (j = 0; j < roi.cols; j++)
                    {
                        lx = histImg[i*roi.cols + j];
                        if (j + GLCM_dis >= 0 && j + GLCM_dis < roi.cols)
                        {
                            mm = histImg[i*roi.cols + j + GLCM_dis];
                            GLCM[lx*GLCM_class + mm]++;
                        }
                    }
                }
                break;
            case 2:
                //45度 得到右下对角方向GLCM
                for (i = 0; i < roi.rows; i++)
                {
                    for (j = 0; j < roi.cols; j++)
                    {
                        lx = histImg[i*roi.cols + j];
                        if (i + GLCM_dis >= 0 && i + GLCM_dis < roi.rows && j + GLCM_dis >= 0 && j + GLCM_dis < roi.cols)
                        {
                            mm = histImg[(i + GLCM_dis)*roi.cols + j + GLCM_dis];
                            GLCM[lx*GLCM_class + mm]++;
                        }
                    }
                }
                break;
            case 3:
                //90度 得到竖直方向GLCM
                for (i = 0; i < roi.rows; i++)
                {
                    for (j = 0; j < roi.cols; j++)
                    {
                        lx = histImg[i*roi.cols + j];
                        if (i + GLCM_dis >= 0 && i + GLCM_dis < roi.rows)
                        {
                            mm = histImg[(i + GLCM_dis)*roi.cols + j];
                            GLCM[lx*GLCM_class + mm]++;
                        }
                    }
                }
                break;
            case 4:
                //135度 得到左下对角方向GLCM
                for (i = 0; i < roi.rows; i++)
                {
                    for (j = 0; j < roi.cols; j++)
                    {
                        lx = histImg[i*roi.cols + j];
                        if (i + GLCM_dis >= 0 && i + GLCM_dis < roi.rows && j + GLCM_dis >= 0 && j - GLCM_dis < roi.cols)
                        {
                            mm = histImg[(i + GLCM_dis)*roi.cols + j - GLCM_dis];
                            GLCM[lx*GLCM_class + mm]++;
                        }
                    }
                }
                break;
        }

        //GLCM归一化
        sum = (roi.cols - GLCM_dis)*(roi.rows - GLCM_dis);
        for (i = 0; i < GLCM_class; i++)
        {
            for (j = 0; j < GLCM_class; j++)
            {
                GLCM[i*GLCM_class + j] = GLCM[i*GLCM_class + j] / sum;
            }
        }

        //计算GLCM特征值
        double contrast = 0, Asm = 0, entropy = 0, correlation = 0;
        double u1 = 0, u2 = 0, d1 = 0, d2 = 0, temp = 0;//计算自相关性所用变量
        for (i = 0; i < GLCM_class; i++)//计算u1
        {
            temp = 0;
            for (j = 0; j < GLCM_class; j++)
            {
                temp += GLCM[i*GLCM_class + j];
            }
            u1 = (i + 1)*temp;
        }
        for (j = 0; j < GLCM_class; j++)//计算u2
        {
            temp = 0;
            for (i = 0; i < GLCM_class; i++)
            {
                temp += GLCM[i*GLCM_class + j];
            }
            u2 = (j + 1)*temp;
        }
        for (i = 0; i < GLCM_class; i++)//计算d1
        {
            temp = 0;
            for (j = 0; j < GLCM_class; j++)
            {
                temp += GLCM[i*GLCM_class + j];
            }
            d1 += (i + 1 - u1)*(i + 1 - u1)*temp;
        }
        for (j = 0; j < GLCM_class; j++)//计算d2
        {
            temp = 0;
            for (i = 0; i < GLCM_class; i++)
            {
                temp += GLCM[i*GLCM_class + j];
            }
            d2 += (j + 1 - u2)*(j + 1 - u2)*temp;
        }
        for (i = 0; i < GLCM_class; i++)
        {
            for (j = 0; j < GLCM_class; j++)
            {
                contrast += (i - j)*(i - j)*GLCM[i*GLCM_class + j];//对比度
                Asm += GLCM[i*GLCM_class + j] * GLCM[i*GLCM_class + j];//能量
                if (GLCM[i*GLCM_class + j] > 0)//熵
                {
                    entropy -= GLCM[i*GLCM_class + j] * log10(double(GLCM[i*GLCM_class + j]));
                }
                correlation += (i + 1)*(j + 1)*GLCM[i*GLCM_class + j];//自相关性第一部分
            }
        }
        correlation = (correlation - u1*u2) / (d1*d2);//自相关性

        glcm_features[0] = contrast;
        glcm_features[1] = Asm;
        glcm_features[2] = entropy;
        glcm_features[3] = correlation;
        delete histImg, GLCM;
    }

    JNIEXPORT jdoubleArray JNICALL Java_com_intsig_yann_analysis_FeatureNdkManager_featuresCal(JNIEnv *env, jobject instance,
                                                                                  jintArray buf, jint w, jint h) {

        jint *cbuf;
        jboolean ptfalse = false;
        cbuf = env->GetIntArrayElements(buf, &ptfalse);
        if(cbuf == NULL){
            return 0;
        }
        Mat imgData(h, w, CV_8UC4, (unsigned char*)cbuf);
        // begin 计算值
        //得到区域BGR各分量
        Mat image_B, image_G, image_R;//正常区域的BGR各分量
        int ch_1[] = { 0, 0 };//从某个通道到一个通道        //获取Blue分量
        image_B.create(imgData.size(), imgData.depth());
        mixChannels(&imgData, 1, &image_B, 1, ch_1, 1);//将第一个通道(也就是蓝色)的数复制到b中，0索引数组
        int ch_2[] = { 1, 0 };//获取Green分量
        image_G.create(imgData.size(), imgData.depth());
        mixChannels(&imgData, 1, &image_G, 1, ch_2, 1);
        int ch_3[] = { 2, 0 };//获取Red分量
        image_R.create(imgData.size(), imgData.depth());
        mixChannels(&imgData, 1, &image_R, 1, ch_3, 1);

        //得到区域HSV各分量
        Mat image_hsv;
        cvtColor(imgData, image_hsv, COLOR_BGR2HSV_FULL);

        Mat image_h, image_s, image_v;//正常区域的hsv各分量
        image_h.create(image_hsv.size(), image_hsv.depth());//获取hue色调分量    //hue初始化为与hsv大小深度一样的矩阵，色调的度量是用角度表示的，红绿蓝之间相差120度，反色相差180
        mixChannels(&image_hsv, 1, &image_h, 1, ch_1, 1);//将hsv第一个通道(也就是色调)的数复制到hue中，0索引数组
        image_s.create(image_hsv.size(), image_hsv.depth());//获取saturation饱和度分量
        mixChannels(&image_hsv, 1, &image_s, 1, ch_2, 1);
        image_v.create(image_hsv.size(), image_hsv.depth());//获取value亮度分量
        mixChannels(&image_hsv, 1, &image_v, 1, ch_3, 1);


        //计算正常图像与疲劳图像HSV与BGR各分量均值及方差,并归一化
        double features[28];
        features[0] = ave(image_B) / 255;
        features[1] = var(image_B, features[0]) / 65025;
        features[2] = ave(image_G) / 255;
        features[3] = var(image_G, features[2]) / 65025;
        features[4] = ave(image_R) / 255;
        features[5] = var(image_R, features[4]) / 65025;
        features[6] = ave(image_h) / 255;
        features[7] = var(image_h, features[6]) / 65025;
        features[8] = ave(image_s) / 255;
        features[9] = var(image_s, features[8]) / 65025;
        features[10] = ave(image_v) / 255;
        features[11] = var(image_v, features[10]) / 65025;



        Mat image_gray;
        cvtColor(imgData, image_gray, COLOR_BGR2GRAY);//将图像转化为灰度图

        //计算GLCM特征值，1 2 3 4分别表示4个角度矩阵
        double glcm_features[4];
        int i, j;
        for (i = 1; i < 5; i++)
        {
            feature_texture(image_gray, i, glcm_features);
            for (j = 0; j < 4; j++)
            {
                features[8 + 4 * i + j] = glcm_features[j];
            }
        }
        // end 计算值
        jdoubleArray result = env->NewDoubleArray(28);
        env->SetDoubleArrayRegion(result, 0, 28, features);
        env->ReleaseIntArrayElements(buf, cbuf, 0);
        return result;
    }
    //计算得到BP网络输出值
    JNIEXPORT jdouble JNICALL Java_com_intsig_yann_analysis_FeatureNdkManager_featuresResult(JNIEnv *env, jobject instance,
                                                                                jdoubleArray featuresNormal, jdoubleArray featuresTest) {
        jdouble* normal_features = env->GetDoubleArrayElements(featuresNormal, 0);
        jdouble* features = env->GetDoubleArrayElements(featuresTest, 0);

        //计算差值数据
        double diff_features0[28];//未标准化数据
        double diff_features[28];//标准化后数据


        //输入前标准化数据参数y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin
        double Xmin[28] = { -0.14466199999999996, -0.13393300000000002, -0.15032644444444432, -0.16379261111111112, -0.12047888888888902, -0.17515716666666664, -0.023529500000000002, -0.010212780000000001, -0.14994205882352937, -0.084082317647058813, -0.12047888888888902, -0.17517555555555547, -0.07133500000000001, -0.08564780000000001, -0.17885582352941165, -0.0038593200000000012, -0.096570000000000017, -0.079699400000000004, -0.18326588235294106, -0.003880719999999999, -0.079573000000000005, -0.089415700000000015, -0.18362417647058826, -0.0038531299999999997, -0.084923999999999999, -0.079440399999999994, -0.18452135294117633, -0.0038521199999999988 };
        double Xmax[28] = { 0.11764699999999995, 0.111569, 0.10980299999999998, 0.126861, 0.141177, 0.222742, 0.020261405555555563, 0.024117290555555553, 0.094771166666666629, 0.088617000000000029, 0.141177, 0.22274299999999991, 0.224051, 0.13594188235294119, 0.36024499999999993, 0.0051236277777777801, 0.25216, 0.1317355294117647, 0.38076999999999983, 0.0051656166666666677, 0.22768600000000003, 0.13650811764705884, 0.3909729999999999, 0.0051444444444444445, 0.26816899999999999, 0.1319598823529412, 0.39132999999999996, 0.0051407388888888902 };

        for (int i = 0; i < 28; i++)
        {
            diff_features0[i] = features[i] - normal_features[i];
            diff_features[i] = 2 * (diff_features0[i] - Xmin[i]) / (Xmax[i] - Xmin[i]) - 1;//数据标准化
        }


        //输入层到隐层权值
        double w1[280] = { 0.010169671, -0.448248481, 0.278797428, 0.479899852, -0.172882077, 0.596239494, -0.184846469, 0.192539775, 0.37248929, 0.306764676, 0.125005551, 0.110693493, 0.462806515, 0.210162279, -0.06618618, 0.04533851, 0.157770452, 0.613447233, -0.161953019, -0.177817986, -0.514653452, 0.464263827, -0.557963802, -0.049783876, 0.163181263, 0.144690725, -0.313651982, 0.101515847,
                           -0.691243234, -0.471791462, -0.817010218, -1.232152901, 0.750976457, 0.134478216, -1.283366149, -0.483543021, 0.950706529, 0.595943108, 0.214468663, 0.766492761, 0.158050707, 0.067135913, -0.119050854, -0.687845995, 0.273965758, 1.080813025, -0.278582375, -0.799491571, 1.226319998, -0.161130846, 0.762075528, -0.512439778, 0.891330487, 0.147814376, 0.210847645, -0.70880744,
                           0.519108337, 0.323128974, -0.159628847, 0.157446607, 0.424356273, 0.120361822, -0.051075359, 0.428641592, 0.38989393, -0.102792195, -0.220230903, 0.636165222, 0.246851021, -0.923532635, 0.28942046, -0.402516269, 0.097815354, -0.762325295, 0.666385358, 0.102590213, 0.470661299, -0.498104558, 0.686111979, -0.217814591, 0.677331172, -0.200171674, 0.27851328, -0.397301002,
                           -0.704910677, -0.870950812, 0.717866624, 0.12623054, -0.304957121, -0.399275574, 2.056705454, -0.752095488, 0.465489802, 0.5954213, 0.176068602, -0.725335659, 0.600688813, -1.147236052, -0.533385754, 0.313812591, 1.458633179, -1.414370666, 0.452604103, 0.669270027, 0.101423453, -0.315498034, -0.988101156, 0.365822992, 0.5072273, -1.059922887, -0.034250097, 0.095349474,
                           -0.173455907, -0.569408116, 0.382286499, 0.173115786, 0.515891968, -0.451913336, 0.850191253, -0.806506516, 0.143413447, -0.215754425, 0.148033429, -0.185314819, -0.342939865, 0.201189342, -1.187030725, 0.391306022, 0.240183958, -0.485426783, -0.976708501, 0.237235751, -0.081397911, 0.867443633, -0.70556394, -0.215334909, -0.44008475, 0.13619925, -0.674364005, -0.28652199,
                           -0.372658521, -0.402127846, 0.333975618, -0.128403188, 0.407257581, 0.764623047, 0.113446981, -0.854271683, 0.209364223, -0.0685346, 0.775403475, 0.334564955, -1.014357094, -0.082611376, -0.499759435, -0.12967904, -0.80926072, 0.284946987, -0.442941963, 0.489106129, -1.394375407, 0.808926909, -1.037815184, -0.332884128, -0.942293166, 0.074854878, -0.063756509, 0.308319471,
                           0.278433288, 0.34823919, 0.170029175, 0.185672342, 0.236261304, 0.141912675, 0.332530114, 0.419559862, 0.166701878, -0.282365435, 0.063648052, 0.259593614, -0.166037014, -0.596059622, -0.140225437, 0.07168345, 0.312996058, -0.537452775, 0.383304662, -0.157148415, 0.385312593, 0.266587047, -0.333692258, -0.246202814, 0.052764688, -0.090183929, 0.432079646, 0.319545344,
                           0.356018484, 0.210064766, -0.049698202, 0.189036088, -0.409549206, 0.041126739, -1.061954619, -0.05509388, -0.899209131, -0.669880227, -0.345817187, 0.220506211, 0.080278468, 0.634116302, 0.554739942, -0.036974303, 0.274718904, 0.411808494, -0.326479986, 0.083232452, 0.9495606, 0.11866133, 0.681406775, -0.369896622, 0.421849871, 0.852902213, 0.785477184, -0.130605172,
                           -0.342050258, -0.158594971, -0.648864988, -0.425190537, -0.857957091, -0.457956938, -0.894718415, 0.164139505, 0.354573204, 0.233727516, -0.05233372, -0.258391778, -1.240149628, -0.491769008, 0.860069415, 0.656763555, -1.153917524, 0.31455071, 0.560788314, 0.090479003, -0.124690142, -0.403513753, 0.437713562, 0.509826253, -0.179566923, -0.027502873, 0.458671302, 0.093721179,
                           -0.531792581, -0.870878159, -0.060586392, -0.054939014, 0.435416119, 0.214539193, -0.761994422, -0.68790476, 0.515902882, 1.078932286, 0.61058201, 0.428609269, 0.566324859, 0.29142382, 0.289708113, -0.341532053, 0.517454878, -0.018299048, 0.324554317, -0.10440223, 0.179710832, 0.452317365, -0.41109048, -0.279598083, 0.386672173, 0.440953804, -0.621737152, 0.275177289 };
        //输入层到隐层阈值
        double b1[10] = { -1.536692994, 1.728779935, -0.802584202, 0.071213196, -0.273586802, -0.996926955, 0.447331104, 1.152667248, -0.859873171, -1.313159887 };
        //隐层到输出层权值
        double w2[10] = { -1.002979457, 3.062044267, 0.779429543, -2.873914257, -1.064603318, -2.679581076, -0.430557734, 1.529945457, 1.607035235, -2.043083514 };
        //隐层到输出层阈值
        double b2 = 0.652580732;
        double hidden[10] = { 0 }, hidden_out[10] = { 0 };
        double out = 0, output;
        int i, j;
        double temp;
        //hidden = w1*features + b1;
        for (i = 0; i < 10; i++)
        {
            temp = 0;
            for (j = 0; j < 28; j++)
            {
                temp += diff_features[j] * w1[28 * i + j];
            }
            temp = temp + b1[i];
            hidden[i] = temp;
        }
        //transig传递函数
        for (i = 0; i < 10; i++)
        {
            hidden_out[i] = 2 / (1 + exp(-2 * hidden[i])) - 1;
        }
        //out = w2*hidden_out + b2;
        for (i = 0; i < 10; i++)
        {
            out += hidden_out[i] * w2[i];
        }
        out = out + b2;
        //transig传递函数
        output = 2 / (1 + exp(-2 * out)) - 1;

        output = (output + 1) / 2;//数据去标准化
        return ((jdouble)output * 100);
    }
}
