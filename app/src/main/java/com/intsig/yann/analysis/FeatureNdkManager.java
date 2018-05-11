package com.intsig.yann.analysis;

/**
 * Created by wo on 2018/5/11.
 */

public class FeatureNdkManager {

    static {
        System.loadLibrary("native-lib");
    }
    public static native double[] featuresCal(int[] pixels, int w, int h);

    public static native double featuresResult(double[] featuresNormal, double[] featuresTest);

}
