package net.johnhany.ndk;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageView;

import com.intsig.yann.analysis.FeatureNdkManager;

public class MainActivity extends AppCompatActivity implements OnClickListener {

    private Button btnProc;
    private ImageView imageView;
    private Bitmap normalBitmap;
    private Bitmap testBitmap;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        btnProc = (Button) findViewById(R.id.btn_gray_process);
        imageView = (ImageView) findViewById(R.id.image_view);
        normalBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.normal);
        testBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test);
        imageView.setImageBitmap(normalBitmap);
        onClick(null);
    }

    static {
        System.loadLibrary("native-lib");
    }
    @Override
    public void onClick(View v) {

        int w = normalBitmap.getWidth();
        int h = normalBitmap.getHeight();
        int[] pixels = new int[w*h];
        normalBitmap.getPixels(pixels, 0, w, 0, 0, w, h);
        double[] resultNormal = FeatureNdkManager.featuresCal(pixels, w, h);

        int w1 = testBitmap.getWidth();
        int h1 = testBitmap.getHeight();
        int[] pixels1 = new int[w1*h1];
        testBitmap.getPixels(pixels1, 0, w1, 0, 0, w1, h1);
        double[] resultTest = FeatureNdkManager.featuresCal(pixels1, w1, h1);

//        for (int i = 0; i < resultNormal.length; i++) {
//            Log.d("MainActivity", "feature:" + resultNormal[i]);
//        }
        double result = FeatureNdkManager.featuresResult(resultNormal, resultTest);
        Log.d("MainActivity", "result:" + result);
//        Bitmap resultImg = Bitmap.createBitmap(w, h, Config.ARGB_8888);
//        resultImg.setPixels(resultInt, 0, w, 0, 0, w, h);
//        imageView.setImageBitmap(resultImg);
    }

    @Override
    public void onResume(){
        super.onResume();
    }
}
