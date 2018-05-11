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
    private Bitmap bmp;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        btnProc = (Button) findViewById(R.id.btn_gray_process);
        imageView = (ImageView) findViewById(R.id.image_view);
        bmp = BitmapFactory.decodeResource(getResources(), R.drawable.testpic1);
        imageView.setImageBitmap(bmp);
        btnProc.setOnClickListener(this);
    }

    static {
        System.loadLibrary("native-lib");
    }
    @Override
    public void onClick(View v) {

        int w = bmp.getWidth();
        int h = bmp.getHeight();
        int[] pixels = new int[w*h];
        bmp.getPixels(pixels, 0, w, 0, 0, w, h);
        double[] resultInt = FeatureNdkManager.featuresCal(pixels, w, h);
        for (int i = 0; i < resultInt.length; i++) {
            Log.d("MainActivity", "feature:" + resultInt[i]);
        }
        double result = FeatureNdkManager.featuresResult(resultInt, resultInt);
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
