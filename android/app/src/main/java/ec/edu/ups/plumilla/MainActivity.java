// =============================================================================
//  MainActivity.java — UI Android: cámara, controles y métricas
//  Proyecto Integrador Visión Artificial — Parte I
//  Integrantes: Angélica Panamá, José Ayala
//
//  Java solo captura la cámara y lee los controles. TODO el procesamiento
//  ocurre en C++ (native-lib.cpp) a través de JNI.
// =============================================================================
package ec.edu.ups.plumilla;

import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.widget.SwitchCompat;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import java.util.Collections;
import java.util.List;
import java.util.Locale;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "Plumilla";

    // ---- Métodos nativos (C++) ----
    private native String nativeInit();
    private native double[] nativeProcess(long rgbaAddr, long outAddr, int mode, int stage, boolean split,
                                          int kernel, int bilateral, int edge, int spacing, int morph);

    private CameraBridgeViewBase cameraView;
    private TextView txtMetrics;
    private Mat outFrame;
    private String oclInfo = "";

    // Parámetros leídos por el hilo de la cámara (volatile = visibles entre hilos)
    private volatile int mode = 2, stage = 6, kernel = 15, bilateral = 9, edge = 45, spacing = 4, morph = 3;
    private volatile boolean split = true;

    private long lastNs = 0;
    private double fpsEma = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "No se pudo cargar OpenCV", Toast.LENGTH_LONG).show();
            return;
        }
        System.loadLibrary("plumilla");
        oclInfo = nativeInit();
        Log.i(TAG, oclInfo);

        cameraView = findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);
        cameraView.setMaxFrameSize(640, 480);   // misma resolución que en escritorio

        txtMetrics = findViewById(R.id.txt_metrics);
        setupControls();
    }

    private void setupControls() {
        Spinner spMode = findViewById(R.id.sp_mode);
        spMode.setSelection(mode);
        spMode.setOnItemSelectedListener(new SimpleSelect(pos -> mode = pos));

        Spinner spStage = findViewById(R.id.sp_stage);
        spStage.setSelection(stage);
        spStage.setOnItemSelectedListener(new SimpleSelect(pos -> stage = pos));

        SwitchCompat swSplit = findViewById(R.id.sw_split);
        swSplit.setOnCheckedChangeListener((b, checked) -> split = checked);

        bindSeek(R.id.sb_kernel, R.id.lbl_kernel, "Kernel propio k", v -> kernel = v);
        bindSeek(R.id.sb_bil, R.id.lbl_bil, "Bilateral d", v -> bilateral = v);
        bindSeek(R.id.sb_edge, R.id.lbl_edge, "Umbral bordes", v -> edge = v);
        bindSeek(R.id.sb_space, R.id.lbl_space, "Niveles de color", v -> spacing = v);
        bindSeek(R.id.sb_morph, R.id.lbl_morph, "Morfología", v -> morph = v);
    }

    private interface IntSink { void set(int v); }

    private void bindSeek(int seekId, int labelId, String name, IntSink sink) {
        SeekBar sb = findViewById(seekId);
        TextView lbl = findViewById(labelId);
        lbl.setText(name + ": " + sb.getProgress());
        sb.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar s, int v, boolean fromUser) {
                sink.set(v);
                lbl.setText(name + ": " + v);
            }
            @Override public void onStartTrackingTouch(SeekBar s) { }
            @Override public void onStopTrackingTouch(SeekBar s) { }
        });
    }

    private static class SimpleSelect implements AdapterView.OnItemSelectedListener {
        private final IntSink sink;
        SimpleSelect(IntSink s) { sink = s; }
        @Override public void onItemSelected(AdapterView<?> p, android.view.View v, int pos, long id) { sink.set(pos); }
        @Override public void onNothingSelected(AdapterView<?> p) { }
    }

    // ---------------- Ciclo de vida de la cámara ----------------
    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(cameraView);
    }

    @Override public void onResume() { super.onResume(); if (cameraView != null) cameraView.enableView(); }
    @Override public void onPause() { super.onPause(); if (cameraView != null) cameraView.disableView(); }
    @Override public void onDestroy() { super.onDestroy(); if (cameraView != null) cameraView.disableView(); }

    @Override public void onCameraViewStarted(int width, int height) { outFrame = new Mat(); }
    @Override public void onCameraViewStopped() { if (outFrame != null) outFrame.release(); }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        double[] r = nativeProcess(rgba.getNativeObjAddr(), outFrame.getNativeObjAddr(),
                mode, stage, split, kernel, bilateral, edge, spacing, morph);

        // FPS de toda la app (captura + proceso + dibujo)
        long now = System.nanoTime();
        if (lastNs != 0) {
            double fps = 1e9 / (now - lastNs);
            fpsEma = fpsEma == 0 ? fps : 0.1 * fps + 0.9 * fpsEma;
        }
        lastNs = now;

        final String txt = String.format(Locale.US,
                "%s%nCPU : %7.2f ms%nGPU : %7.2f ms%nSpeedup: x%.2f%nFPS : %5.1f%nRAM : %6.1f MB%nCPU proc: %5.1f %%",
                oclInfo, r[0], r[1], r[1] > 0 ? r[0] / r[1] : 0.0, fpsEma, r[2], r[3]);
        runOnUiThread(() -> txtMetrics.setText(txt));
        return outFrame;
    }
}
