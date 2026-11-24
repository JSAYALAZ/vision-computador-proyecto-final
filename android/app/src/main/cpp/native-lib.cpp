// =============================================================================
//  native-lib.cpp — Puente JNI para la app ANDROID
//  Proyecto Integrador Visión Artificial — Parte I
//  Integrantes: Angélica Panamá, José Ayala
//
//  Java entrega la dirección nativa de cv::Mat (getNativeObjAddr) del frame de
//  la cámara. Aquí se procesa con el MISMO header del escritorio
//  (common/serigrafia_effect.hpp), usando cv::Mat (CPU) o cv::UMat (OpenCL).
// =============================================================================
#include <jni.h>
#include <android/log.h>
#include <string>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>

#include "serigrafia_effect.hpp"
#include "sys_metrics.hpp"

#define TAG "PlumillaNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

using Clock = std::chrono::high_resolution_clock;

namespace {
serigrafia::SerigrafiaPipeline<cv::Mat>  gCpu;
serigrafia::SerigrafiaPipeline<cv::UMat> gGpu;
cv::Mat  gBgr, gOutCpu, gStage;
cv::UMat gUIn, gUOut, gUStage;
metrics::Ema gEmaCpu, gEmaGpu;
metrics::CpuMeter gCpuMeter;
double gCpuPct = 0;
int gFrame = 0;

enum Mode { MODE_CPU = 0, MODE_GPU = 1, MODE_BOTH = 2 };
}

extern "C" JNIEXPORT jstring JNICALL
Java_ec_edu_ups_plumilla_MainActivity_nativeInit(JNIEnv* env, jobject) {
    cv::ocl::setUseOpenCL(true);
    std::string info;
    if (cv::ocl::haveOpenCL() && cv::ocl::useOpenCL()) {
        const cv::ocl::Device& d = cv::ocl::Device::getDefault();
        info = "OpenCL: " + d.name() + " | " + d.vendorName() +
               " | CUs=" + std::to_string(d.maxComputeUnits());
    } else {
        info = "OpenCL NO disponible (modo GPU usara CPU)";
    }
    LOGI("%s", info.c_str());
    return env->NewStringUTF(info.c_str());
}

// Devuelve double[] { msCpu, msGpu, ramMB, cpuPct }
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_ec_edu_ups_plumilla_MainActivity_nativeProcess(
        JNIEnv* env, jobject,
        jlong rgbaAddr, jlong outAddr,
        jint mode, jint stage, jboolean split,
        jint kernel, jint bilateral, jint edge, jint spacing, jint morph) {

    cv::Mat& rgba = *reinterpret_cast<cv::Mat*>(rgbaAddr);
    cv::Mat& out  = *reinterpret_cast<cv::Mat*>(outAddr);

    serigrafia::Params p;
    p.kernelK      = serigrafia::makeOdd(kernel, 3);
    p.bilateralD   = serigrafia::makeOdd(bilateral, 1);
    p.edgeThresh   = std::max<int>(edge, 1);
    p.colorLevels  = std::clamp<int>(spacing, 2, 8);
    p.morphK       = serigrafia::makeOdd(morph, 1);

    cv::cvtColor(rgba, gBgr, cv::COLOR_RGBA2BGR);   // la cámara entrega RGBA

    const bool doCpu = mode == MODE_CPU || mode == MODE_BOTH;
    const bool doGpu = mode == MODE_GPU || mode == MODE_BOTH;

    if (doCpu) {
        auto t0 = Clock::now();
        gCpu.run(gBgr, gOutCpu, p);
        gEmaCpu.push(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    if (doGpu) {
        auto t0 = Clock::now();
        gBgr.copyTo(gUIn);          // RAM -> memoria del dispositivo OpenCL
        gGpu.run(gUIn, gUOut, p);
        cv::ocl::finish();          // sincronizar: las colas OpenCL son asíncronas
        gEmaGpu.push(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }

    // Etapa a mostrar (resultado GPU si se calculó, si no el de CPU)
    if (doGpu) { gGpu.stage(stage, gUIn, gUOut, gUStage); gUStage.copyTo(gStage); }
    else       { gCpu.stage(stage, gBgr, gOutCpu, gStage); }

    cv::Mat stageRgba;
    cv::cvtColor(gStage, stageRgba, cv::COLOR_BGR2RGBA);

    if (split) {
        // Pantalla dividida: original | efecto (o etapa intermedia elegida)
        out.create(rgba.size(), CV_8UC4);
        const int w2 = rgba.cols / 2;
        cv::Mat left = out(cv::Rect(0, 0, w2, rgba.rows));
        cv::Mat right = out(cv::Rect(w2, 0, rgba.cols - w2, rgba.rows));
        cv::resize(rgba, left, left.size());
        cv::resize(stageRgba, right, right.size());
        cv::line(out, {w2, 0}, {w2, rgba.rows}, cv::Scalar(255, 200, 0, 255), 2);
    } else {
        stageRgba.copyTo(out);
    }

    if (++gFrame % 15 == 0) gCpuPct = gCpuMeter.sample();

    jdoubleArray res = env->NewDoubleArray(4);
    const jdouble vals[4] = {gEmaCpu.value, gEmaGpu.value, metrics::ramMB(), gCpuPct};
    env->SetDoubleArrayRegion(res, 0, 4, vals);
    return res;
}
