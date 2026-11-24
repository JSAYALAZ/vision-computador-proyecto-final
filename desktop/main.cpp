// =============================================================================
//  main.cpp — App de ESCRITORIO (C++ / OpenCV / OpenCL T-API / VTK cv::viz)
//  Proyecto Integrador Visión Artificial — Parte I
//  Integrantes: Angélica Panamá, José Ayala
//
//  Uso:   ./plumilla_desktop            (cámara 0)
//         ./plumilla_desktop 1          (cámara 1)
//         ./plumilla_desktop video.mp4  (archivo de video)
//
//  Ventanas:
//   * "Serigrafia 3D (VTK)"   -> cubo 3D con el video procesado como TEXTURA
//                              dinámica en sus caras (no se usa cv::imshow
//                              para el video). Se rota con el mouse.
//   * "Controles"           -> trackbars + panel de métricas (ms CPU vs GPU,
//                              FPS, RAM, %CPU, dispositivo OpenCL).
//
//  Teclas (con cualquiera de las dos ventanas activa):
//   m : cambiar modo  CPU (cv::Mat) / GPU (cv::UMat) / AMBOS (comparativa)
//   a : activar/desactivar rotación automática del cubo
//       (no se usa 'k': en la ventana VTK esa tecla exporta la escena a .obj)
//   f : invertir verticalmente las texturas (si se ven al revés)
//   b : benchmark automático (barrido del tamaño del kernel) -> benchmark_sweep.csv
//   ESC (en Controles) o q/e (en la ventana 3D): salir
// =============================================================================
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/viz.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "serigrafia_effect.hpp"
#include "sys_metrics.hpp"

using Clock = std::chrono::high_resolution_clock;

namespace {

enum Mode { MODE_CPU = 0, MODE_GPU = 1, MODE_BOTH = 2 };
const char* modeName(int m) { return m == MODE_CPU ? "CPU (cv::Mat)" : m == MODE_GPU ? "GPU OpenCL (cv::UMat)" : "AMBOS (comparativa)"; }

const char* WIN_CTRL = "Controles";

// Estado compartido con el callback de teclado de VTK
struct AppState {
    int mode = MODE_BOTH;
    bool autoRotate = true;
    bool runSweep = false;
    bool flipTex = false;   // tecla 'f': si la textura sale invertida en su VTK
};

// Una sola función de teclas para las dos ventanas. Devuelve true si hay que salir.
// Se normaliza el código: con el backend Qt/GTK, cv::waitKey puede traer bits de
// modificadores (Bloq Num, Shift...) por encima del byte bajo, y 'M' == 'm'.
bool handleKey(AppState& st, int key) {
    if (key < 0) return false;
    key = std::tolower(key & 0xFF);
    switch (key) {
        case 27:  return true;
        case 'm':
            st.mode = (st.mode + 1) % 3;
            std::cout << "[MODO] " << modeName(st.mode) << std::endl;
            break;
        case 'a':
            st.autoRotate = !st.autoRotate;
            std::cout << "[ROTACION] " << (st.autoRotate ? "automatica" : "solo mouse") << std::endl;
            break;
        case 'b': st.runSweep = true; break;
        case 'f': st.flipTex = !st.flipTex; break;
        default: break;
    }
    return false;
}

void vizKeyboard(const cv::viz::KeyboardEvent& ev, void* cookie) {
    if (ev.action != cv::viz::KeyboardEvent::KEY_DOWN) return;
    const int code = ev.code ? static_cast<unsigned char>(ev.code)
                             : (ev.symbol.size() == 1 ? ev.symbol[0] : -1);
    if (code != 27) handleKey(*static_cast<AppState*>(cookie), code);   // ESC en VTK: lo maneja VTK
}

// ----------------------------------------------------------------------------
// Medición CPU: pipeline con cv::Mat
// ----------------------------------------------------------------------------
double timeCpu(serigrafia::SerigrafiaPipeline<cv::Mat>& pipe, const cv::Mat& in, cv::Mat& out,
               const serigrafia::Params& p) {
    const auto t0 = Clock::now();
    pipe.run(in, out, p);
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ----------------------------------------------------------------------------
// Medición GPU: subida RAM->VRAM + pipeline con cv::UMat + sincronización.
// IMPORTANTE: las llamadas sobre UMat se encolan de forma ASÍNCRONA en la
// cola de comandos OpenCL. Sin cv::ocl::finish() el cronómetro se detendría
// antes de que la GPU termine y mediríamos tiempos falsamente pequeños.
// ----------------------------------------------------------------------------
double timeGpu(serigrafia::SerigrafiaPipeline<cv::UMat>& pipe, const cv::Mat& in, cv::UMat& uIn,
               cv::UMat& uOut, const serigrafia::Params& p) {
    const auto t0 = Clock::now();
    in.copyTo(uIn);                 // RAM -> VRAM
    pipe.run(uIn, uOut, p);
    cv::ocl::finish();              // esperar a que la GPU termine
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ----------------------------------------------------------------------------
// Benchmark automático: barre el tamaño del kernel propio y guarda un CSV
// para las gráficas del informe (tools/plot_benchmark.py).
// ----------------------------------------------------------------------------
// `progress` se llama antes de cada tamaño de kernel para redibujar las ventanas
// (si no, la app se ve congelada durante varios segundos).
void runSweep(const cv::Mat& frame, serigrafia::Params base, bool haveOcl,
              const std::function<void(const std::string&)>& progress) {
    serigrafia::SerigrafiaPipeline<cv::Mat> cpu;
    serigrafia::SerigrafiaPipeline<cv::UMat> gpu;
    cv::Mat out; cv::UMat uIn, uOut;
    std::ofstream csv("benchmark_sweep.csv");
    csv << "kernel,bilateral_d,cpu_ms,gpu_ms,speedup,ram_mb\n";
    const int reps = 15, warmup = 3;

    std::cout << "\n[BENCHMARK] Barrido de kernel propio (" << frame.cols << "x" << frame.rows << ")\n";
    for (int k = 3; k <= 61; k += 4) {
        progress("BENCHMARK en curso: kernel " + std::to_string(k) + "x" + std::to_string(k) +
                 "  (" + std::to_string((k - 3) / 4 + 1) + "/15)");
        base.kernelK = k;
        double sc = 0, sg = 0;
        for (int i = 0; i < warmup; ++i) {           // calentamiento: compila kernels OpenCL
            timeCpu(cpu, frame, out, base);
            if (haveOcl) timeGpu(gpu, frame, uIn, uOut, base);
        }
        for (int i = 0; i < reps; ++i) {
            sc += timeCpu(cpu, frame, out, base);
            if (haveOcl) sg += timeGpu(gpu, frame, uIn, uOut, base);
        }
        const double mc = sc / reps, mg = haveOcl ? sg / reps : 0.0;
        const double sp = mg > 0 ? mc / mg : 0.0;
        csv << k << "," << base.bilateralD << "," << mc << "," << mg << "," << sp << "," << metrics::ramMB() << "\n";
        std::cout << "  k=" << std::setw(2) << k << "  CPU " << std::fixed << std::setprecision(2) << mc
                  << " ms   GPU " << mg << " ms   speedup x" << sp << "\n";
    }
    std::cout << "[BENCHMARK] Guardado en benchmark_sweep.csv\n\n";
}

// ----------------------------------------------------------------------------
// Panel 2D de métricas (NO muestra el video, solo números y barras)
// ----------------------------------------------------------------------------
cv::Mat drawPanel(int mode, double msCpu, double msGpu, double fps, double ram, double cpuPct,
                  const std::string& dev, const serigrafia::Params& p, int stage) {
    cv::Mat panel(300, 560, CV_8UC3, cv::Scalar(30, 30, 30));
    auto txt = [&](const std::string& s, int y, cv::Scalar c = cv::Scalar(230, 230, 230), double sc = 0.55) {
        cv::putText(panel, s, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, sc, c, 1, cv::LINE_AA);
    };
    std::ostringstream o;
    txt(std::string("Modo: ") + modeName(mode), 25, cv::Scalar(120, 220, 255), 0.6);
    txt("OpenCL: " + dev, 50, cv::Scalar(180, 180, 180), 0.45);

    o << std::fixed << std::setprecision(2) << "CPU: " << msCpu << " ms";  txt(o.str(), 85);  o.str("");
    o << "GPU: " << msGpu << " ms";                                         txt(o.str(), 110); o.str("");
    if (msGpu > 0 && msCpu > 0) { o << "Speedup GPU: x" << msCpu / msGpu; txt(o.str(), 135, cv::Scalar(120, 255, 120)); o.str(""); }
    o << std::setprecision(1) << "FPS: " << fps << "   RAM: " << ram << " MB   CPU proc: " << cpuPct << " %";
    txt(o.str(), 165); o.str("");
    o << "Kernel propio: " << serigrafia::makeOdd(p.kernelK, 3) << "x" << serigrafia::makeOdd(p.kernelK, 3)
      << "   Bilateral d=" << serigrafia::makeOdd(p.bilateralD);
    txt(o.str(), 190); o.str("");
    txt(std::string("Etapa (cara frontal): ") + serigrafia::stageName(stage), 215);

    // Barras comparativas (escala relativa al mayor tiempo)
    const double mx = std::max({msCpu, msGpu, 1.0});
    const int W = 380;
    cv::rectangle(panel, cv::Rect(80, 235, int(W * msCpu / mx), 20), cv::Scalar(60, 90, 230), cv::FILLED);
    cv::rectangle(panel, cv::Rect(80, 265, int(W * msGpu / mx), 20), cv::Scalar(80, 200, 80), cv::FILLED);
    txt("CPU", 250); txt("GPU", 280);
    return panel;
}

// Pose de una cara del cubo: rotación R y traslación R*(0,0,h)
cv::Affine3d facePose(const cv::Vec3d& rvec, double h) {
    cv::Affine3d R(rvec);
    return cv::Affine3d(R.rotation(), R.rotation() * cv::Vec3d(0, 0, h));
}

// Si hay varias plataformas OpenCL, prefiere el driver del fabricante (NVIDIA,
// AMD, Intel) antes que PoCL: PoCL también expone la GPU NVIDIA a través de CUDA,
// pero algunos kernels de OpenCV fallan ahí (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES).
// Se respeta OPENCV_OPENCL_DEVICE si el usuario ya la definió.
void preferVendorOpenCL() {
    if (std::getenv("OPENCV_OPENCL_DEVICE")) return;
    std::vector<cv::ocl::PlatformInfo> platforms;
    cv::ocl::getPlatfomsInfo(platforms);
    for (const auto& p : platforms) {
        const std::string name = p.name();
        if (name.find("Portable Computing Language") != std::string::npos) continue;
        for (int i = 0; i < p.deviceNumber(); ++i) {
            cv::ocl::Device d;
            p.getDevice(d, i);
            if (d.type() & cv::ocl::Device::TYPE_GPU) {
                const std::string sel = name + ":GPU:";
#ifdef _WIN32
                _putenv_s("OPENCV_OPENCL_DEVICE", sel.c_str());
#else
                setenv("OPENCV_OPENCL_DEVICE", sel.c_str(), 0);
#endif
                return;
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    // ---------------- 1. Fuente de video ----------------
    cv::VideoCapture cap;
    std::string src = argc > 1 ? argv[1] : "0";
    if (src.size() <= 2 && std::isdigit(static_cast<unsigned char>(src[0]))) cap.open(std::stoi(src));
    else cap.open(src);
    if (!cap.isOpened()) { std::cerr << "No se pudo abrir la fuente: " << src << "\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // ---------------- 2. OpenCL / T-API ----------------
    preferVendorOpenCL();
    cv::ocl::setUseOpenCL(true);
    const bool haveOcl = cv::ocl::haveOpenCL() && cv::ocl::useOpenCL();
    std::string devName = "NO DISPONIBLE";
    if (haveOcl) {
        const cv::ocl::Device& d = cv::ocl::Device::getDefault();
        devName = d.name() + " (" + d.vendorName() + ")";
        std::cout << "OpenCL activo en: " << devName << "\n"
                  << "  Unidades de cómputo: " << d.maxComputeUnits()
                  << "  Memoria global: " << d.globalMemSize() / (1024 * 1024) << " MB\n";
    } else {
        std::cout << "ADVERTENCIA: OpenCL no disponible, el modo GPU caerá a CPU.\n";
    }

    // ---------------- 3. Controles (trackbars) ----------------
    serigrafia::Params prm;
    AppState st;
    cv::namedWindow(WIN_CTRL, cv::WINDOW_AUTOSIZE);
    int tbKernel = prm.kernelK, tbBil = prm.bilateralD, tbEdge = prm.edgeThresh;
    int tbSpace = prm.colorLevels, tbMorph = prm.morphK, tbStage = serigrafia::ST_FINAL;
    cv::createTrackbar("Kernel propio k", WIN_CTRL, &tbKernel, 61);   // satura la carga O(k^2)
    cv::createTrackbar("Bilateral d",     WIN_CTRL, &tbBil, 31);      // O(d^2)
    cv::createTrackbar("Umbral bordes",   WIN_CTRL, &tbEdge, 150);
    cv::createTrackbar("Niveles color",   WIN_CTRL, &tbSpace, 8);     // tintas por canal
    cv::createTrackbar("Morfologia",      WIN_CTRL, &tbMorph, 9);
    cv::createTrackbar("Etapa",           WIN_CTRL, &tbStage, serigrafia::ST_COUNT - 1);

    // ---------------- 4. Escena 3D VTK (cv::viz) ----------------
    cv::viz::Viz3d win("Serigrafia 3D (VTK)");
    win.setBackgroundColor(cv::viz::Color(25, 25, 35), cv::viz::Color(70, 60, 50));
    win.registerKeyboardCallback(vizKeyboard, &st);

    const double side = 1.0, h = side / 2.0;
    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar::all(0));
    // Seis caras: id, rotación local (Rodrigues) -> textura que muestra
    struct Face { std::string id; cv::Vec3d rvec; int stage; };
    const std::vector<Face> faces = {
        {"frente",    {0, 0, 0},              -1},                      // etapa elegida con trackbar
        {"atras",     {0, CV_PI, 0},          serigrafia::ST_ORIGINAL},
        {"derecha",   {0, CV_PI / 2, 0},      serigrafia::ST_POSTERIZADA},
        {"izquierda", {0, -CV_PI / 2, 0},     serigrafia::ST_BORDES},
        {"arriba",    {-CV_PI / 2, 0, 0},     serigrafia::ST_DIFUSION},
        {"abajo",     {CV_PI / 2, 0, 0},      serigrafia::ST_ECUALIZADA},
    };
    std::vector<cv::Affine3d> localPose;
    for (const auto& f : faces) {
        win.showWidget(f.id, cv::viz::WImage3D(blank, cv::Size2d(side, side)));
        localPose.push_back(facePose(f.rvec, h));
    }
    win.showWidget("ejes", cv::viz::WCoordinateSystem(0.3));
    win.showWidget("hud", cv::viz::WText("", cv::Point(10, 10), 14, cv::viz::Color::white()));
    win.setViewerPose(cv::viz::makeCameraPose(cv::Vec3d(1.8, 1.2, 2.6), cv::Vec3d(0, 0, 0), cv::Vec3d(0, 1, 0)));
    // setViewerPose deja el punto focal de VTK a 1 unidad delante de la cámara, no en
    // el cubo; el mouse orbita alrededor del foco y el cubo "se desplaza". resetCamera()
    // mantiene la dirección de vista y mueve el foco al centro de la escena (el cubo).
    win.resetCamera();

    // ---------------- 5. Pipelines y buffers ----------------
    serigrafia::SerigrafiaPipeline<cv::Mat>  cpuPipe;
    serigrafia::SerigrafiaPipeline<cv::UMat> gpuPipe;
    cv::Mat frame, outCpu, outShow, stageMat;
    cv::UMat uIn, uOut, uStage;

    metrics::Ema emaCpu, emaGpu, emaFps;
    metrics::CpuMeter cpuMeter;
    double cpuPct = 0;
    std::ofstream live("benchmark_live.csv");
    live << "t_s,mode,kernel,bilateral_d,cpu_ms,gpu_ms,fps,ram_mb,cpu_pct\n";
    const auto tStart = Clock::now();
    auto tPrev = tStart;
    double angle = 0.0;
    int frameCount = 0;

    while (!win.wasStopped()) {
        if (!cap.read(frame) || frame.empty()) {
            // si es archivo, reiniciar al terminar
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!cap.read(frame) || frame.empty()) break;
        }
        if (frame.cols != 640) cv::resize(frame, frame, cv::Size(640, 480));

        // Leer controles
        prm.kernelK      = serigrafia::makeOdd(tbKernel, 3);
        prm.bilateralD   = serigrafia::makeOdd(tbBil, 1);
        prm.edgeThresh   = std::max(tbEdge, 1);
        prm.colorLevels  = std::max(tbSpace, 2);
        prm.morphK       = serigrafia::makeOdd(tbMorph, 1);

        // ---------- Procesamiento según modo ----------
        double msCpu = 0, msGpu = 0;
        const bool doCpu = st.mode == MODE_CPU || st.mode == MODE_BOTH;
        const bool doGpu = (st.mode == MODE_GPU || st.mode == MODE_BOTH);
        if (doCpu) msCpu = emaCpu.push(timeCpu(cpuPipe, frame, outCpu, prm));
        if (doGpu) msGpu = emaGpu.push(timeGpu(gpuPipe, frame, uIn, uOut, prm));
        if (!doCpu) msCpu = emaCpu.value;   // conservar último valor para comparar
        if (!doGpu) msGpu = emaGpu.value;

        // ---------- FPS / recursos ----------
        const auto now = Clock::now();
        const double dt = std::chrono::duration<double>(now - tPrev).count();
        tPrev = now;
        const double fps = emaFps.push(dt > 0 ? 1.0 / dt : 0.0);
        if (++frameCount % 15 == 0) cpuPct = cpuMeter.sample();
        const double ram = metrics::ramMB();
        live << std::chrono::duration<double>(now - tStart).count() << "," << st.mode << "," << prm.kernelK << ","
             << prm.bilateralD << "," << msCpu << "," << msGpu << "," << fps << "," << ram << "," << cpuPct << "\n";

        // ---------- Texturas para el cubo 3D ----------
        const bool fromGpu = doGpu;   // en modo AMBOS se muestra el resultado de la GPU
        for (size_t i = 0; i < faces.size(); ++i) {
            const int s = faces[i].stage < 0 ? tbStage : faces[i].stage;
            if (fromGpu) { gpuPipe.stage(s, uIn, uOut, uStage); uStage.copyTo(stageMat); }  // VRAM -> RAM
            else         { cpuPipe.stage(s, frame, outCpu, stageMat); }
            if (st.flipTex) cv::flip(stageMat, stageMat, 0);   // corrige orientación si hace falta
            win.getWidget(faces[i].id).cast<cv::viz::WImage3D>().setImage(stageMat);
        }

        // ---------- Rotación del cubo (transformación afín 3D) ----------
        if (st.autoRotate) angle += 0.01;
        const cv::Affine3d cubePose(cv::Vec3d(0.35 * std::sin(angle * 0.7), angle, 0.0));
        for (size_t i = 0; i < faces.size(); ++i)
            win.setWidgetPose(faces[i].id, cubePose * localPose[i]);   // global = cubo · local

        // ---------- HUD en la ventana 3D ----------
        std::ostringstream hud;
        hud << std::fixed << std::setprecision(2) << modeName(st.mode) << " | CPU " << msCpu << " ms | GPU "
            << msGpu << " ms | x" << (msGpu > 0 ? msCpu / msGpu : 0.0) << " | FPS " << std::setprecision(1) << fps
            << " | RAM " << ram << " MB | k=" << prm.kernelK;
        win.getWidget("hud").cast<cv::viz::WText>().setText(hud.str());

        cv::imshow(WIN_CTRL, drawPanel(st.mode, msCpu, msGpu, fps, ram, cpuPct, devName, prm, tbStage));

        // ---------- Eventos ----------
        win.spinOnce(1, true);
        if (handleKey(st, cv::waitKey(1))) break;
        if (st.runSweep) {
            auto progress = [&](const std::string& msg) {
                cv::Mat busy(300, 560, CV_8UC3, cv::Scalar(30, 30, 30));
                cv::putText(busy, msg, cv::Point(12, 140), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                            cv::Scalar(120, 220, 255), 1, cv::LINE_AA);
                cv::putText(busy, "La app sigue viva: espere a que termine el barrido", cv::Point(12, 175),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
                cv::imshow(WIN_CTRL, busy);
                cv::waitKey(1);
                win.getWidget("hud").cast<cv::viz::WText>().setText(msg);
                win.spinOnce(1, true);
            };
            runSweep(frame, prm, haveOcl, progress);
            st.runSweep = false;
            tPrev = Clock::now();   // no contar el barrido como un frame lento
        }
    }
    return 0;
}
