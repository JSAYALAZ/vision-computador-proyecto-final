// =============================================================================
//  plumilla_effect.hpp  —  Efecto "Grabado a Plumilla" (pen-and-ink crosshatch)
//  Proyecto Integrador Visión Artificial - Parte I
//  Integrantes: Angélica Panamá, José Ayala
//
//  Este header se comparte entre la app de ESCRITORIO y la app ANDROID (JNI).
//  El pipeline está escrito UNA sola vez como plantilla:
//      PlumillaPipeline<cv::Mat>   -> CPU (RAM)
//      PlumillaPipeline<cv::UMat>  -> GPU vía OpenCL / API Transparente (VRAM)
//  Así la comparación CPU vs GPU es justa: mismas operaciones, mismo orden.
//
//  Etapas (4 filtros vistos en clase + 1 filtro propio):
//   1. Ecualización de histograma          (cv::equalizeHist)      [Estadística]
//   2. Filtro bilateral                    (cv::bilateralFilter)   [Suavizado]
//   3. Mediana + Laplaciano + umbral       (bordes, 2da derivada)  [Cálculo]
//   4. Morfología: apertura + dilatación   (limpieza de bordes)
//   5. FILTRO PROPIO: kernel gaussiano anisotrópico a 45° ("trazo de tinta")
//      aplicado con cv::filter2D + tramado por niveles con operaciones AND/OR/NOT
//   6. Coloración tinta/papel mediante LUT (manipulación de píxeles)
// =============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <array>
#include <cmath>
#include <string>

namespace plumilla {

// -----------------------------------------------------------------------------
// Parámetros controlables en tiempo real (trackbars / SeekBars)
// -----------------------------------------------------------------------------
struct Params {
    int bilateralD     = 9;    // diámetro de la ventana bilateral (carga O(d²))
    double bilateralSigma = 60.0;
    int medianK        = 5;    // ventana de la mediana (impar)
    int strokeK        = 15;   // tamaño del kernel propio (impar) -> "saturar" carga
    int edgeThresh     = 45;   // umbral sobre |Laplaciano|
    int morphK         = 3;    // tamaño del elemento estructurante
    int hatchSpacing   = 9;    // separación entre líneas del tramado (px)
    std::array<int, 4> levels {175, 125, 80, 40}; // umbrales de tono por capa
};

enum Stage {
    ST_ORIGINAL = 0,
    ST_ECUALIZADA,
    ST_BILATERAL,
    ST_BORDES,
    ST_TRAZO,
    ST_TRAMADO,
    ST_FINAL,
    ST_COUNT
};

inline const char* stageName(int s) {
    static const char* names[] = {"Original", "Ecualizada", "Bilateral", "Bordes",
                                  "Trazo (kernel propio)", "Tramado", "Final"};
    return (s >= 0 && s < ST_COUNT) ? names[s] : "?";
}

inline int makeOdd(int k, int minimum = 1) {
    k = std::max(k, minimum);
    return (k % 2 == 0) ? k + 1 : k;
}

// -----------------------------------------------------------------------------
// FILTRO PROPIO: kernel gaussiano anisotrópico orientado a 45°
//
//   Coordenadas centradas:  x, y ∈ [-c, c],  c = (k-1)/2
//   Rotación 45°:            u = (x + y)/√2   (eje a lo largo del trazo)
//                            v = (x - y)/√2   (eje perpendicular al trazo)
//   K(x,y) = exp( -u²/(2σu²) - v²/(2σv²) ) / Σ K
//   σu = k/4  (largo: suaviza A LO LARGO del trazo)
//   σv = max(0.6, k/24)  (angosto: conserva detalle a TRAVÉS del trazo)
//
//  Como Σ K = 1, el brillo medio se conserva (filtro pasa-bajos direccional).
//  Aplicado sobre (tono + grano de papel) convierte el ruido aleatorio en
//  "rayas" diagonales parecidas a trazos de lápiz/tinta. Su costo por píxel
//  es O(k²), por eso es el control ideal para saturar la CPU en el benchmark.
// -----------------------------------------------------------------------------
inline cv::Mat makeStrokeKernel(int k) {
    k = makeOdd(k, 3);
    const double c  = (k - 1) / 2.0;
    const double su = k / 4.0;
    const double sv = std::max(0.6, k / 24.0);
    cv::Mat K(k, k, CV_32F);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            const double x = j - c, y = i - c;
            const double u = (x + y) / std::sqrt(2.0);
            const double v = (x - y) / std::sqrt(2.0);
            const double val = std::exp(-(u * u) / (2 * su * su) - (v * v) / (2 * sv * sv));
            K.at<float>(i, j) = static_cast<float>(val);
            sum += val;
        }
    }
    K /= sum;                       // normalización: Σ K = 1
    return K;
}

// -----------------------------------------------------------------------------
// Patrones de tramado (manipulación directa de píxeles):
//   capa 0: diagonal 45°   -> (x + y) mod s == 0
//   capa 1: diagonal -45°  -> (x - y) mod s == 0
//   capa 2: horizontal     ->  y mod s == 0
//   capa 3: vertical       ->  x mod s == 0
// Mientras más oscuro el tono, más capas se superponen (cross-hatching).
// -----------------------------------------------------------------------------
inline std::array<cv::Mat, 4> makeHatchPatterns(cv::Size sz, int s) {
    s = std::max(s, 2);
    std::array<cv::Mat, 4> p;
    for (auto& m : p) m = cv::Mat::zeros(sz, CV_8UC1);
    for (int y = 0; y < sz.height; ++y) {
        uchar* p0 = p[0].ptr<uchar>(y);
        uchar* p1 = p[1].ptr<uchar>(y);
        uchar* p2 = p[2].ptr<uchar>(y);
        uchar* p3 = p[3].ptr<uchar>(y);
        for (int x = 0; x < sz.width; ++x) {
            if ((x + y) % s == 0)              p0[x] = 255;
            if (((x - y) % s + s) % s == 0)    p1[x] = 255;
            if (y % s == 0)                    p2[x] = 255;
            if (x % s == 0)                    p3[x] = 255;
        }
    }
    return p;
}

// LUT tinta/papel: v=0 -> tinta azul-sepia, v=255 -> papel crema (BGR)
inline cv::Mat makeInkPaperLUT() {
    const cv::Vec3f ink(70, 40, 25), paper(228, 242, 250);
    cv::Mat lut(1, 256, CV_8UC3);
    for (int v = 0; v < 256; ++v) {
        const float t = v / 255.0f;
        cv::Vec3f c = ink + (paper - ink) * t;
        lut.at<cv::Vec3b>(0, v) = cv::Vec3b(cv::saturate_cast<uchar>(c[0]),
                                            cv::saturate_cast<uchar>(c[1]),
                                            cv::saturate_cast<uchar>(c[2]));
    }
    return lut;
}

// -----------------------------------------------------------------------------
// Pipeline plantilla. M = cv::Mat (CPU) o cv::UMat (OpenCL).
// Todos los buffers intermedios son miembros para NO reservar memoria en cada
// frame (en UMat esto evita copias RAM<->VRAM y re-asignaciones en la GPU).
// -----------------------------------------------------------------------------
template <typename M>
class PlumillaPipeline {
public:
    // in: BGR 8UC3.  out: BGR 8UC3 con el efecto final.
    void run(const M& in, M& out, const Params& prm) {
        prepare(in.size(), prm);

        // (0) Gris
        cv::cvtColor(in, gray_, cv::COLOR_BGR2GRAY);

        // (1) Ecualización de histograma: estira el contraste usando la CDF.
        //     s_k = (L-1) * Σ_{j<=k} p(r_j). Atenúa problemas de iluminación.
        cv::equalizeHist(gray_, eq_);

        // (2) Bilateral: suaviza zonas planas pero conserva bordes.
        //     w = Gs(||p-q||) * Gr(|I_p - I_q|)
        cv::bilateralFilter(eq_, bil_, makeOdd(prm.bilateralD), prm.bilateralSigma, prm.bilateralSigma);

        // (3) Bordes: mediana (quita ruido sal/pimienta) + Laplaciano (2da derivada)
        //     ∇²I = ∂²I/∂x² + ∂²I/∂y²  -> |∇²I| > umbral = borde
        cv::medianBlur(bil_, med_, makeOdd(prm.medianK, 3));
        cv::Laplacian(med_, lap16_, CV_16S, 3);
        cv::convertScaleAbs(lap16_, lapAbs_);
        cv::threshold(lapAbs_, edges_, prm.edgeThresh, 255, cv::THRESH_BINARY);

        // (4) Morfología: apertura elimina puntos aislados; dilatación engrosa la línea
        cv::morphologyEx(edges_, edgesTmp_, cv::MORPH_OPEN, seOpen_);
        cv::dilate(edgesTmp_, edges_, seDil_);

        // (5) FILTRO PROPIO: tono + grano -> kernel anisotrópico 45° -> trazo
        cv::addWeighted(bil_, 0.75, grain_, 0.25, 0.0, mix_);
        cv::filter2D(mix_, stroke_, -1, kernel_, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

        //     Tramado por niveles: ink = OR_i ( (trazo < T_i) AND patrón_i )
        inkAcc_.setTo(cv::Scalar::all(0));
        for (int i = 0; i < 4; ++i) {
            cv::compare(stroke_, prm.levels[i], mask_, cv::CMP_LT);
            cv::bitwise_and(mask_, patterns_[i], layer_);
            cv::bitwise_or(inkAcc_, layer_, inkAcc_);
        }
        cv::bitwise_not(inkAcc_, hatch_);              // papel blanco, tinta negra

        //     Combinar contornos + tramado: final = tramado AND NOT(bordes)
        cv::bitwise_not(edges_, edgesInv_);
        cv::bitwise_and(hatch_, edgesInv_, finalGray_);

        // (6) Coloración tinta/papel con LUT (mapeo punto a punto)
        cv::cvtColor(finalGray_, final3_, cv::COLOR_GRAY2BGR);
        cv::LUT(final3_, lut_, out);
    }

    // Devuelve (como BGR) la etapa pedida para visualizar pasos intermedios
    void stage(int s, const M& in, const M& finalOut, M& dst) {
        switch (s) {
            case ST_ORIGINAL:   in.copyTo(dst); return;
            case ST_ECUALIZADA: cv::cvtColor(eq_, dst, cv::COLOR_GRAY2BGR); return;
            case ST_BILATERAL:  cv::cvtColor(bil_, dst, cv::COLOR_GRAY2BGR); return;
            case ST_BORDES:     cv::cvtColor(edges_, dst, cv::COLOR_GRAY2BGR); return;
            case ST_TRAZO:      cv::cvtColor(stroke_, dst, cv::COLOR_GRAY2BGR); return;
            case ST_TRAMADO:    cv::cvtColor(hatch_, dst, cv::COLOR_GRAY2BGR); return;
            default:            finalOut.copyTo(dst); return;
        }
    }

private:
    // Recalcula kernel / patrones / grano solo cuando cambia el tamaño o un parámetro
    void prepare(cv::Size sz, const Params& prm) {
        const int k  = makeOdd(prm.strokeK, 3);
        const int sp = std::max(prm.hatchSpacing, 2);
        const int mk = makeOdd(prm.morphK, 1);

        if (k != lastK_) {
            kernel_ = makeStrokeKernel(k);           // el kernel vive en RAM; OpenCV lo sube a la GPU
            lastK_ = k;
        }
        if (mk != lastMorph_) {
            seOpen_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
            seDil_  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(mk, mk));
            lastMorph_ = mk;
        }
        if (sz != lastSize_ || sp != lastSpacing_) {
            auto pats = makeHatchPatterns(sz, sp);
            for (int i = 0; i < 4; ++i) pats[i].copyTo(patterns_[i]);  // RAM -> (V)RAM una sola vez
            lastSpacing_ = sp;
        }
        if (sz != lastSize_) {
            cv::Mat g(sz, CV_8UC1);
            cv::RNG rng(7);                          // semilla fija: grano estable entre frames
            rng.fill(g, cv::RNG::UNIFORM, 0, 256);
            g.copyTo(grain_);
            inkAcc_.create(sz, CV_8UC1);
            lastSize_ = sz;
        }
        if (lut_.empty()) makeInkPaperLUT().copyTo(lut_);
    }

    cv::Mat kernel_, seOpen_, seDil_;
    M lut_;
    std::array<M, 4> patterns_;
    M grain_;
    M gray_, eq_, bil_, med_, lap16_, lapAbs_, edges_, edgesTmp_, edgesInv_;
    M mix_, stroke_, mask_, layer_, inkAcc_, hatch_, finalGray_, final3_;

    cv::Size lastSize_ {0, 0};
    int lastK_ = -1, lastSpacing_ = -1, lastMorph_ = -1;
};

} // namespace plumilla
