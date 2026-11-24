// =============================================================================
//  serigrafia_effect.hpp — Efecto "Posterización Cromática con Contorno
//  Morfológico" (estilo Serigrafía / Pop-Print)
//  Proyecto Integrador Visión Artificial - Parte I
//  Integrantes: Angélica Panamá, José Ayala
//
//  Este header se comparte entre la app de ESCRITORIO y la app ANDROID (JNI).
//  El pipeline está escrito UNA sola vez como plantilla:
//      SerigrafiaPipeline<cv::Mat>   -> CPU (RAM)
//      SerigrafiaPipeline<cv::UMat>  -> GPU vía OpenCL / API Transparente (VRAM)
//  Así la comparación CPU vs GPU es justa: mismas operaciones, mismo orden.
//
//  Idea: una serigrafía imprime pocas tintas planas y saturadas, una por
//  pantalla, y encima una pantalla negra con el contorno. Se imita así:
//   1. Ecualización de histograma en la luminancia Y (YCrCb)  [Estadística]
//   2. Filtro bilateral a color                               [Suavizado]
//   3. Mediana + Laplaciano + umbral -> contornos             [Cálculo]
//   4. Morfología: cierre + apertura + dilatación             (contorno de tinta)
//   5. FILTRO PROPIO: kernel radial de coseno alzado (Hann) con cv::filter2D,
//      "difusión de tinta" que simplifica las formas antes de cuantizar
//   6. Saturación pop (LUT sobre el canal S de HSV) + posterización por canal
//      (LUT de cuantización) + contorno negro con máscara (manipulación de píxeles)
// =============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace serigrafia {

// -----------------------------------------------------------------------------
// Parámetros controlables en tiempo real (trackbars / SeekBars)
// -----------------------------------------------------------------------------
struct Params {
    int bilateralD        = 9;     // diámetro de la ventana bilateral (carga O(d²))
    double bilateralSigma = 60.0;
    int medianK           = 5;     // ventana de la mediana (impar)
    int kernelK           = 15;    // tamaño del kernel propio (impar) -> "saturar" carga
    int edgeThresh        = 45;    // umbral sobre |Laplaciano|
    int morphK            = 3;     // grosor del contorno (elemento estructurante)
    int colorLevels       = 4;     // tintas por canal (2..8) -> hasta niveles³ colores
    double saturation     = 1.8;   // factor de saturación "pop"
    double eqStrength     = 0.5;   // 0 = sin ecualizar, 1 = ecualización completa
};

enum Stage {
    ST_ORIGINAL = 0,
    ST_ECUALIZADA,
    ST_BILATERAL,
    ST_BORDES,
    ST_DIFUSION,
    ST_POSTERIZADA,
    ST_FINAL,
    ST_COUNT
};

inline const char* stageName(int s) {
    static const char* names[] = {"Original", "Ecualizada", "Bilateral", "Contorno",
                                  "Difusion (kernel propio)", "Posterizada", "Final"};
    return (s >= 0 && s < ST_COUNT) ? names[s] : "?";
}

inline int makeOdd(int k, int minimum = 1) {
    k = std::max(k, minimum);
    return (k % 2 == 0) ? k + 1 : k;
}

// -----------------------------------------------------------------------------
// FILTRO PROPIO: kernel radial de coseno alzado (ventana de Hann circular)
//
//   Coordenadas centradas:  x, y ∈ [-c, c],  c = (k-1)/2,   R = c + 1
//   r = √(x² + y²)
//   K(x,y) = ½·(1 + cos(π·r/R))   si r < R,   0 en otro caso;   luego K /= ΣK
//
//  * Soporte circular: difumina igual en todas las direcciones (isotrópico),
//    como la tinta que se expande en el papel. Las esquinas del k×k valen 0.
//  * Decae suave hasta 0 en el borde, así evita el "anillado" de un filtro
//    de caja (box) y las manchas quedan redondeadas.
//  * Σ K = 1 -> conserva el color medio (pasa-bajos).
//  Al aplicarlo ANTES de posterizar, las zonas de color quedan como manchas
//  grandes y simples (lo que hace la plantilla de una serigrafía).
//  Costo O(k²) por píxel y por canal -> control para saturar la CPU.
// -----------------------------------------------------------------------------
inline cv::Mat makeInkKernel(int k) {
    k = makeOdd(k, 3);
    const double c = (k - 1) / 2.0;
    const double R = c + 1.0;
    cv::Mat K(k, k, CV_32F);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            const double x = j - c, y = i - c;
            const double r = std::sqrt(x * x + y * y);
            const double val = (r < R) ? 0.5 * (1.0 + std::cos(CV_PI * r / R)) : 0.0;
            K.at<float>(i, j) = static_cast<float>(val);
            sum += val;
        }
    }
    K /= sum;                       // normalización: Σ K = 1
    return K;
}

// LUT de posterización (cuantización uniforme por canal, manipulación de píxeles):
//   paso Δ = 256 / n,   q(v) = ⌊v/Δ⌋·Δ + Δ/2   -> n valores por canal
inline cv::Mat makePosterLUT(int n) {
    n = std::clamp(n, 2, 8);
    const double step = 256.0 / n;
    cv::Mat lut(1, 256, CV_8UC1);
    for (int v = 0; v < 256; ++v) {
        const int idx = std::min(n - 1, static_cast<int>(v / step));
        lut.at<uchar>(0, v) = cv::saturate_cast<uchar>(idx * step + step / 2.0);
    }
    return lut;
}

// LUT HSV de 3 canales: H y V sin cambio; S con "zona muerta" + ganancia pop:
//   S' = 0                       si S < s0   (los grises quedan neutros: sin halos
//                                             de color por ruido en paredes/fondos)
//   S' = min(255, (S - s0)·f')   si S ≥ s0,  con f' = factor·255/(255 - s0)
inline cv::Mat makeSaturationLUT(double factor, int s0 = 28) {
    cv::Mat lut(1, 256, CV_8UC3);
    const double gain = factor * 255.0 / (255.0 - s0);
    for (int v = 0; v < 256; ++v) {
        const double s = (v < s0) ? 0.0 : (v - s0) * gain;
        lut.at<cv::Vec3b>(0, v) = cv::Vec3b(static_cast<uchar>(v),
                                            cv::saturate_cast<uchar>(s),
                                            static_cast<uchar>(v));
    }
    return lut;
}

// -----------------------------------------------------------------------------
// Pipeline plantilla. M = cv::Mat (CPU) o cv::UMat (OpenCL).
// Todos los buffers intermedios son miembros para NO reservar memoria en cada
// frame (en UMat esto evita copias RAM<->VRAM y re-asignaciones en la GPU).
// -----------------------------------------------------------------------------
template <typename M>
class SerigrafiaPipeline {
public:
    // in: BGR 8UC3.  out: BGR 8UC3 con el efecto final.
    void run(const M& in, M& out, const Params& prm) {
        prepare(prm);

        // (1) Ecualización solo de la luminancia: Y de YCrCb. Si se ecualizara
        //     cada canal B, G, R por separado, los colores cambiarían de tono.
        //     s_k = (L-1) * Σ_{j<=k} p(r_j)
        cv::cvtColor(in, ycc_, cv::COLOR_BGR2YCrCb);
        cv::split(ycc_, ch_);
        //     Ecualización PARCIAL: Y' = α·eq(Y) + (1-α)·Y. Con α = 1 el ruido del
        //     sensor en zonas planas (paredes) se amplifica y aparece como grano.
        cv::equalizeHist(ch_[0], yEq_);
        cv::addWeighted(yEq_, prm.eqStrength, ch_[0], 1.0 - prm.eqStrength, 0.0, ch_[0]);
        cv::merge(ch_, ycc_);
        cv::cvtColor(ycc_, eq_, cv::COLOR_YCrCb2BGR);

        // (2) Bilateral a color: aplana texturas sin mezclar colores en los bordes.
        //     w = Gs(||p-q||) * Gr(||I_p - I_q||)
        cv::bilateralFilter(eq_, bil_, makeOdd(prm.bilateralD), prm.bilateralSigma, prm.bilateralSigma);

        // (3) Contornos: gris -> mediana (ruido impulsivo) -> Laplaciano (2da derivada)
        //     ∇²I = ∂²I/∂x² + ∂²I/∂y²  -> |∇²I| > umbral = borde
        cv::cvtColor(bil_, gray_, cv::COLOR_BGR2GRAY);
        cv::medianBlur(gray_, med_, makeOdd(prm.medianK, 3));
        cv::Laplacian(med_, lap16_, CV_16S, 3);
        cv::convertScaleAbs(lap16_, lapAbs_);
        cv::threshold(lapAbs_, edgesRaw_, prm.edgeThresh, 255, cv::THRESH_BINARY);

        // (4) Contorno morfológico:
        //     cierre (dilatación→erosión) une primero los trazos cortados,
        //     apertura (erosión→dilatación) borra luego los puntos sueltos,
        //     dilatación final engrosa la línea como tinta de serigrafía.
        //     (En orden inverso la apertura borraría también las líneas finas.)
        cv::morphologyEx(edgesRaw_, edgesTmp_, cv::MORPH_CLOSE, seClose_);
        cv::morphologyEx(edgesTmp_, edgesRaw_, cv::MORPH_OPEN, seOpen_);
        cv::dilate(edgesRaw_, edges_, seDil_);

        // (5) FILTRO PROPIO: difusión de tinta (kernel de Hann circular)
        cv::filter2D(bil_, soft_, -1, kernel_, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

        // (6a) Saturación pop: HSV, S' = min(255, S·factor) por LUT, de vuelta a BGR
        cv::cvtColor(soft_, hsv_, cv::COLOR_BGR2HSV);
        cv::LUT(hsv_, satLut_, hsvSat_);
        cv::cvtColor(hsvSat_, vivid_, cv::COLOR_HSV2BGR);

        // (6b) Posterización: cada canal B, G, R se cuantiza a n tintas
        cv::LUT(vivid_, posterLut_, poster_);

        // (6c) Pantalla de contorno: donde hay borde se imprime tinta casi negra
        poster_.copyTo(out);
        out.setTo(cv::Scalar(25, 20, 20), edges_);
    }

    // Devuelve (como BGR) la etapa pedida para visualizar pasos intermedios
    void stage(int s, const M& in, const M& finalOut, M& dst) {
        switch (s) {
            case ST_ORIGINAL:    in.copyTo(dst); return;
            case ST_ECUALIZADA:  eq_.copyTo(dst); return;
            case ST_BILATERAL:   bil_.copyTo(dst); return;
            case ST_BORDES:      cv::cvtColor(edges_, dst, cv::COLOR_GRAY2BGR); return;
            case ST_DIFUSION:    soft_.copyTo(dst); return;
            case ST_POSTERIZADA: poster_.copyTo(dst); return;
            default:             finalOut.copyTo(dst); return;
        }
    }

private:
    // Recalcula kernel / elementos estructurantes / LUTs solo si cambia un parámetro
    void prepare(const Params& prm) {
        const int k  = makeOdd(prm.kernelK, 3);
        const int mk = makeOdd(prm.morphK, 1);
        const int n  = std::clamp(prm.colorLevels, 2, 8);

        if (k != lastK_) {
            kernel_ = makeInkKernel(k);              // el kernel vive en RAM; OpenCV lo sube a la GPU
            lastK_ = k;
        }
        if (mk != lastMorph_) {
            seOpen_  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            seClose_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            seDil_   = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(mk, mk));
            lastMorph_ = mk;
        }
        if (n != lastLevels_) {
            makePosterLUT(n).copyTo(posterLut_);    // RAM -> (V)RAM una sola vez
            lastLevels_ = n;
        }
        if (prm.saturation != lastSat_) {
            makeSaturationLUT(prm.saturation).copyTo(satLut_);
            lastSat_ = prm.saturation;
        }
    }

    cv::Mat kernel_, seOpen_, seClose_, seDil_;
    M posterLut_, satLut_;
    std::vector<M> ch_;
    M ycc_, yEq_, eq_, bil_, gray_, med_, lap16_, lapAbs_, edgesRaw_, edgesTmp_, edges_;
    M soft_, hsv_, hsvSat_, vivid_, poster_;

    int lastK_ = -1, lastMorph_ = -1, lastLevels_ = -1;
    double lastSat_ = -1.0;
};

} // namespace serigrafia
