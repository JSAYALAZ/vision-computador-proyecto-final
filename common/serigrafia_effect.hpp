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
//   3. FILTRO PROPIO: kernel radial de coseno alzado (Hann) con cv::filter2D,
//      "difusión de tinta" que simplifica las formas antes de cuantizar
//   4. Limpieza fuerte del tono: mediana + apertura/cierre en gris (manchas limpias)
//   5. Cuantización a n niveles (LUT)
//   6. Laplaciano + umbral sobre las manchas -> contorno entre tintas  [Cálculo]
//   7. Morfología: cierre + apertura + dilatación             (contorno de tinta)
//   8. PALETA FIJA de alto contraste en una sola LUT + contorno negro con
//      máscara (manipulación de píxeles)
// =============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <algorithm>
#include <array>
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
    int colorLevels       = 4;     // número de tintas de la paleta (2..8)
    int cleanK            = 5;     // limpieza del tono: mediana + apertura/cierre (px)
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
                                  "Tono limpio (kernel propio)", "Posterizada", "Final"};
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

// Paleta fija de serigrafía (BGR), ordenada de oscuro a claro por luminancia
// Y = 0.299R + 0.587G + 0.114B, para que la imagen se siga "leyendo" igual:
//   tinta, azul profundo, rojo, magenta, verde azulado, naranja, amarillo, crema
inline const std::array<cv::Vec3b, 8>& popPalette() {
    static const std::array<cv::Vec3b, 8> pal = {{
        {40, 25, 20},    // Y≈ 27  tinta azul-negra
        {140, 50, 30},   // Y≈ 54  azul profundo
        {45, 35, 210},   // Y≈ 89  rojo intenso
        {140, 60, 230},  // Y≈120  magenta
        {170, 170, 40},  // Y≈131  verde azulado
        {30, 120, 245},  // Y≈147  naranja
        {40, 215, 250},  // Y≈205  amarillo
        {215, 240, 250}, // Y≈236  crema (papel)
    }};
    return pal;
}

// LUT de posterización + paleta (cuantización y coloreado en un solo paso):
//   nivel(v) = ⌊v·n/256⌋ ∈ {0..n-1}
//   color(v) = paleta[ round(nivel·7/(n-1)) ]
// Con n = 4: tinta, rojo, naranja, crema. Se eligen colores repartidos en la
// paleta (siempre el más oscuro y el más claro) para máximo contraste.
inline cv::Mat makePaletteLUT(int n) {
    n = std::clamp(n, 2, 8);
    const auto& pal = popPalette();
    cv::Mat lut(1, 256, CV_8UC3);
    for (int v = 0; v < 256; ++v) {
        const int level = std::min(n - 1, v * n / 256);
        const int idx = static_cast<int>(std::lround(level * 7.0 / (n - 1)));
        lut.at<cv::Vec3b>(0, v) = pal[idx];
    }
    return lut;
}

// LUT de niveles (1 canal): v -> nivel·255/(n-1). Imagen escalonada de las
// manchas, usada para sacar el contorno entre tintas.
inline cv::Mat makeLevelLUT(int n) {
    n = std::clamp(n, 2, 8);
    cv::Mat lut(1, 256, CV_8UC1);
    for (int v = 0; v < 256; ++v)
        lut.at<uchar>(0, v) = static_cast<uchar>(std::min(n - 1, v * n / 256) * 255 / (n - 1));
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

        // (3) Tono (luminancia) sobre el que trabaja el resto del pipeline
        cv::cvtColor(bil_, gray_, cv::COLOR_BGR2GRAY);

        // (5) FILTRO PROPIO: difusión de tinta (kernel de Hann circular) sobre el tono
        cv::filter2D(gray_, soft_, -1, kernel_, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

        // (6) Limpieza fuerte ANTES de cuantizar, para manchas limpias sin "salpicaduras":
        //     mediana: quita puntos aislados conservando el borde de las manchas;
        //     apertura en gris (erosión→dilatación): borra manchitas CLARAS pequeñas;
        //     cierre en gris (dilatación→erosión): borra manchitas OSCURAS pequeñas.
        cv::medianBlur(soft_, toneTmp_, makeOdd(prm.medianK, 3));
        cv::morphologyEx(toneTmp_, tone_, cv::MORPH_OPEN, seClean_);
        cv::morphologyEx(tone_, toneTmp_, cv::MORPH_CLOSE, seClean_);
        toneTmp_.copyTo(tone_);

        // (7) Posterización: n niveles de tono. levels_ = nivel·255/(n-1) (gris
        //     escalonado) sirve para el contorno; la paleta da el color final.
        cv::LUT(tone_, levelLut_, levels_);

        // (8) Contorno sobre las MANCHAS ya cuantizadas (no sobre la imagen con
        //     detalle): el Laplaciano ∇²I = ∂²I/∂x² + ∂²I/∂y² solo responde donde
        //     cambia la tinta, así no aparecen salpicaduras de textura.
        //     |∇²I| > umbral = borde entre dos tintas.
        cv::Laplacian(levels_, lap16_, CV_16S, 3);
        cv::convertScaleAbs(lap16_, lapAbs_);
        cv::threshold(lapAbs_, edgesRaw_, prm.edgeThresh, 255, cv::THRESH_BINARY);

        //     Contorno morfológico:
        //     cierre (dilatación→erosión) une los trazos cortados,
        //     apertura (erosión→dilatación) borra los puntos sueltos,
        //     dilatación final engrosa la línea como tinta de serigrafía.
        cv::morphologyEx(edgesRaw_, edgesTmp_, cv::MORPH_CLOSE, seClose_);
        cv::morphologyEx(edgesTmp_, edgesRaw_, cv::MORPH_OPEN, seOpen_);
        cv::dilate(edgesRaw_, edges_, seDil_);

        // (9) Paleta fija: una LUT de 3 canales hace cuantización + coloreado.
        cv::cvtColor(tone_, tone3_, cv::COLOR_GRAY2BGR);
        cv::LUT(tone3_, paletteLut_, poster_);

        //     Pantalla de contorno: donde hay borde se imprime la tinta más oscura
        poster_.copyTo(out);
        out.setTo(cv::Scalar(popPalette()[0]), edges_);
    }

    // Devuelve (como BGR) la etapa pedida para visualizar pasos intermedios
    void stage(int s, const M& in, const M& finalOut, M& dst) {
        switch (s) {
            case ST_ORIGINAL:    in.copyTo(dst); return;
            case ST_ECUALIZADA:  eq_.copyTo(dst); return;
            case ST_BILATERAL:   bil_.copyTo(dst); return;
            case ST_BORDES:      cv::cvtColor(edges_, dst, cv::COLOR_GRAY2BGR); return;
            case ST_DIFUSION:    cv::cvtColor(tone_, dst, cv::COLOR_GRAY2BGR); return;
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
            makePaletteLUT(n).copyTo(paletteLut_);  // RAM -> (V)RAM una sola vez
            makeLevelLUT(n).copyTo(levelLut_);
            lastLevels_ = n;
        }
        const int ck = makeOdd(prm.cleanK, 3);
        if (ck != lastClean_) {
            seClean_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ck, ck));
            lastClean_ = ck;
        }
    }

    cv::Mat kernel_, seOpen_, seClose_, seDil_, seClean_;
    M paletteLut_, levelLut_;
    std::vector<M> ch_;
    M ycc_, yEq_, eq_, bil_, gray_, levels_, lap16_, lapAbs_, edgesRaw_, edgesTmp_, edges_;
    M soft_, toneTmp_, tone_, tone3_, poster_;

    int lastK_ = -1, lastMorph_ = -1, lastLevels_ = -1, lastClean_ = -1;
};

} // namespace serigrafia
