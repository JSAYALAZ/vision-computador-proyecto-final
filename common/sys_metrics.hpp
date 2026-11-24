// =============================================================================
//  sys_metrics.hpp — Métricas del sistema multiplataforma
//  RAM del proceso (RSS) y uso de CPU del proceso.
//  Linux / Android : /proc/self/status (VmRSS) y /proc/self/stat
//  Windows         : GetProcessMemoryInfo / GetProcessTimes
//  macOS           : task_info (MACH_TASK_BASIC_INFO)
// =============================================================================
#pragma once

#include <chrono>
#include <string>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
  #include <psapi.h>
#elif defined(__APPLE__)
  #include <mach/mach.h>
  #include <sys/resource.h>
#else
  #include <sys/resource.h>
  #include <unistd.h>
#endif

namespace metrics {

// Memoria residente del proceso en MB
inline double ramMB() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
#elif defined(__APPLE__)
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
        return info.resident_size / (1024.0 * 1024.0);
    return 0.0;
#else
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {          // "VmRSS:   123456 kB"
            std::istringstream ss(line.substr(6));
            double kb = 0; ss >> kb;
            return kb / 1024.0;
        }
    }
    return 0.0;
#endif
}

// Tiempo de CPU consumido por el proceso (usuario + sistema), en segundos
inline double processCpuSeconds() {
#if defined(_WIN32)
    FILETIME c, e, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0.0;
    auto toSec = [](FILETIME ft) {
        ULARGE_INTEGER li; li.LowPart = ft.dwLowDateTime; li.HighPart = ft.dwHighDateTime;
        return li.QuadPart / 1e7;                    // unidades de 100 ns
    };
    return toSec(k) + toSec(u);
#else
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6 +
           ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6;
#endif
}

// % de CPU del proceso entre dos llamadas (puede pasar de 100% con varios núcleos)
class CpuMeter {
public:
    double sample() {
        const auto now = std::chrono::steady_clock::now();
        const double cpu = processCpuSeconds();
        double pct = 0.0;
        if (initialized_) {
            const double wall = std::chrono::duration<double>(now - lastWall_).count();
            if (wall > 0) pct = 100.0 * (cpu - lastCpu_) / wall;
        }
        lastWall_ = now; lastCpu_ = cpu; initialized_ = true;
        return pct;
    }
private:
    std::chrono::steady_clock::time_point lastWall_;
    double lastCpu_ = 0.0;
    bool initialized_ = false;
};

// Media móvil exponencial para que los números en pantalla no "tiemblen"
struct Ema {
    double value = 0.0; bool init = false; double alpha = 0.1;
    double push(double x) { value = init ? alpha * x + (1 - alpha) * value : x; init = true; return value; }
};

} // namespace metrics
