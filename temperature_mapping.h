#ifndef GPU_BURN_TEMPERATURE_MAPPING_H
#define GPU_BURN_TEMPERATURE_MAPPING_H

#include <cstdlib>
#include <string>
#include <vector>

namespace temperature_mapping {

inline std::string gpuUuid(const unsigned char *bytes) {
    static const char hex[] = "0123456789abcdef";
    std::string result = "GPU-";
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10)
            result += '-';
        result += hex[bytes[i] >> 4];
        result += hex[bytes[i] & 0x0f];
    }
    return result;
}

inline std::string trim(const std::string &value) {
    const std::string::size_type first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";
    const std::string::size_type last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

// Apply a CSV temperature reading to the matching CUDA worker by GPU UUID,
// rather than assuming CUDA logical indices match nvidia-smi physical indices.
inline bool applyTemperatureRow(const std::string &line,
                                const std::vector<std::string> &workerUuids,
                                std::vector<int> *temperatures) {
    const std::string::size_type comma = line.find(',');
    if (comma == std::string::npos || temperatures == nullptr ||
        temperatures->size() != workerUuids.size())
        return false;

    const std::string uuid = trim(line.substr(0, comma));
    const std::string temperatureText = trim(line.substr(comma + 1));
    if (temperatureText.empty())
        return false;

    char *end = nullptr;
    const long value = std::strtol(temperatureText.c_str(), &end, 10);
    if (end == temperatureText.c_str() || *end != '\0' ||
        value < 0 || value > 150)
        return false;

    for (std::size_t i = 0; i < workerUuids.size(); ++i) {
        if (workerUuids[i] == uuid) {
            (*temperatures)[i] = static_cast<int>(value);
            return true;
        }
    }
    return false;
}

} // namespace temperature_mapping

#endif
