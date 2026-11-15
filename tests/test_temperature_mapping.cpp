#include "../temperature_mapping.h"
#include <cassert>
#include <string>
#include <vector>

int main() {
    const unsigned char uuidBytes[16] = {
        0x6c, 0x84, 0x9b, 0x47, 0x71, 0x11, 0x92, 0x65,
        0x9d, 0x42, 0xff, 0x8e, 0xdd, 0xc0, 0x5a, 0x2d
    };
    const std::string gpu0 = temperature_mapping::gpuUuid(uuidBytes);
    const std::string gpu1 = "GPU-81ac6e96-0bfe-390d-beb1-241466569d6b";
    assert(gpu0 == "GPU-6c849b47-7111-9265-9d42-ff8eddc05a2d");

    // One selected CUDA worker: another GPU's reading must not overwrite it.
    std::vector<std::string> selected(1, gpu0);
    std::vector<int> temperatures(1, 0);
    assert(temperature_mapping::applyTemperatureRow(gpu0 + ", 70", selected, &temperatures));
    assert(!temperature_mapping::applyTemperatureRow(gpu1 + ", 39", selected, &temperatures));
    assert(temperatures[0] == 70);

    // CUDA_VISIBLE_DEVICES can reorder CUDA indices relative to nvidia-smi.
    std::vector<std::string> reordered{gpu1, gpu0};
    std::vector<int> reversedTemperatures(2, 0);
    assert(temperature_mapping::applyTemperatureRow(gpu0 + ", 71", reordered, &reversedTemperatures));
    assert(temperature_mapping::applyTemperatureRow(gpu1 + ", 37", reordered, &reversedTemperatures));
    assert(reversedTemperatures[0] == 37 && reversedTemperatures[1] == 71);

    // Unsupported values and unrelated IDs leave the last valid reading intact.
    assert(!temperature_mapping::applyTemperatureRow(gpu1 + ", N/A", reordered, &reversedTemperatures));
    assert(!temperature_mapping::applyTemperatureRow(gpu1 + ", 200", reordered, &reversedTemperatures));
    assert(!temperature_mapping::applyTemperatureRow("garbage", reordered, &reversedTemperatures));
    assert(!temperature_mapping::applyTemperatureRow(
        "GPU-00000000-0000-0000-0000-000000000000, 30", reordered, &reversedTemperatures));
    assert(reversedTemperatures[0] == 37 && reversedTemperatures[1] == 71);
    return 0;
}
