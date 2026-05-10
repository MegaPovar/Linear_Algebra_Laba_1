#include "gauss.h"
#include "lu.h"
#include "utils.h"
#include <iostream>

int main() {
    try {
        runAllExperiments();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}