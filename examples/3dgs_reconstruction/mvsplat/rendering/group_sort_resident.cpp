// Keep the frozen sorting implementation, but serve multiple view requests.
#define main frozen_group_sort_main
#include "group_sort.cpp"
#undef main

#include <sstream>

int main() try {
    std::cout << "SORT_READY" << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string action, source, destination, extra;
        if (!(input >> action)) throw std::runtime_error("Empty sort command");
        if (action == "QUIT") break;
        if (action != "SORT" || !(input >> std::quoted(source) >> std::quoted(destination)) ||
            (input >> extra)) throw std::runtime_error("SORT needs input and output paths");
        char program[] = "group_sort";
        char *argv[] = {program, source.data(), destination.data()};
        if (frozen_group_sort_main(3, argv)) throw std::runtime_error("Frozen group sort failed");
        std::cout << "SORT_COMPLETE" << std::endl;
    }
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
