#include "optioryx/util/format.hpp"

#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>

namespace optioryx {

void configure_lossless_floating_output(std::ostream& output) {
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
}

std::string format_double(double value) {
    std::ostringstream output;
    configure_lossless_floating_output(output);
    output << value;
    return output.str();
}

}  // namespace optioryx
