#pragma once

#include <iosfwd>
#include <string>

namespace optioryx {

// max_digits10 garandeert dat een eindige double na tekstserialisatie exact
// dezelfde binaire double oplevert. De classic locale houdt JSON/CSV numeriek.
void configure_lossless_floating_output(std::ostream& output);
std::string format_double(double value);

}  // namespace optioryx
