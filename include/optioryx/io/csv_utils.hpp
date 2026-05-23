#pragma once

#include <string>
#include <vector>

namespace optioryx {

inline std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (character == '"') {
            if (in_quotes && index + 1U < line.size() && line[index + 1U] == '"') {
                field += '"';
                ++index;
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }
        if (character == ',' && !in_quotes) {
            fields.push_back(field);
            field.clear();
            continue;
        }
        if (character != '\r') {
            field += character;
        }
    }

    fields.push_back(field);
    return fields;
}

}  // namespace optioryx
