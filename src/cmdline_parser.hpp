#ifndef CMDLINE_PARSER_HPP_
#define CMDLINE_PARSER_HPP_

#include <any>
#include <string>
#include <vector>

namespace cmdline_options {

struct option_base {
    option_base(std::string sn, std::string ln, std::string d, bool r);

    std::string shortname;
    std::string longname;
    std::string description;
    bool required;
    bool set;
};

template<typename T = bool> 
struct option : option_base {
    option(T& bindto, std::string sn, std::string ln, std::string d, bool r, T dv, const std::vector<T> vv); 
    
    void set_value(T v);
    T get_value();

    const T default_value;
private:
    const std::vector<T> valid_values;
    T& value;
};

struct options_parser {
    options_parser(int argc, char *argv[]) : argc{argc}, argv{argv} {}

    template<typename T>
    void add_option(option<T> opt);

    void parse_input ();
    void print_usage();
private:
    option_base& get_option_description(std::any& o);
    
    int argc;
    char **argv;
    std::vector<std::any> options;
};

} // namespace cmdline_options

#include "cmdline_parser_imp.hpp"

#endif