#ifndef CMDLINE_PARSER_IMPL_HPP_
#define CMDLINE_PARSER_IMPL_HPP_

#include <typeindex>
#include <typeinfo>
#include <stdexcept>
#include <iostream>

namespace cmdline_options {

option_base::option_base(std::string sn, std::string ln, std::string d, bool r) :
    shortname{sn},
    longname{ln},
    description{d},
    required{r}
{
    set = false;
}

template<typename T> 
option<T>::option(T& bindto, std::string sn, std::string ln, std::string d, bool r, T dv, const std::vector<T> vv) : 
        value{bindto},
        option_base{sn, ln, d, r},
        default_value{dv},
        valid_values{vv}
{
    value = default_value;
}

template<typename T> 
void option<T>::set_value(T v) {
        if (!std::is_same_v<T, bool> && !valid_values.empty()) {
            auto r = std::find(begin(valid_values), end(valid_values), v);
            if (r == std::end(valid_values)) {
                throw std::runtime_error("Option value is not valid");
            }
        } 
        
        value = v;
        set = true;
}

template<typename T> 
T option<T>::get_value() {
    return value;
}

template<typename T>
void options_parser::add_option(option<T> opt) {
    options.push_back(opt);
}

void options_parser::parse_input () {
    int i = 1;
    
    while (i < argc) {
        std::string arg{argv[i]};
        bool found = false;
        if (arg == "-h" || arg == "--help") {
            print_usage();
            std::exit(1);
        }
        for (auto& o : options) {
            option_base& ob = get_option_description(o);
            if (("-"+ob.shortname) == arg || ("--"+ob.longname) == arg) {
                found = true;
                std::type_index ti = std::type_index(o.type());
                if (ti == typeid(option<bool>)) {
                    auto& opt = std::any_cast<option<bool>&>(o);
                    opt.set_value(true);
                } else if ( ti == typeid(option<int>) ) {
                    i++;
                    auto& opt = std::any_cast<option<int>&>(o);
                    opt.set_value(std::stoi(argv[i]));
                } else if ( ti == typeid(option<unsigned int>) ) {
                    i++;
                    auto& opt = std::any_cast<option<unsigned int>&>(o);
                    opt.set_value(std::stoi(argv[i]));
                } else if ( ti == typeid(option<long>) ) {
                    i++;
                    auto& opt = std::any_cast<option<long>&>(o);
                    opt.set_value(std::stol(argv[i]));
                } else if ( ti == typeid(option<float>) ) {
                    i++;
                    auto& opt = std::any_cast<option<float>&>(o);
                    opt.set_value(std::stof(argv[i]));
                } else if ( ti == typeid(option<double>) ) {
                    i++;
                    auto& opt = std::any_cast<option<double>&>(o);
                    opt.set_value(std::stod(argv[i]));
                } else if ( ti == typeid(option<std::string>) ) {
                    i++;
                    auto& opt = std::any_cast<option<std::string>&>(o);
                    opt.set_value(argv[i]);
                } else {
                    throw std::runtime_error{"Bad option type"};
                }
            }
        }
        if (!found) {
            throw std::runtime_error("Bad option name");
        }
        i++;
    }
    for (auto&o : options) {
        auto& opt = get_option_description(o);
        if (opt.required && !opt.set) {
            throw std::runtime_error("Required option not set");
        }
    }
}

void options_parser::print_usage() {
    std::string prog_path{argv[0]};
    auto pos=prog_path.find_last_of('/');
    auto prog_name=prog_path.substr(pos+1);
    std::cout << "Usage: " << prog_name << " [options]\nOptions: \n  -h, --help: print usage\n";
    for (auto& o : options) {
        auto& opt = get_option_description(o);
        std::cout << "  -" << opt.shortname << ((opt.shortname.empty() || opt.longname.empty())? "" : ", ") 
                    << (!opt.longname.empty()? (opt.shortname.empty()? "-" : "--") : "") << opt.longname << ": " 
                    << opt.description << (opt.required? " (required!)" : "") << '\n';
    }
}

option_base& options_parser::get_option_description(std::any& o) {
    std::type_index ti = std::type_index(o.type());
    if (ti == typeid(option_base)) {
        return std::any_cast<option_base&>(o);
    } else if (ti == typeid(option<bool>)) {
        return std::any_cast<option<bool>&>(o);
    } else if ( ti == typeid(option<int>) ) {
        return std::any_cast<option<int>&>(o);
    } else if ( ti == typeid(option<unsigned int>) ) {
        return std::any_cast<option<unsigned int>&>(o);
    } else if ( ti == typeid(option<long>) ) {
        return std::any_cast<option<long>&>(o);
    } else if ( ti == typeid(option<float>) ) {
        return std::any_cast<option<float>&>(o);
    } else if ( ti == typeid(option<double>) ) {
        return std::any_cast<option<double>&>(o);
    } else if ( ti == typeid(option<std::string>) ) {
        return std::any_cast<option<std::string>&>(o);
    }else {
        throw std::runtime_error{"Bad option type"};
    }
}
}

#endif






// using namespace cmdline_options;

// struct opts {
//     int a;
//     long b;
//     float c;
//     double d;
//     bool bl;
//     std::string str;
// };

// int main (int argc, char *argv[]) {
//     opts opts{};
//     options_parser parser(argc, argv);
//     parser.add_option<std::string>({opts.str, "", "str", "value of str option", false, "ciao"});
//     parser.add_option<bool>({opts.bl, "", "bool", "value of bool option", false, false});
//     parser.add_option<int>({opts.a, "a", "avariable", "value of a option", true, 5});
//     parser.add_option<long>({opts.b, "b", "", "value of b option", false, 10});
//     parser.add_option<float>({opts.c, "c", "", "value of c option", false, 4.2});
//     parser.add_option<double>({opts.d, "d", "", "value of d option", false, 42.0});

//     try {
//         parser.parse_input();
//     } catch (const std::exception& e) {
//         std::cerr << e.what() << '\n';
//         parser.print_usage();
//         return 1;
//     }

//     std::cout << opts.a << "," << opts.b << "," << opts.c << "," << opts.d << 
//                 "," << opts.bl << "," << opts.str << '\n';
//     return 0;
// }
