#include <perceptron.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


auto timer()
{
    return std::chrono::steady_clock::now();
}

void output_time(std::ostream& _out, const auto& tik, const auto& tok)
{
    using ms = std::chrono::milliseconds;
    _out << std::chrono::duration<double, std::milli>(tok - tik).count()//static_cast<double>(std::chrono::duration_cast<ms>(tok - tik).count())
        << "ms."
        << std::endl;

}

struct Training_set {
    std::vector<std::vector<double>> m_xs;
    std::vector<double> m_vs;
    Training_set() = default;
    void input(std::istream& _in,
               size_t N = std::numeric_limits<size_t>::max());
    auto xs_beg() { return m_xs.begin(); }
    auto xs_end() { return m_xs.end(); }
    auto vs_beg() { return m_vs.begin(); }
    auto vs_end() { return m_vs.end(); }
};

std::pair<std::vector<double>, double> parse(const std::string& s)
{
    std::vector<double> ret(6);
    double val{};
    std::stringstream ss{ s };
    ss >> val;
    for (int i=0; i<6; ++i)
        ss >> ret[i];
    return std::make_pair(ret, val);
}

void Training_set::input(std::istream& _in, size_t N)
{
    size_t count = 0;
    std::string buf;
    while (std::getline(_in, buf) && count < N) {
        auto [feature, val] = parse(buf);
        m_xs.push_back(feature);
        m_vs.push_back(val);
        ++count;
    }
}

double test(std::istream& ifs, const size_t n_train, const size_t n_test)
{
    Training_set train_s{};
    train_s.input(ifs, train_size);
    Training_set test_s{};
    test_s.input(ifs, test_size);
    ifs.close();

    ml::Perceptron perc{};
    perc.init(n_features);
    perc.set_training(train_s.xs_beg(), train_s.xs_end(), train_s.vs_beg(), train_s.vs_end());
    perc.set_error(error);
    perc.set_training_rate(tr_rate);

    std::vector<bool> res;
    double total = 0.0;
    res.reserve(test_size);

    auto tik = timer();
    perc.train();
    auto tok = timer();

    std::cout << "Time taken to train "
        << train_size
        << " inputs: ";
    output_time(std::cout, tik, tok);

    auto vs_it = test_s.vs_beg();

    tik = timer();
    for (auto [xs_it, vs_it] = std::make_pair(test_s.xs_beg(), test_s.vs_beg());
         xs_it != test_s.xs_end();
         ++xs_it, ++vs_it)
    {
        auto val = *vs_it;
        auto guess = perc.query(*xs_it);
        total += (val > 0.5 ? guess > 0.5 : guess < 0.5);
    }
    tok = timer();

    std::cout << "Time taken to test "
        << test_size
        << " inputs: ";
    output_time(std::cout, tik, tok);

    std::cout << "\n"
        << perc.n_pass()
        << " passes were needed for error < "
        << error
        << " and r = "
        << tr_rate
        << std::endl;

    double accuracy = 100.0 * total / test_size;
    std::cout << "\nAccuracy: "
        << accuracy
        << "%."
        << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc == 1) {
        std::cerr << "usage: "
            << argv[0]
            << " FILE..."
            << std::endl;
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];
    std::ifstream ifs{ filename };
    if (!ifs) {
        std::cerr << "Failed to open "
            << filename
            << std::endl;
        return EXIT_FAILURE;
    }

    const auto n_features = 6;
    const auto train_size = 891 - 200;
    const auto test_size = 200;
    const auto error = 0.2;
    const auto tr_rate = 0.1;



    return EXIT_SUCCESS;
}
