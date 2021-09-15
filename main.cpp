#include <perceptron.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


constexpr auto n_features = 6;

auto timer()
{
    return std::chrono::steady_clock::now();
}

void output_time(std::ostream& _out, const auto& tik, const auto& tok)
{
    using ms = std::chrono::milliseconds;
    _out << std::chrono::duration<double, std::milli>(tok - tik).count()
        << "ms."
        << std::endl;

}

struct Training_set {
    std::vector<std::vector<double>> m_xs;
    std::vector<double> m_vs;
    Training_set() = default;
    void input(std::istream& _in,
               bool training=true,
               size_t N = std::numeric_limits<size_t>::max());
    auto xs_beg() { return m_xs.begin(); }
    auto xs_end() { return m_xs.end(); }
    auto vs_beg() { return m_vs.begin(); }
    auto vs_end() { return m_vs.end(); }
};

std::pair<std::vector<double>, double> parse(const std::string& s, bool training)
{
    std::vector<double> ret(6);
    double val{0.0};
    double p_id{0.0};
    std::stringstream ss{ s };
    ss >> val;
    for (int i=0; i<6; ++i)
        ss >> ret[i];
    return std::make_pair(ret, val);
}

inline auto& normalize(auto& feature) {
    feature[3] = feature[3] / 40;
    return feature;
}

void Training_set::input(std::istream& _in, bool training, size_t N)
{
    size_t count = 0;
    std::string buf;
    while (std::getline(_in, buf) && count < N) {
        auto [feature, val] = parse(buf, training);
        m_xs.push_back(normalize(feature));
        m_vs.push_back(val);
        ++count;
    }
}

void test(std::istream& ifs,
            const size_t n_train,
            const size_t n_test,
            const double error = 0.05,
            const double r = 0.05,
            const int max_pass = 10)
{
    auto tik = timer();

    Training_set train_s{};
    train_s.input(ifs, true, n_train);
    Training_set test_s{};
    test_s.input(ifs, true, n_test);

    ml::Perceptron<double> perc{};
    perc.init(n_features);
    perc.set_training(train_s.xs_beg(), train_s.xs_end(), train_s.vs_beg(), train_s.vs_end());
    perc.set_error(error);
    perc.set_training_rate(r);
    perc.set_max_n_passes(max_pass);

    //perc.show(std::cout);

    auto tok = timer();
    std::cout << "Time taken to input and set up: ";
    output_time(std::cout, tik, tok);

    std::vector<bool> res;
    double total = 0.0;
    res.reserve(n_test);

    tik = timer();
    perc.train();
    tok = timer();

    std::cout << "Time taken to train "
        << n_train
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
        << n_test
        << " inputs: ";
    output_time(std::cout, tik, tok);

    std::cout << "\n"
        << perc.n_pass()
        << " passes were needed for error < "
        << error
        << " and r = "
        << r
        << std::endl;

    double accuracy = 100.0 * total / n_test;
    std::cout << "\nAccuracy: "
        << accuracy
        << "%."
        << std::endl;
}

void write_result(std::ofstream& ofs, const auto& res)
{
    ofs << "PassengerId,Survived\n";
    for (const auto& res_id : res) {
        int guess = res_id.first ? 1 : 0;
        ofs << res_id.first
            << ','
            << res_id.second
            << '\n';
    }
}

void solve_kaggle(std::ifstream& ifs_train,
                  std::ifstream& ifs_test,
                  std::ofstream& ofs,
                  const double error = 0.05,
                  const double r = 0.05)
{
    Training_set train_s{};
    ml::Perceptron perc{};
    train_s.input(ifs_train, true);

    perc.init(n_features);
    perc.set_training(train_s.xs_beg(), train_s.xs_end(), train_s.vs_beg(), train_s.vs_end());
    perc.set_error(error);
    perc.set_training_rate(r);

    auto tik = timer();
    perc.train();
    auto tok = timer();
    ifs_train.close();

    std::cout << "Time taken to train "
        << std::distance(train_s.xs_beg(), train_s.xs_end())
        << " inputs: "
        << "\nError: ";
        //<< perc.error();
    output_time(std::cout, tik, tok);

    Training_set test_s{};
    test_s.input(ifs_test, false);

    std::vector<std::pair<int, bool>> queries;

    tik = timer();
    for (auto [xs_it, vs_it] = std::make_pair(test_s.xs_beg(), test_s.vs_beg());
         xs_it != test_s.xs_end();
         ++xs_it, ++vs_it)
    {
        auto id = *vs_it;
        auto guess = perc.query(*xs_it) > 0.5;
        queries.push_back(std::make_pair(id, guess));
    }
    tok = timer();

    write_result(ofs, queries);

    std::cout << "Time taken to test "
        << std::distance(test_s.xs_beg(), test_s.xs_end())
        << " inputs: ";
    output_time(std::cout, tik, tok);

    std::cout << "\n"
        << perc.n_pass()
        << " passes were needed for error < "
        << error
        << " and r = "
        << r
        << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 4 && argc != 2) {
        std::cerr << "usage: "
            << argv[0]
            << "[Training Testing Output]"
            << std::endl;
        return EXIT_FAILURE;
    }

    const char* fn_train = argv[1];
    std::ifstream ifs_train { fn_train };
    if (!ifs_train) {
        std::cerr << "Failed to open "
            << fn_train
            << std::endl;
        return EXIT_FAILURE;
    }

    const auto error = 0.1;
    const auto tr_rate = 0.1;
    const auto max_pass = 1000;

    if (argc == 2) {
        const auto n_train = 891 - 300;
        const auto n_test = 300;
        test(ifs_train, n_train, n_test, error, tr_rate, max_pass);
        return EXIT_SUCCESS;
    }

    const char* fn_test = argv[2];
    std::ifstream ifs_test { fn_test };
    if (!ifs_test) {
        std::cerr << "Failed to open "
            << fn_test
            << std::endl;
        return EXIT_FAILURE;
    }

    const char* fn_out = argv[3];
    std::ofstream ofs { fn_out };
    if (!ofs) {
        std::cerr << "Failed to open "
                  << fn_out
                  << std::endl;
        return EXIT_FAILURE;
    }

    solve_kaggle(ifs_train,
                 ifs_test,
                 ofs,
                 error,
                 tr_rate);

    return EXIT_SUCCESS;
}
