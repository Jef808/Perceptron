#ifndef __PERCEPTRON_H_
#define __PERCEPTRON_H_

#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <vector>

namespace ml {


template<typename ValueType = double>
class Perceptron {
public:
    using NT = double;
    using VT = ValueType;
    Perceptron() = default;
    void init(size_t n_features);
    template<typename InputIteratorX, typename InputIteratorV>
    void set_training(InputIteratorX beg_x, InputIteratorX end_x, InputIteratorV beg_v, InputIteratorV end_v);
    void set_error(NT err) { m_error = err; }
    void set_training_rate(NT tr) { r = tr; }
    void set_max_n_passes(int n) { max_n_passes = n; }
    int n_pass() { return n_passes; }
    void train();
    VT query(const std::vector<NT>& q);
    void show(std::ostream& _out);
private:
    double r{0.1};  // learning rate <- (0, 1)
    size_t n_fs;
    std::vector<std::vector<NT>> m_xs;
    std::vector<VT> m_vs;
    std::vector<NT> m_ws;
    std::vector<VT> m_ys;
    NT m_error { 0.1 };
    int max_n_passes{ 10 };
    int n_passes{ 0 };
};

template<typename ValueType>
void Perceptron<ValueType>::show(std::ostream& _out)
{
    for (auto ndx = 0; ndx < m_xs.size(); ++ndx) {
        _out << "Val: "
             << m_vs[ndx]
             << "Weight: "
             << m_ws[ndx]
             << " X: {";
        for (auto i=0; i<n_fs; ++i) {
            _out << m_xs[ndx][i] << ", ";
        }
        _out << "}\n";
    }
    _out << std::endl;
}

template<typename ValueType>
void Perceptron<ValueType>::init(size_t n_fs)
{
    this->n_fs = n_fs;
    m_ws.reserve(n_fs + 1);
    std::fill_n(std::back_inserter(m_ws), n_fs+1, 0.0);
}


template<typename ValueType>
template<typename IterXs, typename IterV>
void Perceptron<ValueType>::set_training(
    IterXs beg_xs, IterXs end_xs,
    IterV beg_v, IterV end_v)
{
    auto xs_it = beg_xs;
    for (; xs_it != end_xs; ++xs_it) {
        auto x = *xs_it;
        x.push_back(1.0);
        m_xs.push_back(x);
    }
    m_vs.reserve(m_xs.size());
    std::copy(beg_v, end_v, std::back_inserter(m_vs));
}

template<typename ValueType>
void Perceptron<ValueType>::train()
{
    double total_error = std::numeric_limits<double>::max();
    n_passes = 0;
    double goal_error = m_error * m_xs.size();

    while (total_error > goal_error && n_passes < max_n_passes) {
        ++n_passes;
        m_ys.clear();
        total_error = 0.0;

        // For every vector in our training set:
        for (int ndx=0; ndx<m_xs.size(); ++ndx) {
            // Compute the output of the perceptron
            const auto& x = m_xs[ndx];
            auto dot = std::transform_reduce(m_ws.begin(), m_ws.end(), x.begin(), 0.0);
            auto y = (dot > -0.00001 ? 1.0 : 0.0);
            auto v = m_vs[ndx];

            m_ys.push_back(y);

            // Update the weights
            double err_ndx = v - y;
            for (int i=0; i<n_fs+1; ++i) {
                m_ws[i] += r * err_ndx * x[i];
            }
            total_error += std::abs(err_ndx);
        }
    }
}

template<typename ValueType>
ValueType Perceptron<ValueType>::query(const std::vector<NT>& q)
{
    std::vector<NT> qq = q;
    qq.push_back(1.0);
    auto dot = std::transform_reduce(m_ws.begin(), m_ws.end(), qq.begin(), 0.0);
    return (dot > -0.00001) ? 1.0 : 0.0;
}

} // namespace ml



#endif
