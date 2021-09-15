#ifndef __PERCEPTRON_H_
#define __PERCEPTRON_H_

#include <algorithm>
#include <cmath>
#include <numeric>
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
    int n_pass() { return n_passes; }
    void train();
    VT query(const std::vector<NT>& q);
private:
    double r{0.1};  // learning rate <- (0, 1)
    size_t n_fs;
    std::vector<std::vector<NT>> m_xs;
    std::vector<VT> m_vs;
    std::vector<NT> m_ws;
    std::vector<VT> m_ys;
    NT m_error { 0.1 };
    int n_passes{ 0 };
    NT feature(size_t x_ndx, size_t f_ndx) const;
};

template<typename Iter1>
inline auto threshold(Iter1 w_beg, Iter1 w_end)
{
    using WT = typename Iter1::value_type;
    return [&, w_beg, w_end]<typename Iter2>(Iter2 x_beg){
        WT res = std::transform_reduce(w_beg, w_end, x_beg, WT{0.0});
        return res > -0.000001 ? WT{1.0} : WT{0.0};
    };
}

template<typename ValueType>
void Perceptron<ValueType>::init(size_t n_fs)
{
    this->n_fs = n_fs;
    m_xs.reserve(m_xs.size() * (n_fs+1));
    m_vs.reserve(m_xs.size());
    m_ws.reserve(n_fs + 1);
    m_ys.reserve(n_fs + 1);
    std::fill(m_ws.begin(), m_ws.end(), 0.0);
}


template<typename ValueType>
template<typename IterXs, typename IterV>
void Perceptron<ValueType>::set_training(
    IterXs beg_xs, IterXs end_xs,
    IterV beg_v, IterV end_v)
{
    std::transform(beg_xs, end_xs, std::back_inserter(m_xs), [](const auto& x){
        std::vector<NT> ret = x;
        ret.push_back(1.0);
        return ret;
    });
    std::copy(beg_v, end_v, std::back_inserter(m_vs));
}

template<typename ValueType>
inline typename Perceptron<ValueType>::NT Perceptron<ValueType>::feature(size_t x_ndx, size_t f_ndx) const {
    return m_xs[x_ndx][n_fs];
}

template<typename ValueType>
void Perceptron<ValueType>::train()
{
    NT cur_error = std::numeric_limits<NT>::max();
    n_passes = 0;
    while (cur_error > m_error) {
        ++n_passes;
        m_ys.clear();
        auto f = threshold(m_ws.begin(), m_ws.end());
        auto y_it = m_ys.begin();

        size_t x_ndx = 0;
        // For every vector in our training set:
        for (; y_it != m_ys.end(); ++y_it, ++x_ndx) {
            // Calculate the prediction
            *y_it = f((m_xs.begin() + x_ndx)->begin());
            // Update the weights
            for (int f_ndx=0; f_ndx<n_fs+1; ++f_ndx)
            {
                double err = m_vs[x_ndx] - *y_it;
                m_ws[f_ndx] += r * err * feature(x_ndx, f_ndx);
            }
        }
        cur_error = std::reduce(m_ys.begin(), m_ys.end(), 0.0, [x_ndx=0, v=m_vs[x_ndx]](const auto y, auto& s) mutable {
            s += std::abs(v - y);
            ++x_ndx;
            return s;
        });
    }
}

template<typename ValueType>
ValueType Perceptron<ValueType>::query(const std::vector<NT>& q)
{
    std::vector<NT> qq = q;
    qq.push_back(1.0);
    return threshold(m_ws.begin(), m_ws.end())(qq.begin());
}

} // namespace ml



#endif
