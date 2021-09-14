#include "perceptron.h"

#include <algorithm>
#include <numeric>
#include <vector>

template<typename NT>
inline NT dot(const std::vector<NT>& a, const std::vector<NT>& b)
{
    return std::transform_reduce(a.begin(), a.end(), b.begin(), b.end(), 0);
}
