/* implementation of the algorithm 
 * 
  D. Conte, L. Ixaru, B. Paternoster, G. Santomauro:
    Exponentially-fitted Gauss–Laguerre quadrature rule for integrals over an unbounded interval.
    
  https://www.sciencedirect.com/science/article/pii/S0377042713003385
 
 
  Some tweaks to the algorithm are outlined in 
  https://wordpress.com/post/hpcquantlib.wordpress.com/4557
  
    Copyright (c) 2020, Klaus Spanderen
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
    
    3. Neither the names of the copyright holders nor the names of the QuantLib   
    Group and its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
*/

#include <ql/math/integrals/gaussianquadratures.hpp>

#include <cmath>
#include <iostream>
#include <fstream>
#include <queue>

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/unordered_map.hpp>

using namespace boost::numeric::ublas;

typedef boost::multiprecision::number<
            boost::multiprecision::cpp_dec_float<600> > Float;

template <class real>
boost::numeric::ublas::vector<real> lu(
    const boost::numeric::ublas::matrix<real>& A,
    const boost::numeric::ublas::vector<real>& b) {

    matrix<real> A_fac = A;

    permutation_matrix<std::size_t> piv(b.size());
    int singular = lu_factorize(A_fac, piv);

    if (singular) 
        throw std::runtime_error("lu: A is singular.");

    vector<real> x = b;
    lu_substitute(A_fac, piv, x); 

    return x;
}

template <class real>
boost::numeric::ublas::matrix<real> inv(
    const boost::numeric::ublas::matrix<real>& A) {
    
    using namespace boost::numeric::ublas;

    matrix<real> A_fac = A;

    permutation_matrix<std::size_t> piv(A.size1());
    int singular = lu_factorize(A_fac, piv);

    if (singular) 
        throw std::runtime_error("lu: A is singular.");
    
    matrix<real> inverse = identity_matrix<real>(A.size1());
    
    lu_substitute(A_fac, piv, inverse);
    
    return inverse;
}

template <class result_type, class float_type>            
result_type eta(int m, float_type Z) {
    
    typedef std::pair<int, float_type> key_type;
    typedef boost::unordered_map<key_type, result_type> ResultMap;

    static std::size_t maxCacheSize 
        = 1024*4096 / (sizeof(key_type) + sizeof(int));
        
    static ResultMap results;
    static std::queue<key_type> fifo;

    const key_type key(m, Z);
    const typename ResultMap::const_iterator iter = results.find(key);
    
    if (iter != results.end()) {        
        return iter->second;
    }

    result_type result, z=Z;
      
    if (m == 0) {
        if (z < 0) {
            const result_type sz = sqrt(-z);
            result = sin(sz)/sz;
        }
        else if (z > 0) {
            const result_type sz = sqrt(z);
            result = sinh(sz)/sz;
        }
        else 
            result = result_type(1.0);        
    }
    else if (m == -1)        
        if (z <= 0)
            result = cos(sqrt(-z)); 
        else
            result = cosh(sqrt(z));
    else
        result = (eta<result_type, float_type>(m-2, Z) 
            - (2*m-1)*eta<result_type, float_type>(m-1, Z))/z;
    
    results.emplace(key, result);
    fifo.push(key);
    
    while(fifo.size() > maxCacheSize) {
        results.erase(fifo.front());
        fifo.pop();
    }
        
    return result;        
}

template <class real>
real factorial(std::size_t n) {
    static std::vector<real> cache(1, real(1));

    if (cache.size() > n)
        return cache[n];
        
    const real val = real(n) * factorial<real>(n-1);
    cache.resize(n+1);
    
    return cache[n] = val;
}


template <class real>
vector<real> w_lin(const vector<real>& x, real Z) {
    const int N = x.size();
    
    const int s = N/2;
    const int r = N-s;
    
    matrix<real> A(N, N);
    int row = 0;
    
    vector<real> b(x.size());
    
    for (int n=r+1; n <= N; ++n) {
        for (int k=0; k < N; ++k)
            A(row, k) = pow(x(k), 2*n-2)*eta<real, real>(n-2, x(k)*x(k)*Z);
        
        b(row) = pow(real(2), n-1)*factorial<real>(n-1)/pow(1.0-Z, n);
        ++row;
    }
    
    for (int n=s+1; n <= N; ++n) {
        for (int k=0; k < N; ++k)
            A(row, k) = pow(x(k), 2*n-1)*eta<real, real>(n-1, x(k)*x(k)*Z);

        b(row) = pow(real(2), n-1)*factorial<real>(n-1)/pow(1.0-Z, n);
        ++row;
    }
    
    return lu(A,b);
}


template <class real>
vector<real> f(const vector<real>& w, const vector<real>& x, real Z) {
    const int N = x.size();
    
    const int s = N/2;
    const int r = N-s;

    int row = 0;
    vector<real> b(x.size());

    for (int n=1; n <= r; ++n) {
        b(row) = real(0.0);
        
        for (int k=0; k < N; ++k) 
            b(row) += w(k)*pow(x(k), 2*n-2)*eta<real, real>(n-2, x(k)*x(k)*Z);
        
        b(row) -= pow(real(2), n-1)*factorial<real>(n-1)/pow(1.0-Z, n);        
        ++row;
    }
    
    for (int n=1; n <= s; ++n) {
        b(row) = real(0.0);

        for (int k=0; k < N; ++k) 
            b(row) += w(k)*pow(x(k), 2*n-1)*eta<real, real>(n-1, x(k)*x(k)*Z);
        
        b(row) -= pow(real(2), n-1)*factorial<real>(n-1)/pow(1.0-Z, n);
        ++row;
    }
    
    return b;
}


template <class real>
vector<real> newton_iter(const vector<real>& w, const vector<real>& x, real Z) {

    const int N = x.size();
    const int s = N/2;
    const int r = N-s;
    
    matrix<real> JxA(N, N);
    
    for (int i=1; i <= N; ++i)
        for (int j=0; j < N; ++j) {
            const real xxZ = x(j)*x(j)*Z;
            
            JxA(i-1, j) = (i <= s)
                ? pow(x(j), 2*(i+r)-3)*( 
                    2*(i+r-1)*eta<real, real>(i+r-2, xxZ) 
                    + xxZ*eta<real, real>(i+r-1, xxZ))
                : pow(x(j), 2*(i-1))*(
                    (2*i-1)*eta<real, real>(i-1, xxZ) 
                    + xxZ*eta<real, real>(i, xxZ));
        }
        
        
    for (int i=0; i < N; ++i)
        for (int j=0; j < N; ++j)
            JxA(i, j) *= -w(j);

    matrix<real> A(N, N);
    for (int i=1; i <= N; ++i)
        for (int j=0; j < N; ++j)
            A(i-1, j) = (i <= s) 
               ? pow(x(j), 2*(i+r-1))*eta<real, real>(i+r-2, x(j)*x(j)*Z)
               : pow(x(j), 2*i-1    )*eta<real, real>(i-1,   x(j)*x(j)*Z);
    
    const matrix<real> JxW = prod(inv(A), JxA);

    matrix<real> C(N, N);
    for (int i=1; i <= N; ++i)
        for (int k=0; k < N; ++k) {
            const real xxZ = x(k)*x(k)*Z;
            
            C(i-1, k) = (i <= r)
                ? pow(x(k), 2*i-3)*( (2*i-2)*eta<real, real>(i-2, xxZ)
                    + xxZ*eta<real, real>(i-1, xxZ) )
                : pow(x(k), 2*(i-r-1))*( (2*(i-r)-1)*eta<real, real>(i-r-1, xxZ)
                    + xxZ*eta<real, real>(i-r, xxZ) );                
        }
    
    for (int i=0; i < N; ++i)
        for (int j=0; j < N; ++j)
            C(i, j) *= w(j);
            
    matrix<real> D(N, N);
    for (int i=1; i <= N; ++i)
        for (int k=0; k < N; ++k) {
            const real xxZ = x(k)*x(k)*Z;
            
            D(i-1, k) = ( i <= r)
                ? pow(x(k), 2*i-2)    *eta<real, real>(i-2, xxZ)
                : pow(x(k), 2*(i-r)-1)*eta<real, real>(i-r-1, xxZ);            
        }    

    
    const matrix<real> B = C + prod(D, JxW);

    vector<real> dZ(N);
    for (int i=1; i <= N; ++i)
        dZ(i-1) = (i <= r)
            ? pow(real(2), i-1)*factorial<real>(i-1) / pow(1-Z, i)
            : pow(real(2), i-r-1)*factorial<real>(i-r-1) / pow(1-Z, i-r);
    
    return lu(B, vector<real>(prod(D, w) - dZ));
}
    
    
template <class real>
vector<real> newton(vector<real>& x, real Z) {
    const static real eps = Float("1e-350");

    const std::size_t N = x.size();
    
    vector<real> w(N), dx;
    
    do {
        w = w_lin(x, Z);

        dx = newton_iter(w, x, Z);           

        x = x - dx;
        std::cout << norm_2(dx) << std::endl;
        
        for (std::size_t i=0; i < N; ++i)
            if (x(i) < 0.0) {
                return vector<real>();
            }
    }
    while (norm_2(dx) > eps);
    
    return w;
}


bool greaterThan(vector<Float>& x, vector<Float>& y) {
    bool f = false;
    
    for (std::size_t i=0; i < x.size(); ++i) {
        if (x[i] >= y[i]) {
            f = true;
        }
    }
    return f;
}


int main() {
    
    const std::size_t n = 48;
    const std::size_t maxOrder = 20;
    
    const QuantLib::Array x_laguerre = 
        QuantLib::GaussLaguerreIntegration(n).x();
    const QuantLib::Array w_laguerre = 
        QuantLib::GaussLaguerreIntegration(n).weights();
                
    std::vector<vector<Float> > x(maxOrder, vector<Float>(n));
    std::copy(x_laguerre.begin(), x_laguerre.end(), x[0].begin());

    std::ofstream f("values.txt");
    f << std::setprecision(std::numeric_limits<double>::digits10 + 1)
        << "{ 0.0";
    for (std::size_t i = 0; i < n; ++i)
        f << ", " << x_laguerre[i];
    for (std::size_t i = 0; i < n; ++i)
        f << ", " << w_laguerre[i];
    f << " }," << std::endl;        
    f.flush();

    
    vector<Float> xGuess;
    std::vector<Float> o(maxOrder, Float(0.0));

    std::size_t iter = 0;
    
    while (o[0] < 50.0) {

        ++iter;
        const std::size_t order = std::min(maxOrder, iter);
        
        vector<Float> w;
        vector<Float> xTest(n);
        
        const Float m1 = o[0] + Float(0.01);
        const Float m2 = o[0]*(1 + 0.01);
        
        const Float nomega = (m1 > m2)? m1 : m2;
        const Float z = -nomega*nomega;
                    
        for (std::size_t i=0; i < order; ++i) {
            Float l=1.0;
            for (std::size_t j=0; j < order; ++j) 
                if (i != j)
                    l *= (nomega-o[j])/(o[i]-o[j]);
                                
            xTest += x[i]*l;
            xGuess = xTest;
        }

        w = newton(xTest, z);
        
        std::cout << "start norm " << nomega << " " << norm_2(xGuess - xTest) << std::endl;
        
        if (greaterThan(xTest, x[0]))
            std::cout << "wrong direction" << std::endl;
        
        for (int i=maxOrder-1; i > 0; --i) {
            x[i] = x[i-1];
            o[i] = o[i-1];
        }
        
        x[0] = xTest;
        o[0] = nomega;
                
        Float s=0;
        for (std::size_t i=0; i < n; ++i)
            s+=w(i)*(x[0](i)*cos(o[0]*x[0](i)) + x[0](i)*sin(o[0]*x[0](i)));

        const Float expected = (1+2*o[0]-o[0]*o[0])/(1+o[0]*o[0])/(1+o[0]*o[0]);
        if (abs(s - expected) > 1e-16) {
            std::cout << "integration error " << abs(s - expected) << std::endl;
            exit(-1);
        }
        
        f << std::setprecision(std::numeric_limits<double>::digits10 + 1)
            << "{ " << o[0];
        for (std::size_t i = 0; i < n; ++i)
            f << ", " << x[0](i);
        for (std::size_t i = 0; i < n; ++i)
            f << ", " << exp(x[0](i))*w(i);
        f << " }," << std::endl;        
        f.flush();
    }
    f.close();
}
