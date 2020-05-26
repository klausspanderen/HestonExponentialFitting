/* Implementation of the algorithm
 * 
  D. Conte, L. Ixaru, B. Paternoster, G. Santomauro:
    Exponentially-fitted Gauss–Laguerre quadrature rule for integrals over an unbounded interval.
    
  https://hpcquantlib.wordpress.com/2020/05/17/optimized-heston-model-integration-exponentially-fitted-gauss-laguerre-quadrature-rule/
 
  
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
#include <boost/unordered_map.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>


using namespace boost::numeric::ublas;

//typedef boost::multiprecision::number<
//            boost::multiprecision::cpp_dec_float<400> > Float;

typedef boost::multiprecision::number<
            boost::multiprecision::gmp_float<600> > Float;


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

Float pow_c(const Float& x, int n) {

    if (n == 0)
        return Float(1.0);

    if (n == 1)
        return x;

    typedef std::pair<int, Float> key_type;

    typedef boost::unordered_map<key_type, Float> ResultMap;

    static boost::mutex mutex;

    static std::size_t maxCacheSize = 16384;

    static ResultMap results;
    static std::queue<key_type> fifo;

    const key_type key(n, x);
    {
        boost::lock_guard<boost::mutex> lock(mutex);

        const typename ResultMap::const_iterator iter = results.find(key);

        if (iter != results.end())
            return iter->second;
    }

    const Float result = (n < 0)
                ? Float(1.0)/pow_c(x, -n)
                : Float(pow_c(x, n/2) * pow_c(x, n/2) * pow_c(x, n - 2*(n/2)));

    boost::lock_guard<boost::mutex> lock(mutex);

    results.emplace(key, result);
    fifo.push(key);

    while(fifo.size() > maxCacheSize) {
        results.erase(fifo.front());
        fifo.pop();
    }

    return result;
}

Float pow_i(long n) {
    if (n < 0)
        return pow_c(Float(2), n);
    else if (n < 8*sizeof(unsigned long))
        return Float(1uL << n);
    else
        return pow_c(Float(2), n);
}


template <class result_type, class float_type>            
result_type eta(int m, float_type Z) {
    
    typedef std::pair<int, float_type> key_type;
    typedef boost::unordered_map<key_type, result_type> ResultMap;

    static boost::mutex mutex;

    static std::size_t maxCacheSize = 16384;
        
    static ResultMap results;
    static std::queue<key_type> fifo;

    const key_type key(m, Z);
    {
        boost::lock_guard<boost::mutex> lock(mutex);
        const typename ResultMap::const_iterator iter = results.find(key);

        if (iter != results.end()) {
            return iter->second;
        }
    }

    result_type result;

    if (m == 0) {
        if (Z < 0) {
            const result_type sz = sqrt(-Z);
            result = sin(sz)/sz;
        }
        else if (Z > 0) {
            const result_type sz = sqrt(Z);
            result = sinh(sz)/sz;
        }
        else
            result = result_type(1.0);
    }
    else if (m == -1)
        if (Z <= 0)
            result = cos(sqrt(-Z));
        else
            result = cosh(sqrt(Z));
    else
        result = (eta<result_type, float_type>(m-2, Z) 
            - (2*m-1)*eta<result_type, float_type>(m-1, Z))/Z;
    
    boost::lock_guard<boost::mutex> lock(mutex);
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
vector<real> w_lin(const vector<real>& x, const real& Z) {
    const int N = x.size();
    
    const int s = N/2;
    const int r = N-s;
    
    matrix<real> A(N, N);
    int row = 0;
    
    vector<real> b(x.size());
    
    for (int n=r+1; n <= N; ++n) {
        for (int k=0; k < N; ++k)
            A(row, k) = pow_c(x(k), 2*n-2)*eta<real, real>(n-2, x(k)*x(k)*Z);
        
        b(row) = pow_i(n-1)*factorial<real>(n-1)/pow_c(1.0-Z, n);
        ++row;
    }
    
    for (int n=s+1; n <= N; ++n) {
        for (int k=0; k < N; ++k)
            A(row, k) = pow_c(x(k), 2*n-1)*eta<real, real>(n-1, x(k)*x(k)*Z);

        b(row) = pow_i(n-1)*factorial<real>(n-1)/pow_c(1.0-Z, n);
        ++row;
    }
    
    return lu(A,b);
}


class WorkerJxA {
  public:
    WorkerJxA(
        matrix<Float>& JxA, const vector<Float>& w,
        const vector<Float>& x, const Float& Z)
    : JxA_(JxA), w_(w), x_(x), Z_(Z) {}

    void run() const {
        const int N = JxA_.size1();
        const int s = N/2;
        const int r = N-s;

        for (int j=0; j < N; ++j) {
            const Float xxZ = x_(j)*x_(j)*Z_;

            for (int i=1; i <= N; ++i) {
                JxA_(i-1, j) = (i <= s)
                    ? pow_c(x_(j), 2*(i+r)-3)*(
                        2*(i+r-1)*eta<Float, Float>(i+r-2, xxZ)
                        + xxZ*eta<Float, Float>(i+r-1, xxZ))
                    : pow_c(x_(j), 2*(i-1))*(
                        (2*i-1)*eta<Float, Float>(i-1, xxZ)
                        + xxZ*eta<Float, Float>(i, xxZ));
                JxA_(i-1, j) *= -w_(j);
            }
        }
    }

  private:
    matrix<Float>& JxA_;
    const vector<Float>& w_, x_;
    const Float& Z_;
};

class WorkerInvA {
  public:
    WorkerInvA(matrix<Float>& A, const vector<Float>& x, const Float& Z)
    : A_(A), x_(x), Z_(Z) { }

    void run() const {
        const int N = A_.size1();
        const int s = N/2;
        const int r = N-s;

        for (int j=0; j < N; ++j) {
            const Float xxZ = x_(j)*x_(j)*Z_;
            for (int i=1; i <= N; ++i)
                A_(i-1, j) = (i <= s)
                   ? pow_c(x_(j), 2*(i+r-1))*eta<Float, Float>(i+r-2, xxZ)
                   : pow_c(x_(j), 2*i-1    )*eta<Float, Float>(i-1,   xxZ);
        }

        A_ = inv(A_);
    }

  private:
    matrix<Float>& A_;
    const vector<Float>& x_;
    const Float& Z_;
};

class WorkerC {
  public:
    WorkerC(matrix<Float>& C, const vector<Float>& w,
            const vector<Float>& x, const Float& Z)
    : C_(C), w_(w), x_(x), Z_(Z) {}

    void run() const {
        const int N = C_.size1();
        const int s = N/2;
        const int r = N-s;

        for (int k=0; k < N; ++k) {
            const Float xxZ = x_(k)*x_(k)*Z_;
            for (int i=1; i <= N; ++i) {
                C_(i-1, k) = (i <= r)
                    ? pow_c(x_(k), 2*i-3)*( (2*i-2)*eta<Float, Float>(i-2, xxZ)
                        + xxZ*eta<Float, Float>(i-1, xxZ) )
                    : pow_c(x_(k), 2*(i-r-1))*( (2*(i-r)-1)*eta<Float, Float>(i-r-1, xxZ)
                        + xxZ*eta<Float, Float>(i-r, xxZ) );
                C_(i-1, k) *= w_(k);
            }
        }
    }
  private:
    matrix<Float>& C_;
    const vector<Float>& w_, x_;
    const Float& Z_;
};

class WorkerD {
  public:
    WorkerD(matrix<Float>& D, vector<Float>& dZ, const vector<Float>& x, const Float& Z)
    : D_(D), dZ_(dZ), x_(x), Z_(Z) {}

    void run() const {
        const int N = D_.size1();
        const int s = N/2;
        const int r = N-s;

        for (int k=0; k < N; ++k) {
            const Float xxZ = x_(k)*x_(k)*Z_;
            for (int i=1; i <= N; ++i)
                D_(i-1, k) = ( i <= r)
                    ? pow_c(x_(k), 2*i-2)    *eta<Float, Float>(i-2, xxZ)
                    : pow_c(x_(k), 2*(i-r)-1)*eta<Float, Float>(i-r-1, xxZ);
        }

        const Float omz(1-Z_);

        for (int i=1; i <= N; ++i)
            dZ_(i-1) = (i <= r)
                ? pow_i(i-1)*factorial<Float>(i-1) / pow_c(omz, i)
                : pow_i(i-r-1)*factorial<Float>(i-r-1) / pow_c(omz, i-r);
    }

  private:
    matrix<Float>& D_;
    vector<Float>& dZ_;
    const vector<Float>& x_;
    const Float& Z_;
};


template <class real>
vector<real> newton_iter(const vector<real>& w, const vector<real>& x, const real& Z) {
    const int N = x.size();
    const int s = N/2;
    const int r = N-s;
    
    matrix<real> invA(N, N);
    WorkerInvA workerInvA(invA, x, Z);

    boost::thread invA_thread(&WorkerInvA::run, &workerInvA);

    matrix<real> JxA(N, N);

    WorkerJxA workerJxA(JxA, w, x, Z);
    boost::thread JxA_thread(&WorkerJxA::run, &workerJxA);

    matrix<real> C(N, N);
    WorkerC workerC(C, w, x, Z);
    boost::thread C_thread(&WorkerC::run, &workerC);

    matrix<real> D(N, N);
    vector<real> dZ(N);
    WorkerD workerD(D, dZ, x, Z);
    boost::thread D_thread(&WorkerD::run, &workerD);


    JxA_thread.join();
    invA_thread.join();
    const matrix<real> JxW = prod(invA, JxA);

    C_thread.join();
    D_thread.join();
    
    const matrix<real> B = C + prod(D, JxW);

    return lu(B, vector<real>(prod(D, w) - dZ));
}
    
    
template <class real>
vector<real> newton(vector<real>& x, real Z) {
    const static real eps = Float(1e-300);

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
    
    const std::size_t n = 64;
    const std::size_t maxOrder = 45;
    
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
    o[0] = 0.01;
    
    std::size_t iter = 0;
    
    vector<Float> w;
    w = newton(x[0], Float(-o[0]*o[0]));
    
    while (o[0] < 50.0) {

        ++iter;
        const std::size_t order = std::min(maxOrder, iter);
        
        vector<Float> xTest(n);
        
        const Float m1 = o[0] + Float(0.01);
        const Float m2 = o[0]*(1 + 0.0075);
        
        Float nomega = (m1 > m2)? m1 : m2;
        
        do {
            const Float z = -nomega*nomega;
                        
            for (std::size_t i=0; i < order; ++i) {
                Float l=1.0;
                for (std::size_t j=0; j < order; ++j) 
                    if (i != j)
                        l *= (nomega-o[j])/(o[i]-o[j]);
                                    
                xTest += x[i]*l;
            }
            
            xGuess = xTest;

            w = newton(xTest, z);        
            
            if (w.size() == 0) {
                std::cout << "opps, monotocity violation " << nomega;
                nomega = o[0] + 0.5*(nomega - o[0]);
                std::cout << " new " << nomega << std::endl;
            }
        } while (w.size() == 0);
        
        std::cout << "start norm " << nomega << " " << norm_2(xGuess - xTest) << std::endl;
        
        if (greaterThan(xTest, x[0]))
            std::cout << "wrong direction" << std::endl;
        
        for (int i=std::min(maxOrder-1, order); i > 0; --i) {
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
